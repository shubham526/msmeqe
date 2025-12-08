# scripts/build_sbert_vocab.py

"""
Build SBERT vocabulary embeddings for embedding-based expansion.

This script creates a vocabulary of terms and their Sentence-BERT embeddings
to be used for semantic neighbor retrieval in MS-MEQE expansion.

Two modes:
  1. From term list: Encode a list of terms (e.g., most frequent terms)
  2. From Lucene index: Extract terms from index with min DF threshold

The output is a VocabularyEmbeddings object saved as pickle:
  - terms: List[str] - vocabulary terms
  - embeddings: np.ndarray (V, d) - L2-normalized embeddings
  - metadata: dict - info about vocabulary creation

Usage:
    # Mode 1: From term list
    python scripts/build_sbert_vocab.py \\
        --from-terms data/term_list.txt \\
        --output data/vocab_embeddings.pkl \\
        --model-name sentence-transformers/all-MiniLM-L6-v2 \\
        --max-terms 50000

    # Mode 2: From Lucene index
    python scripts/build_sbert_vocab.py \\
        --from-index data/msmarco_index \\
        --output data/vocab_embeddings.pkl \\
        --model-name sentence-transformers/all-MiniLM-L6-v2 \\
        --max-terms 50000 \\
        --min-df 5 \\
        --field contents

    # Mode 3: From collection (encode documents then extract vocab)
    python scripts/build_sbert_vocab.py \\
        --from-collection msmarco-passage \\
        --output data/vocab_embeddings.pkl \\
        --max-terms 50000
"""

import logging
import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional, Set
import pickle
from collections import Counter
from tqdm import tqdm

import numpy as np

from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.embedding_candidates import VocabularyEmbeddings
from src.utils.lucene_utils import get_lucene_classes

logger = logging.getLogger(__name__)


class VocabularyBuilder:
    """
    Build vocabulary embeddings from various sources.

    Supports:
      - Term lists (text files)
      - Lucene indices
      - Document collections
    """

    def __init__(
            self,
            encoder: SemanticEncoder,
            max_terms: int = 50000,
    ):
        """
        Initialize vocabulary builder.

        Args:
            encoder: SemanticEncoder instance
            max_terms: Maximum vocabulary size
        """
        self.encoder = encoder
        self.max_terms = max_terms

        logger.info(f"Initialized VocabularyBuilder: max_terms={max_terms}")

    def build_from_terms(
            self,
            terms: List[str],
            batch_size: int = 512,
    ) -> VocabularyEmbeddings:
        """
        Build vocabulary from a list of terms.

        Args:
            terms: List of term strings
            batch_size: Encoding batch size

        Returns:
            VocabularyEmbeddings object
        """
        logger.info(f"Building vocabulary from {len(terms)} terms")

        # Limit to max_terms
        if len(terms) > self.max_terms:
            logger.info(f"Limiting to {self.max_terms} terms")
            terms = terms[:self.max_terms]

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        logger.info(f"After deduplication: {len(unique_terms)} unique terms")

        # Encode terms in batches
        logger.info("Encoding terms with Sentence-BERT...")
        embeddings = []

        for i in tqdm(range(0, len(unique_terms), batch_size), desc="Encoding"):
            batch = unique_terms[i:i + batch_size]
            batch_embs = self.encoder.encode(batch)
            embeddings.append(batch_embs)

        embeddings = np.vstack(embeddings)

        # L2-normalize
        logger.info("L2-normalizing embeddings...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-12)

        # Create vocabulary object
        vocab = VocabularyEmbeddings(
            terms=unique_terms,
            embeddings=embeddings,
            metadata={
                'source': 'term_list',
                'num_terms': len(unique_terms),
                'embedding_dim': embeddings.shape[1],
                'model_name': self.encoder.model_name,
            }
        )

        logger.info(f"Built vocabulary: {len(unique_terms)} terms, dim={embeddings.shape[1]}")

        return vocab

    def build_from_term_file(
            self,
            term_file: str,
            batch_size: int = 512,
    ) -> VocabularyEmbeddings:
        """
        Build vocabulary from a text file (one term per line).

        Args:
            term_file: Path to term list file
            batch_size: Encoding batch size

        Returns:
            VocabularyEmbeddings object
        """
        logger.info(f"Loading terms from {term_file}")

        with open(term_file, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(terms)} terms from file")

        return self.build_from_terms(terms, batch_size=batch_size)

    def build_from_index(
            self,
            index_path: str,
            field: str = "contents",
            min_df: int = 5,
            max_df_ratio: float = 0.5,
            batch_size: int = 512,
    ) -> VocabularyEmbeddings:
        """
        Build vocabulary from Lucene index.

        Extracts terms from the specified field with:
          - Document frequency >= min_df
          - Document frequency <= max_df_ratio * num_docs

        Terms are sorted by document frequency (descending).

        Args:
            index_path: Path to Lucene index
            field: Field name to extract terms from
            min_df: Minimum document frequency
            max_df_ratio: Maximum document frequency ratio (to filter stop words)
            batch_size: Encoding batch size

        Returns:
            VocabularyEmbeddings object
        """
        logger.info(f"Building vocabulary from Lucene index: {index_path}")
        logger.info(f"Field: {field}, min_df: {min_df}, max_df_ratio: {max_df_ratio}")

        # Initialize Lucene
        classes = get_lucene_classes()

        DirectoryReader = classes['IndexReader']
        FSDirectory = classes['FSDirectory']
        Path = classes['Path']

        directory = FSDirectory.open(Path.get(index_path))
        reader = DirectoryReader.open(directory)

        num_docs = reader.numDocs()
        max_df = int(max_df_ratio * num_docs)

        logger.info(f"Index: {num_docs} documents, max_df: {max_df}")

        # Extract terms
        logger.info("Extracting terms from index...")
        terms_with_df = []

        terms = reader.terms(field)
        if terms is None:
            raise ValueError(f"Field '{field}' not found in index")

        terms_enum = terms.iterator()

        count = 0
        while terms_enum.next():
            term_bytes = terms_enum.term()
            try:
                term_text = term_bytes.utf8ToString()

                # FIX: Handle multi-word terms typically stored with underscores
                # e.g., "neural_network" -> "neural network"
                if '_' in term_text:
                    term_text = term_text.replace('_', ' ')

                term_text = term_text.strip()
                if not term_text:
                    continue

            except Exception as e:
                # Skip invalid utf-8 sequences
                continue

            df = terms_enum.docFreq()

            if min_df <= df <= max_df:
                terms_with_df.append((term_text, df))
                count += 1

                if count % 10000 == 0:
                    logger.info(f"  Extracted {count} terms...")

        logger.info(f"Extracted {len(terms_with_df)} terms matching criteria")

        # Sort by DF (descending)
        terms_with_df.sort(key=lambda x: x[1], reverse=True)

        # Limit to max_terms
        if len(terms_with_df) > self.max_terms:
            logger.info(f"Limiting to top {self.max_terms} by DF")
            terms_with_df = terms_with_df[:self.max_terms]

        # Extract just the terms
        terms = [term for term, df in terms_with_df]

        # Close reader
        reader.close()

        # Build vocabulary
        vocab = self.build_from_terms(terms, batch_size=batch_size)

        # Update metadata
        vocab.metadata.update({
            'source': 'lucene_index',
            'index_path': index_path,
            'field': field,
            'min_df': min_df,
            'max_df': max_df,
            'num_docs': num_docs,
        })

        return vocab

    def build_from_collection(
            self,
            collection_name: str,
            batch_size: int = 512,
    ) -> VocabularyEmbeddings:
        """
        Build vocabulary from a document collection.

        Extracts most frequent unigrams and bigrams from documents.

        Args:
            collection_name: Collection name (e.g., 'msmarco-passage')
            batch_size: Encoding batch size

        Returns:
            VocabularyEmbeddings object
        """
        logger.info(f"Building vocabulary from collection: {collection_name}")

        # Import here to avoid dependency
        try:
            from ir_datasets import load as ir_load
        except ImportError:
            raise ImportError(
                "ir_datasets required for collection loading. "
                "Install with: pip install ir-datasets"
            )

        dataset = ir_load(collection_name)

        # Extract terms from documents
        logger.info("Extracting terms from documents...")
        term_counts = Counter()

        for i, doc in enumerate(tqdm(dataset.docs_iter(), desc="Processing docs")):
            if i >= 100000:  # Limit for efficiency
                logger.info(f"Processed {i} documents, stopping")
                break

            # Tokenize (simple whitespace)
            tokens = doc.text.lower().split()

            # Unigrams
            term_counts.update(tokens)

            # Bigrams
            bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
            term_counts.update(bigrams)

        logger.info(f"Extracted {len(term_counts)} unique terms")

        # Get most common terms
        most_common = term_counts.most_common(self.max_terms)
        terms = [term for term, count in most_common]

        logger.info(f"Selected top {len(terms)} terms by frequency")

        # Build vocabulary
        vocab = self.build_from_terms(terms, batch_size=batch_size)

        # Update metadata
        vocab.metadata.update({
            'source': 'document_collection',
            'collection_name': collection_name,
            'num_docs_processed': min(i + 1, 100000),
        })

        return vocab


def save_vocabulary(vocab: VocabularyEmbeddings, output_path: str):
    """
    Save vocabulary embeddings to disk.

    Args:
        vocab: VocabularyEmbeddings object
        output_path: Output file path (.pkl)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving vocabulary to {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata as JSON for inspection
    import json
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(vocab.metadata, f, indent=2)

    # Save term list as text for inspection
    terms_path = output_path.with_suffix('.txt')
    with open(terms_path, 'w', encoding='utf-8') as f:
        for term in vocab.terms:
            f.write(f"{term}\n")

    logger.info(f"Vocabulary saved:")
    logger.info(f"  Embeddings: {output_path}")
    logger.info(f"  Metadata:   {metadata_path}")
    logger.info(f"  Terms:      {terms_path}")

    # Print statistics
    logger.info("\nVocabulary Statistics:")
    logger.info(f"  Number of terms: {len(vocab.terms)}")
    logger.info(f"  Embedding dim:   {vocab.embeddings.shape[1]}")
    logger.info(f"  Model:           {vocab.metadata.get('model_name', 'unknown')}")
    logger.info(f"  Source:          {vocab.metadata.get('source', 'unknown')}")

    # File size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  File size:       {file_size_mb:.2f} MB")


def main():
    """Main entry point for vocabulary building."""
    parser = argparse.ArgumentParser(
        description="Build SBERT vocabulary embeddings for MS-MEQE expansion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--from-terms",
        type=str,
        help="Path to term list file (one term per line)",
    )
    source_group.add_argument(
        "--from-index",
        type=str,
        help="Path to Lucene index",
    )
    source_group.add_argument(
        "--from-collection",
        type=str,
        help="Collection name (e.g., 'msmarco-passage')",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for vocabulary embeddings (.pkl)",
    )

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-BERT model name",
    )

    # Vocabulary size
    parser.add_argument(
        "--max-terms",
        type=int,
        default=50000,
        help="Maximum vocabulary size",
    )

    # Index-specific options
    parser.add_argument(
        "--field",
        type=str,
        default="contents",
        help="Field name for index mode",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="Minimum document frequency for index mode",
    )
    parser.add_argument(
        "--max-df-ratio",
        type=float,
        default=0.5,
        help="Maximum DF ratio (to filter stop words) for index mode",
    )

    # Encoding
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Encoding batch size",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    )

    logger.info("=" * 60)
    logger.info("BUILDING SBERT VOCABULARY")
    logger.info("=" * 60)

    # Initialize encoder
    logger.info(f"Loading Sentence-BERT model: {args.model_name}")
    encoder = SemanticEncoder(model_name=args.model_name)

    # Initialize builder
    builder = VocabularyBuilder(
        encoder=encoder,
        max_terms=args.max_terms,
    )

    # Build vocabulary based on source
    if args.from_terms:
        logger.info(f"Source: Term list ({args.from_terms})")
        vocab = builder.build_from_term_file(
            term_file=args.from_terms,
            batch_size=args.batch_size,
        )

    elif args.from_index:
        logger.info(f"Source: Lucene index ({args.from_index})")
        vocab = builder.build_from_index(
            index_path=args.from_index,
            field=args.field,
            min_df=args.min_df,
            max_df_ratio=args.max_df_ratio,
            batch_size=args.batch_size,
        )

    elif args.from_collection:
        logger.info(f"Source: Document collection ({args.from_collection})")
        vocab = builder.build_from_collection(
            collection_name=args.from_collection,
            batch_size=args.batch_size,
        )

    # Save vocabulary
    save_vocabulary(vocab, args.output)

    logger.info("\n" + "=" * 60)
    logger.info("VOCABULARY BUILDING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nUsage in MS-MEQE:")
    logger.info(f"  from msmeqe.expansion.embedding_candidates import EmbeddingCandidateExtractor")
    logger.info(f"  extractor = EmbeddingCandidateExtractor(")
    logger.info(f"      encoder=encoder,")
    logger.info(f"      vocab_path='{args.output}',")
    logger.info(f"  )")
    logger.info(f"  candidates = extractor.extract_candidates('neural networks', k=30)")


if __name__ == "__main__":
    main()