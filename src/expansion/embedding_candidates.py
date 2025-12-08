# src/expansion/embedding_candidates.py

"""
Embedding-based Candidate Expansion Module

Extracts expansion candidates using semantic embeddings (Sentence-BERT).
Finds nearest neighbor terms in embedding space for query expansion.

Following paper Section 3.2:
  "We compute semantic neighbors of the query using pre-trained embeddings.
   We encode the query and retrieve the top-30 nearest neighbors by cosine
   similarity in the embedding space."

Usage:
    from msmeqe.expansion.embedding_candidates import EmbeddingCandidateExtractor
    from msmeqe.reranking.semantic_encoder import SemanticEncoder

    # Initialize with pre-built vocabulary
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    extractor = EmbeddingCandidateExtractor(
        encoder=encoder,
        vocab_path="data/vocab_embeddings.pkl"
    )

    # Extract candidates
    candidates = extractor.extract_candidates(
        query_text="transformer architecture attention mechanism",
        k=30
    )
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np

from msmeqe.reranking.semantic_encoder import SemanticEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VocabularyEmbeddings:
    """
    Pre-computed vocabulary embeddings for efficient nearest neighbor search.

    Fields:
        terms: List of vocabulary terms (aligned with embeddings)
        embeddings: Numpy array of shape (V, d) with L2-normalized embeddings
        metadata: Optional metadata (e.g., term frequencies, source info)
    """
    terms: List[str]
    embeddings: np.ndarray
    metadata: Optional[Dict[str, any]] = None

    def __post_init__(self):
        """Validate that terms and embeddings are aligned."""
        if len(self.terms) != self.embeddings.shape[0]:
            raise ValueError(
                f"Terms length ({len(self.terms)}) doesn't match "
                f"embeddings shape ({self.embeddings.shape[0]})"
            )

        # Ensure embeddings are L2-normalized
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-4):
            logger.warning("Embeddings not L2-normalized, normalizing now")
            self.embeddings = self.embeddings / (norms + 1e-12)


# ---------------------------------------------------------------------------
# Main Embedding Candidate Extractor
# ---------------------------------------------------------------------------

class EmbeddingCandidateExtractor:
    """
    Extract expansion candidates using semantic embeddings.

    This class:
      1. Loads a pre-computed vocabulary with embeddings
      2. Encodes queries using Sentence-BERT
      3. Finds k-nearest neighbors in embedding space
      4. Returns ranked candidates by cosine similarity

    The extractor follows the methodology in Section 3.2 of the paper:
      - Use pre-trained embeddings (Sentence-BERT for consistency with reranking)
      - Encode query as single embedding
      - Retrieve top-k nearest neighbors by cosine similarity

    Note: This uses Sentence-BERT instead of Word2Vec (as mentioned in the paper)
    for consistency with the neural reranking stage (Section 3.7).
    """

    def __init__(
            self,
            encoder: SemanticEncoder,
            vocab_path: Optional[str] = None,
            vocab_terms: Optional[List[str]] = None,
            vocab_embeddings: Optional[np.ndarray] = None,
            max_candidates: int = 30,
            min_similarity: float = 0.0,
    ):
        """
        Initialize embedding candidate extractor.

        Args:
            encoder: SemanticEncoder instance (Sentence-BERT wrapper)
            vocab_path: Path to pickled VocabularyEmbeddings object
            vocab_terms: Alternative: provide terms directly
            vocab_embeddings: Alternative: provide embeddings directly (V, d)
            max_candidates: Maximum number of candidates to return
            min_similarity: Minimum cosine similarity threshold
        """
        self.encoder = encoder
        self.max_candidates = max_candidates
        self.min_similarity = min_similarity

        # Load or set vocabulary
        if vocab_path:
            self._load_vocabulary(vocab_path)
        elif vocab_terms is not None and vocab_embeddings is not None:
            self.vocab = VocabularyEmbeddings(
                terms=vocab_terms,
                embeddings=vocab_embeddings,
            )
        else:
            # No vocabulary provided yet (can be set later)
            self.vocab = None
            logger.warning(
                "No vocabulary provided. Call set_vocabulary() or load_vocabulary() "
                "before extracting candidates."
            )

        if self.vocab:
            logger.info(
                f"Initialized EmbeddingCandidateExtractor: "
                f"vocab_size={len(self.vocab.terms)}, "
                f"embedding_dim={self.vocab.embeddings.shape[1]}, "
                f"max_candidates={max_candidates}"
            )

    def _load_vocabulary(self, vocab_path: str) -> None:
        """
        Load pre-computed vocabulary embeddings from pickle file.

        Args:
            vocab_path: Path to pickled VocabularyEmbeddings object
        """
        vocab_path = Path(vocab_path)

        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        logger.info(f"Loading vocabulary from {vocab_path}")

        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        if not isinstance(self.vocab, VocabularyEmbeddings):
            # Handle legacy format: dict with 'terms' and 'embeddings' keys
            if isinstance(self.vocab, dict):
                self.vocab = VocabularyEmbeddings(
                    terms=self.vocab['terms'],
                    embeddings=self.vocab['embeddings'],
                    metadata=self.vocab.get('metadata'),
                )
            else:
                raise ValueError(
                    f"Unexpected vocabulary format: {type(self.vocab)}"
                )

        logger.info(
            f"Loaded vocabulary: {len(self.vocab.terms)} terms, "
            f"dim={self.vocab.embeddings.shape[1]}"
        )

    def set_vocabulary(
            self,
            terms: List[str],
            embeddings: np.ndarray,
            metadata: Optional[Dict] = None,
    ) -> None:
        """
        Set vocabulary directly (alternative to loading from file).

        Args:
            terms: List of vocabulary terms
            embeddings: Numpy array of embeddings (V, d)
            metadata: Optional metadata dict
        """
        self.vocab = VocabularyEmbeddings(
            terms=terms,
            embeddings=embeddings,
            metadata=metadata,
        )
        logger.info(f"Set vocabulary: {len(terms)} terms")

    def extract_candidates(
            self,
            query_text: str,
            k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Extract embedding-based expansion candidates for a query.

        Args:
            query_text: The query string
            k: Number of candidates to return (default: self.max_candidates)

        Returns:
            List of (term, cosine_similarity) tuples, sorted by similarity descending
        """
        if self.vocab is None:
            raise ValueError(
                "No vocabulary loaded. Call load_vocabulary() or set_vocabulary() first."
            )

        if k is None:
            k = self.max_candidates

        # Encode query
        query_emb = self.encoder.encode([query_text])[0]  # (d,)

        # Compute cosine similarities with all vocab terms
        # Both query_emb and vocab.embeddings are already L2-normalized
        similarities = self.vocab.embeddings @ query_emb  # (V,)

        # Filter by minimum similarity
        if self.min_similarity > 0:
            mask = similarities >= self.min_similarity
            valid_indices = np.where(mask)[0]

            if len(valid_indices) == 0:
                logger.debug(
                    f"No candidates above min_similarity={self.min_similarity} "
                    f"for query: {query_text[:50]}"
                )
                return []

            similarities = similarities[valid_indices]
            terms = [self.vocab.terms[i] for i in valid_indices]
        else:
            terms = self.vocab.terms
            valid_indices = np.arange(len(terms))

        # Get top-k by similarity
        k = min(k, len(similarities))
        top_k_indices = np.argpartition(-similarities, k - 1)[:k]
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]

        # Build result
        result = [
            (terms[i], float(similarities[i]))
            for i in top_k_indices
        ]

        logger.debug(
            f"Extracted {len(result)} embedding candidates for: {query_text[:50]}"
        )

        return result

    def extract_candidates_filtered(
            self,
            query_text: str,
            k: Optional[int] = None,
            exclude_query_terms: bool = True,
            min_term_length: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Extract candidates with additional filtering.

        Args:
            query_text: Query string
            k: Number of candidates
            exclude_query_terms: If True, remove terms that appear in query
            min_term_length: Minimum character length for terms

        Returns:
            Filtered list of (term, similarity) tuples
        """
        # Get raw candidates (request more to account for filtering)
        raw_k = k * 2 if k else self.max_candidates * 2
        candidates = self.extract_candidates(query_text, k=raw_k)

        # Apply filters
        filtered = []
        query_lower = query_text.lower()
        query_tokens = set(query_lower.split())

        for term, score in candidates:
            # Filter by length
            if len(term) < min_term_length:
                continue

            # Filter query terms
            if exclude_query_terms:
                term_lower = term.lower()
                # Skip if term is in query or is a query token
                if term_lower in query_lower or term_lower in query_tokens:
                    continue

            filtered.append((term, score))

            # Stop when we have enough
            if k and len(filtered) >= k:
                break

        return filtered[:k] if k else filtered

    def get_query_expansion_vector(
            self,
            query_text: str,
            k: int = 10,
            weighted: bool = True,
    ) -> np.ndarray:
        """
        Get a weighted average embedding of query + expansion terms.

        This can be used as an enhanced query representation.

        Args:
            query_text: Query string
            k: Number of expansion terms
            weighted: If True, weight by cosine similarity

        Returns:
            Numpy array of shape (d,) representing enhanced query
        """
        if self.vocab is None:
            raise ValueError("No vocabulary loaded")

        # Get query embedding
        query_emb = self.encoder.encode([query_text])[0]

        # Get expansion candidates
        candidates = self.extract_candidates(query_text, k=k)

        if not candidates:
            # No expansion, return original query
            return query_emb

        # Get embeddings for expansion terms
        expansion_terms = [term for term, _ in candidates]
        term_indices = [
            self.vocab.terms.index(term)
            for term in expansion_terms
            if term in self.vocab.terms
        ]

        if not term_indices:
            return query_emb

        term_embs = self.vocab.embeddings[term_indices]  # (k, d)

        if weighted:
            # Weight by similarity scores
            weights = np.array([score for _, score in candidates[:len(term_indices)]])
            weights = weights / (weights.sum() + 1e-12)

            # Weighted average of expansion terms
            expansion_vec = (term_embs.T @ weights)  # (d,)
        else:
            # Simple average
            expansion_vec = term_embs.mean(axis=0)

        # Combine query and expansion (equal weight)
        combined = 0.5 * query_emb + 0.5 * expansion_vec

        # Re-normalize
        combined = combined / (np.linalg.norm(combined) + 1e-12)

        return combined


# ---------------------------------------------------------------------------
# Vocabulary building utilities
# ---------------------------------------------------------------------------

def build_vocabulary_from_collection(
        encoder: SemanticEncoder,
        terms: List[str],
        batch_size: int = 1000,
        output_path: Optional[str] = None,
        min_term_length: int = 3,
        max_terms: Optional[int] = None,
) -> VocabularyEmbeddings:
    """
    Build vocabulary embeddings from a list of terms.

    This is useful for creating the vocabulary file that the extractor loads.

    Args:
        encoder: SemanticEncoder instance
        terms: List of unique terms to embed
        batch_size: Batch size for encoding
        output_path: If provided, save vocabulary to this path
        min_term_length: Minimum character length for terms
        max_terms: Maximum number of terms to include

    Returns:
        VocabularyEmbeddings object
    """
    logger.info(f"Building vocabulary from {len(terms)} terms")

    # Filter terms
    filtered_terms = [
        t for t in terms
        if len(t) >= min_term_length
    ]

    logger.info(f"After filtering: {len(filtered_terms)} terms")

    # Truncate if needed
    if max_terms and len(filtered_terms) > max_terms:
        logger.info(f"Truncating to {max_terms} terms")
        filtered_terms = filtered_terms[:max_terms]

    # Encode in batches
    logger.info("Encoding terms...")
    all_embeddings = []

    for i in range(0, len(filtered_terms), batch_size):
        batch = filtered_terms[i:i + batch_size]
        batch_embs = encoder.encode(batch)
        all_embeddings.append(batch_embs)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Encoded {i + len(batch)}/{len(filtered_terms)} terms")

    embeddings = np.vstack(all_embeddings)
    logger.info(f"Encoding complete: shape={embeddings.shape}")

    # Create vocabulary object
    vocab = VocabularyEmbeddings(
        terms=filtered_terms,
        embeddings=embeddings,
        metadata={
            'encoder_model': encoder.model.model_name if hasattr(encoder.model, 'model_name') else 'unknown',
            'num_terms': len(filtered_terms),
            'embedding_dim': embeddings.shape[1],
        }
    )

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving vocabulary to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(vocab, f)

        logger.info(f"Vocabulary saved ({output_path.stat().st_size / 1e6:.1f} MB)")

    return vocab


def build_vocabulary_from_index(
        encoder: SemanticEncoder,
        index_path: str,
        output_path: str,
        max_terms: int = 50000,
        min_df: int = 5,
        batch_size: int = 1000,
) -> VocabularyEmbeddings:
    """
    Build vocabulary from a Lucene index (extract terms with min doc frequency).

    Args:
        encoder: SemanticEncoder instance
        index_path: Path to Lucene index
        output_path: Where to save vocabulary
        max_terms: Maximum number of terms
        min_df: Minimum document frequency
        batch_size: Encoding batch size

    Returns:
        VocabularyEmbeddings object
    """
    from msmeqe.utils.lucene_utils import get_lucene_classes

    logger.info(f"Building vocabulary from Lucene index: {index_path}")

    # Get Lucene classes
    classes = get_lucene_classes()
    DirectoryReader = classes['IndexReader']
    FSDirectory = classes['FSDirectory']
    Path = classes['Path']

    # Open index
    directory = FSDirectory.open(Path.get(index_path))
    reader = DirectoryReader.open(directory)

    # Extract terms with sufficient document frequency
    logger.info(f"Extracting terms with df >= {min_df}")
    terms_list = []

    fields = reader.getTermVectors(0).iterator()
    field_name = "contents"  # Assume standard field name

    terms_enum = reader.terms(field_name).iterator()
    bytes_ref = terms_enum.next()

    while bytes_ref is not None:
        term_text = bytes_ref.utf8ToString()
        df = terms_enum.docFreq()

        if df >= min_df and len(term_text) >= 3:
            terms_list.append((term_text, df))

        bytes_ref = terms_enum.next()

        if len(terms_list) % 10000 == 0:
            logger.info(f"  Extracted {len(terms_list)} terms so far")

    reader.close()

    logger.info(f"Extracted {len(terms_list)} terms")

    # Sort by document frequency (descending) and take top-k
    terms_list.sort(key=lambda x: x[1], reverse=True)
    terms_list = terms_list[:max_terms]

    terms = [t for t, _ in terms_list]

    # Build vocabulary
    return build_vocabulary_from_collection(
        encoder=encoder,
        terms=terms,
        batch_size=batch_size,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# CLI for building vocabulary
# ---------------------------------------------------------------------------

def _main_cli():
    """
    CLI for building vocabulary embeddings.

    Example:
        # From term list
        python -m msmeqe.expansion.embedding_candidates \\
            --terms-file data/terms.txt \\
            --output data/vocab_embeddings.pkl \\
            --max-terms 50000

        # From Lucene index
        python -m msmeqe.expansion.embedding_candidates \\
            --from-index data/index \\
            --output data/vocab_embeddings.pkl \\
            --max-terms 50000 \\
            --min-df 5
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Build vocabulary embeddings for expansion"
    )
    parser.add_argument(
        "--terms-file",
        type=str,
        help="Text file with one term per line",
    )
    parser.add_argument(
        "--from-index",
        type=str,
        help="Build vocabulary from Lucene index",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for vocabulary pickle file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-BERT model name",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=50000,
        help="Maximum number of terms",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="Minimum document frequency (for --from-index)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Encoding batch size",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # Initialize encoder
    logger.info(f"Loading encoder: {args.model_name}")
    encoder = SemanticEncoder(model_name=args.model_name)

    # Build vocabulary
    if args.from_index:
        vocab = build_vocabulary_from_index(
            encoder=encoder,
            index_path=args.from_index,
            output_path=args.output,
            max_terms=args.max_terms,
            min_df=args.min_df,
            batch_size=args.batch_size,
        )
    elif args.terms_file:
        # Load terms from file
        with open(args.terms_file, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]

        vocab = build_vocabulary_from_collection(
            encoder=encoder,
            terms=terms,
            batch_size=args.batch_size,
            output_path=args.output,
            max_terms=args.max_terms,
        )
    else:
        parser.error("Must provide either --terms-file or --from-index")

    # Print summary
    print("\nVocabulary Summary:")
    print(f"  Terms: {len(vocab.terms)}")
    print(f"  Embedding dim: {vocab.embeddings.shape[1]}")
    print(f"  File: {args.output}")
    print(f"  Size: {Path(args.output).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    _main_cli()