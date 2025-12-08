# src/expansion/candidate_extraction_pipeline.py

"""
Complete pipeline for extracting multi-source candidates with all required stats.

This module bridges:
  1. Source-specific extractors (RM3, KB, Embeddings)
  2. MS-MEQE expansion model (which needs CandidateTerm objects)

It handles:
  - Multi-source candidate extraction
  - Term statistics (DF, CF) from Lucene
  - Pseudo-document statistics (TF, coverage)
  - Query statistics for budget prediction
  - Pseudo-document centroid computation

Usage:
    from msmeqe.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
    from msmeqe.reranking.semantic_encoder import SemanticEncoder

    encoder = SemanticEncoder()
    extractor = MultiSourceCandidateExtractor(
        index_path="data/msmarco_index",
        encoder=encoder,
    )

    # Extract candidates
    candidates = extractor.extract_all_candidates(
        query_text="neural networks",
        query_id="q123",
    )

    # Compute query stats
    query_stats = extractor.compute_query_stats("neural networks")

    # Compute pseudo-doc centroid
    centroid = extractor.compute_pseudo_centroid("neural networks")
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re

import numpy as np

from src.expansion.rm_expansion import LuceneRM3Scorer
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.reranking.semantic_encoder import SemanticEncoder
from src.expansion.msmeqe_expansion import CandidateTerm
from src.utils.lucene_utils import get_lucene_classes

logger = logging.getLogger(__name__)


class MultiSourceCandidateExtractor:
    """
    Extract candidates from all three sources with complete statistics.

    This is the bridge between:
      1. Source-specific extractors (RM3, KB, Embeddings)
      2. MS-MEQE expansion model (which needs CandidateTerm objects)

    Extracts candidates with:
      - RM3 scores (for docs source)
      - Entity linking confidence (for KB source)
      - Cosine similarity (for embedding source)
      - Term statistics: DF, CF (from Lucene index)
      - Pseudo-doc statistics: TF, coverage
    """

    def __init__(
            self,
            index_path: str,
            encoder: SemanticEncoder,
            kb_extractor: Optional[KBCandidateExtractor] = None,
            emb_extractor: Optional[EmbeddingCandidateExtractor] = None,
            n_docs_rm3: int = 30,
            n_kb: int = 30,
            n_emb: int = 30,
            n_pseudo_docs: int = 10,
    ):
        """
        Initialize multi-source extractor.

        Args:
            index_path: Path to Lucene index (for RM3 and term stats)
            encoder: SemanticEncoder for embeddings
            kb_extractor: Optional KBCandidateExtractor
            emb_extractor: Optional EmbeddingCandidateExtractor
            n_docs_rm3: Number of RM3 terms
            n_kb: Number of KB terms
            n_emb: Number of embedding terms
            n_pseudo_docs: Number of pseudo-relevant documents
        """
        self.encoder = encoder
        self.index_path = index_path

        # Initialize RM3 scorer
        logger.info(f"Initializing RM3 scorer with index: {index_path}")
        self.rm3_scorer = LuceneRM3Scorer(
            index_dir=index_path,
            field="contents",
            mu=1000.0,
            orig_query_weight=0.5,
        )

        # Store extractors
        self.kb_extractor = kb_extractor
        self.emb_extractor = emb_extractor

        self.n_docs_rm3 = n_docs_rm3
        self.n_kb = n_kb
        self.n_emb = n_emb
        self.n_pseudo_docs = n_pseudo_docs

        # Initialize Lucene for term statistics
        self._init_lucene_stats()

        logger.info(
            f"Initialized MultiSourceCandidateExtractor: "
            f"RM3={n_docs_rm3}, KB={n_kb}, Emb={n_emb}"
        )

    def _init_lucene_stats(self):
        """Initialize Lucene index for term statistics (DF, CF)."""
        try:
            classes = get_lucene_classes()

            self.DirectoryReader = classes['IndexReader']
            self.FSDirectory = classes['FSDirectory']
            self.Path = classes['Path']
            self.IndexSearcher = classes['IndexSearcher']
            self.QueryParser = classes['QueryParser']
            self.EnglishAnalyzer = classes['EnglishAnalyzer']
            self.BooleanQueryBuilder = classes['BooleanQueryBuilder']
            self.BooleanClause = classes['BooleanClause']
            self.Term = classes['Term']
            self.TermQuery = classes['TermQuery']
            self.BytesRef = classes['BytesRef']

            directory = self.FSDirectory.open(self.Path.get(self.index_path))
            self.lucene_reader = self.DirectoryReader.open(directory)
            self.lucene_searcher = self.IndexSearcher(self.lucene_reader.getContext())
            self.analyzer = self.EnglishAnalyzer()
            self.field_name = "contents"

            # Get collection size
            self.collection_size = self.lucene_reader.numDocs()

            logger.info(
                f"Lucene index opened: {self.collection_size} documents, "
                f"field='{self.field_name}'"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Lucene: {e}")
            raise

    def extract_all_candidates(
            self,
            query_text: str,
            query_id: Optional[str] = None,
            kb_override: Optional[List[Dict]] = None,  # ← ADD THIS PARAMETER
    ) -> List[CandidateTerm]:
        """
        Extract candidates from all sources with complete statistics.

        Args:
            query_text: Query string
            query_id: Optional query ID (for KB lookup)
            kb_override: Optional precomputed KB candidates (skips KB extraction)
                        List of dicts with keys: 'term', 'confidence', 'rank'

        Returns:
            List of CandidateTerm objects with all fields populated
        """
        all_candidates = []

        # Get pseudo-relevant documents (shared across all sources)
        pseudo_docs = self._get_pseudo_relevant_docs(query_text, self.n_pseudo_docs)

        # === 1. DOCS SOURCE (RM3) ===
        logger.debug(f"Extracting RM3 candidates for: {query_text[:50]}")

        try:
            rm3_terms = self.rm3_scorer.expand(
                query_str=query_text,
                n_docs=self.n_pseudo_docs,
                n_terms=self.n_docs_rm3,
                use_rm3=True,
            )

            for rank, (term, rm3_score) in enumerate(rm3_terms, start=1):
                df, cf = self._get_term_stats(term)
                tf_pseudo = self._compute_tf_pseudo(term, pseudo_docs)
                coverage = self._compute_coverage(term, pseudo_docs)

                all_candidates.append(CandidateTerm(
                    term=term,
                    source="docs",
                    rm3_score=rm3_score,
                    tf_pseudo=tf_pseudo,
                    coverage_pseudo=coverage,
                    df=df,
                    cf=cf,
                    native_rank=rank,
                    native_score=rm3_score,
                ))

            logger.debug(f"Extracted {len(rm3_terms)} RM3 candidates")

        except Exception as e:
            logger.warning(f"RM3 extraction failed: {e}")

        # === 2. KB SOURCE (WITH OVERRIDE SUPPORT) ===
        if kb_override is not None:
            # Use precomputed KB candidates
            logger.debug(f"Using {len(kb_override)} precomputed KB candidates")

            try:
                for kb_cand_dict in kb_override[:self.n_kb]:
                    term = kb_cand_dict['term']
                    confidence = kb_cand_dict.get('confidence', 1.0)
                    rank = kb_cand_dict.get('rank', 1)

                    df, cf = self._get_term_stats(term)

                    all_candidates.append(CandidateTerm(
                        term=term,
                        source="kb",
                        rm3_score=0.0,
                        tf_pseudo=0.0,
                        coverage_pseudo=0.0,
                        df=df,
                        cf=cf,
                        native_rank=rank,
                        native_score=confidence,
                    ))

                logger.debug(f"Added {len(kb_override[:self.n_kb])} precomputed KB candidates")

            except Exception as e:
                logger.warning(f"Failed to process precomputed KB candidates: {e}")

        elif self.kb_extractor:
            # Extract KB candidates on-the-fly (original behavior)
            logger.debug(f"Extracting KB candidates on-the-fly")

            try:
                kb_candidates = self.kb_extractor.extract_candidates_with_metadata(
                    query_text=query_text,
                    query_id=query_id,
                )

                for rank, kb_cand in enumerate(kb_candidates[:self.n_kb], start=1):
                    df, cf = self._get_term_stats(kb_cand.term)

                    all_candidates.append(CandidateTerm(
                        term=kb_cand.term,
                        source="kb",
                        rm3_score=0.0,
                        tf_pseudo=0.0,
                        coverage_pseudo=0.0,
                        df=df,
                        cf=cf,
                        native_rank=rank,
                        native_score=kb_cand.confidence,
                    ))

                logger.debug(f"Extracted {len(kb_candidates)} KB candidates")

            except Exception as e:
                logger.warning(f"KB extraction failed: {e}")

        # === 3. EMBEDDING SOURCE ===
        if self.emb_extractor:
            logger.debug(f"Extracting embedding candidates for: {query_text[:50]}")

            try:
                emb_candidates = self.emb_extractor.extract_candidates(
                    query_text=query_text,
                    k=self.n_emb,
                )

                for rank, (term, cos_sim) in enumerate(emb_candidates, start=1):
                    df, cf = self._get_term_stats(term)

                    all_candidates.append(CandidateTerm(
                        term=term,
                        source="emb",
                        rm3_score=0.0,
                        tf_pseudo=0.0,
                        coverage_pseudo=0.0,
                        df=df,
                        cf=cf,
                        native_rank=rank,
                        native_score=cos_sim,
                    ))

                logger.debug(f"Extracted {len(emb_candidates)} embedding candidates")

            except Exception as e:
                logger.warning(f"Embedding extraction failed: {e}")

        logger.info(f"Total candidates extracted: {len(all_candidates)}")
        return all_candidates

    def _get_pseudo_relevant_docs(
            self,
            query: str,
            n: int = 10
    ) -> List[str]:
        """
        Get top-n pseudo-relevant documents using BM25.

        Args:
            query: Query string
            n: Number of documents to retrieve

        Returns:
            List of document texts
        """
        try:
            # Parse query
            query_parser = self.QueryParser(self.field_name, self.analyzer)
            lucene_query = query_parser.parse(query)

            # Search
            top_docs = self.lucene_searcher.search(lucene_query, n)

            # Extract document texts
            docs = []
            for score_doc in top_docs.scoreDocs:
                doc = self.lucene_searcher.storedFields().document(score_doc.doc)
                doc_text = doc.get(self.field_name)
                if doc_text:
                    docs.append(doc_text)

            logger.debug(f"Retrieved {len(docs)} pseudo-relevant documents")
            return docs

        except Exception as e:
            logger.warning(f"Failed to retrieve pseudo-relevant docs: {e}")
            return []

    def _get_term_stats(self, term: str) -> Tuple[int, int]:
        """
        Get document frequency (DF) and collection frequency (CF) for a term.

        Args:
            term: Term string

        Returns:
            Tuple of (df, cf)
        """
        try:
            # Get terms for the field
            terms = self.lucene_reader.terms(self.field_name)
            if terms is None:
                return 1, 1

            # Get term iterator
            terms_enum = terms.iterator()

            # Create BytesRef for the term
            term_bytes = self.BytesRef(term.lower().encode('utf-8'))

            # Seek to term
            if terms_enum.seekExact(term_bytes):
                df = terms_enum.docFreq()
                cf = terms_enum.totalTermFreq()
                return max(df, 1), max(cf, 1)

            # Term not found
            return 1, 1

        except Exception as e:
            logger.debug(f"Error getting stats for term '{term}': {e}")
            return 1, 1

    def _compute_tf_pseudo(
            self,
            term: str,
            pseudo_docs: List[str]
    ) -> float:
        """
        Compute normalized term frequency in pseudo-relevant documents.

        Args:
            term: Term string
            pseudo_docs: List of document texts

        Returns:
            Normalized TF (term count / total words)
        """
        if not pseudo_docs:
            return 0.0

        term_lower = term.lower()
        total_count = 0
        total_words = 0

        for doc in pseudo_docs:
            # Simple whitespace tokenization
            doc_words = doc.lower().split()
            total_words += len(doc_words)

            # Count term occurrences
            total_count += doc_words.count(term_lower)

        if total_words == 0:
            return 0.0

        return total_count / total_words

    def _compute_coverage(
            self,
            term: str,
            pseudo_docs: List[str]
    ) -> float:
        """
        Compute fraction of pseudo-docs containing the term.

        Args:
            term: Term string
            pseudo_docs: List of document texts

        Returns:
            Coverage fraction [0, 1]
        """
        if not pseudo_docs:
            return 0.0

        term_lower = term.lower()
        count = sum(1 for doc in pseudo_docs if term_lower in doc.lower())

        return count / len(pseudo_docs)

    def compute_pseudo_centroid(self, query_text: str) -> np.ndarray:
        """
        Compute centroid of pseudo-relevant document embeddings.

        Args:
            query_text: Query string

        Returns:
            Centroid embedding (d,)
        """
        pseudo_docs = self._get_pseudo_relevant_docs(query_text, self.n_pseudo_docs)

        if not pseudo_docs:
            # No pseudo-docs, return zero vector
            return np.zeros(self.encoder.get_dim(), dtype=np.float32)

        # Encode documents
        pseudo_embeddings = self.encoder.encode(pseudo_docs)

        # Compute centroid
        centroid = np.mean(pseudo_embeddings, axis=0)

        return centroid

    def compute_query_stats(self, query_text: str) -> Dict[str, float]:
        """
        Compute query-level statistics for budget prediction.

        Returns dict with:
          - clarity: Query clarity score
          - entropy: Entropy over pseudo-doc scores
          - avg_idf: Average IDF of query terms
          - max_idf: Maximum IDF of query terms
          - avg_bm25: Average BM25 score of pseudo-docs
          - var_bm25: Variance of BM25 scores
          - q_len: Query length (number of terms)
          - q_type: Query type ("navigational", "informational", "transactional")

        Args:
            query_text: Query string

        Returns:
            Dictionary of query statistics
        """
        # Tokenize query
        query_tokens = query_text.lower().split()
        q_len = len(query_tokens)

        # === IDF STATISTICS ===
        idfs = []
        for token in query_tokens:
            df, _ = self._get_term_stats(token)
            idf = np.log(self.collection_size / max(df, 1))
            idfs.append(idf)

        avg_idf = float(np.mean(idfs)) if idfs else 0.0
        max_idf = float(np.max(idfs)) if idfs else 0.0

        # === CLARITY ===
        clarity = self._compute_query_clarity(query_text, query_tokens)

        # === ENTROPY (over pseudo-doc retrieval scores) ===
        entropy = self._compute_retrieval_entropy(query_text)

        # === BM25 STATISTICS ===
        avg_bm25, var_bm25 = self._compute_bm25_stats(query_text)

        # === QUERY TYPE ===
        q_type = self._classify_query_type(query_text)

        return {
            'clarity': float(clarity),
            'entropy': float(entropy),
            'avg_idf': float(avg_idf),
            'max_idf': float(max_idf),
            'avg_bm25': float(avg_bm25),
            'var_bm25': float(var_bm25),
            'q_len': int(q_len),
            'q_type': q_type,
        }

    def _compute_query_clarity(
            self,
            query: str,
            query_tokens: List[str]
    ) -> float:
        """
        Compute query clarity score.

        Clarity = sum_t P(t|q) * log(P(t|q) / P(t|C))

        Simplified version using query terms and their collection frequencies.

        Args:
            query: Query string
            query_tokens: List of query tokens

        Returns:
            Clarity score
        """
        if not query_tokens:
            return 0.0

        clarity = 0.0

        for token in query_tokens:
            df, cf = self._get_term_stats(token)

            # P(t|C) = collection frequency / collection size
            p_t_c = cf / (self.collection_size * 100)  # Assume avg doc length ~100

            # P(t|q) = uniform over query terms
            p_t_q = 1.0 / len(query_tokens)

            if p_t_c > 0:
                clarity += p_t_q * np.log(p_t_q / p_t_c)

        return float(clarity)

    def _compute_retrieval_entropy(self, query: str) -> float:
        """
        Compute entropy over pseudo-relevant document scores.

        High entropy = diverse/ambiguous results
        Low entropy = concentrated/clear results

        Args:
            query: Query string

        Returns:
            Entropy value
        """
        try:
            # Parse and search
            query_parser = self.QueryParser(self.field_name, self.analyzer)
            lucene_query = query_parser.parse(query)
            top_docs = self.lucene_searcher.search(lucene_query, self.n_pseudo_docs)

            if not top_docs.scoreDocs:
                return 0.0

            # Get scores
            scores = np.array([sd.score for sd in top_docs.scoreDocs])

            # Normalize to probabilities
            if scores.sum() == 0:
                return 0.0

            probs = scores / scores.sum()

            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-12))

            return float(entropy)

        except Exception as e:
            logger.debug(f"Failed to compute retrieval entropy: {e}")
            return 0.0

    def _compute_bm25_stats(self, query: str) -> Tuple[float, float]:
        """
        Compute BM25 statistics from pseudo-relevant documents.

        Args:
            query: Query string

        Returns:
            Tuple of (avg_bm25, var_bm25)
        """
        try:
            # Parse and search
            query_parser = self.QueryParser(self.field_name, self.analyzer)
            lucene_query = query_parser.parse(query)
            top_docs = self.lucene_searcher.search(lucene_query, self.n_pseudo_docs)

            if not top_docs.scoreDocs:
                return 0.0, 0.0

            # Get scores
            scores = np.array([sd.score for sd in top_docs.scoreDocs])

            avg_bm25 = float(np.mean(scores))
            var_bm25 = float(np.var(scores))

            return avg_bm25, var_bm25

        except Exception as e:
            logger.debug(f"Failed to compute BM25 stats: {e}")
            return 0.0, 0.0

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type using heuristics.

        Returns: "navigational", "informational", or "transactional"

        Heuristics:
          - Navigational: homepage, website, official, login, site
          - Transactional: buy, purchase, price, download, order, shop, cheap
          - Informational: what, how, why, who, when, where, definition
          - Default: informational

        Args:
            query: Query string

        Returns:
            Query type string
        """
        query_lower = query.lower()

        # Navigational indicators
        nav_keywords = [
            'homepage', 'website', 'official', 'login', 'site',
            'main page', 'home page', 'portal'
        ]
        if any(kw in query_lower for kw in nav_keywords):
            return "navigational"

        # Transactional indicators
        trans_keywords = [
            'buy', 'purchase', 'price', 'download', 'order', 'shop',
            'cheap', 'deal', 'discount', 'sale', 'cost', 'rent',
            'book', 'reserve', 'subscribe'
        ]
        if any(kw in query_lower for kw in trans_keywords):
            return "transactional"

        # Informational indicators (explicit)
        info_keywords = [
            'what', 'how', 'why', 'who', 'when', 'where',
            'definition', 'explain', 'guide', 'tutorial', 'learn'
        ]
        if any(kw in query_lower for kw in info_keywords):
            return "informational"

        # Default: informational
        return "informational"

    def __del__(self):
        """Clean up Lucene reader."""
        try:
            if hasattr(self, 'lucene_reader'):
                self.lucene_reader.close()
        except Exception as e:
            logger.debug(f"Error closing Lucene reader: {e}")


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def _main_cli():
    """
    CLI for testing candidate extraction.

    Example:
        python -m msmeqe.expansion.candidate_extraction_pipeline \\
            --index-path data/msmarco_index \\
            --query "neural networks machine learning"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Test multi-source candidate extraction"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        required=True,
        help="Path to Lucene index",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text",
    )
    parser.add_argument(
        "--kb-wat-output",
        type=str,
        default=None,
        help="Path to WAT entity linking output (optional)",
    )
    parser.add_argument(
        "--vocab-embeddings",
        type=str,
        default=None,
        help="Path to vocabulary embeddings (optional)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # Initialize encoder
    logger.info("Initializing encoder...")
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize KB extractor (optional)
    kb_extractor = None
    if args.kb_wat_output:
        logger.info("Initializing KB extractor...")
        kb_extractor = KBCandidateExtractor(
            wat_output_path=args.kb_wat_output,
        )

    # Initialize embedding extractor (optional)
    emb_extractor = None
    if args.vocab_embeddings:
        logger.info("Initializing embedding extractor...")
        emb_extractor = EmbeddingCandidateExtractor(
            encoder=encoder,
            vocab_path=args.vocab_embeddings,
        )

    # Initialize extractor
    logger.info("Initializing candidate extractor...")
    extractor = MultiSourceCandidateExtractor(
        index_path=args.index_path,
        encoder=encoder,
        kb_extractor=kb_extractor,
        emb_extractor=emb_extractor,
    )

    # Extract candidates
    logger.info(f"Extracting candidates for: {args.query}")
    candidates = extractor.extract_all_candidates(
        query_text=args.query,
    )

    # Print results
    print(f"\nQuery: {args.query}")
    print(f"Total candidates: {len(candidates)}")
    print("\n" + "=" * 80)

    # Group by source
    from collections import defaultdict
    by_source = defaultdict(list)
    for cand in candidates:
        by_source[cand.source].append(cand)

    for source in ["docs", "kb", "emb"]:
        if source in by_source:
            print(f"\n{source.upper()} SOURCE ({len(by_source[source])} candidates):")
            print("-" * 80)
            for i, cand in enumerate(by_source[source][:10], 1):
                print(f"{i:2d}. {cand.term:30s} "
                      f"score={cand.native_score:.4f} "
                      f"df={cand.df:6d} "
                      f"rm3={cand.rm3_score:.4f}")

    # Compute query stats
    print("\n" + "=" * 80)
    print("QUERY STATISTICS:")
    print("-" * 80)
    query_stats = extractor.compute_query_stats(args.query)
    for key, value in query_stats.items():
        if isinstance(value, float):
            print(f"  {key:15s}: {value:.4f}")
        else:
            print(f"  {key:15s}: {value}")

    # Compute pseudo-centroid
    print("\n" + "=" * 80)
    print("PSEUDO-DOCUMENT CENTROID:")
    print("-" * 80)
    centroid = extractor.compute_pseudo_centroid(args.query)
    print(f"  Shape: {centroid.shape}")
    print(f"  Norm:  {np.linalg.norm(centroid):.4f}")


if __name__ == "__main__":
    _main_cli()