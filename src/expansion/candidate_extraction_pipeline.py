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
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Any
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
            kb_override: Optional[List[Dict]] = None,
    ) -> List[CandidateTerm]:
        """
        Extract candidates from all sources with complete statistics.

        Optimized to pre-tokenize pseudo-documents once to avoid overhead during
        individual term statistics calculation.
        """
        all_candidates = []

        # 1. Retrieve pseudo-relevant documents once (shared across calculations)
        # Returns (list of text, list of tokenized lists)
        pseudo_docs, pseudo_docs_tokens = self._get_pseudo_relevant_docs_and_tokens(
            query_text, self.n_pseudo_docs
        )

        # === 1. DOCS SOURCE (RM3) ===
        logger.debug(f"Extracting RM3 candidates for: {query_text[:50]}")

        try:
            # We assume RM3 scorer handles its own retrieval internally for the model,
            # though ideally we would pass the retrieved docs to it to save compute.
            rm3_terms = self.rm3_scorer.expand(
                query_str=query_text,
                n_docs=self.n_pseudo_docs,
                n_terms=self.n_docs_rm3,
                use_rm3=True,
            )

            for rank, (term, rm3_score) in enumerate(rm3_terms, start=1):
                df, cf = self._get_term_stats(term)
                # Use pre-tokenized docs for efficiency
                tf_pseudo = self._compute_tf_pseudo_optimized(term, pseudo_docs_tokens)
                coverage = self._compute_coverage_optimized(term, pseudo_docs)

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
                for rank, kb_cand_dict in enumerate(kb_override[:self.n_kb], start=1):
                    # Validation: Ensure required keys exist
                    term = kb_cand_dict.get('term')
                    if not term:
                        continue

                    confidence = kb_cand_dict.get('confidence', 1.0)
                    # Use provided rank if available, else enumeration
                    cand_rank = kb_cand_dict.get('rank', rank)

                    df, cf = self._get_term_stats(term)

                    all_candidates.append(CandidateTerm(
                        term=term,
                        source="kb",
                        rm3_score=0.0,
                        tf_pseudo=0.0,
                        coverage_pseudo=0.0,
                        df=df,
                        cf=cf,
                        native_rank=cand_rank,
                        native_score=confidence,
                    ))
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

    def _perform_lucene_search(self, query: str, n: int):
        """
        Centralized helper to perform Lucene search and return top docs.
        Returns Tuple(TopDocs, score_docs).
        """
        try:
            query_parser = self.QueryParser(self.field_name, self.analyzer)
            lucene_query = query_parser.parse(query)
            top_docs = self.lucene_searcher.search(lucene_query, n)
            return top_docs
        except Exception as e:
            logger.warning(f"Lucene search failed for query '{query}': {e}")
            return None

    def _get_pseudo_relevant_docs_and_tokens(
            self,
            query: str,
            n: int = 10
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Get pseudo-relevant docs AND their tokenized versions.
        Optimized to do retrieval once.
        """
        docs_text = []
        docs_tokens = []

        top_docs = self._perform_lucene_search(query, n)
        if top_docs is None:
            return [], []

        for score_doc in top_docs.scoreDocs:
            try:
                doc = self.lucene_searcher.storedFields().document(score_doc.doc)
                doc_text = doc.get(self.field_name)
                if doc_text:
                    docs_text.append(doc_text)
                    # Simple tokenization for stats calculation
                    # (lowercasing done here to save time in loop)
                    docs_tokens.append(doc_text.lower().split())
            except Exception as e:
                continue

        return docs_text, docs_tokens

    def _get_term_stats(self, term: str) -> Tuple[int, int]:
        """
        Get document frequency (DF) and collection frequency (CF) for a term.
        Includes robust UTF-8 handling for JNI.
        """
        if not term or not isinstance(term, str):
            return 1, 1

        try:
            # Get terms for the field
            terms = self.lucene_reader.terms(self.field_name)
            if terms is None:
                return 1, 1

            # Get term iterator
            terms_enum = terms.iterator()

            # SAFELY Create BytesRef
            try:
                # Ensure clean UTF-8 encoding
                term_encoded = term.strip().lower().encode('utf-8')
                term_bytes = self.BytesRef(term_encoded)
            except Exception as e:
                logger.debug(f"Failed to encode term '{term}' for Lucene lookup: {e}")
                return 1, 1

            # Seek to term
            if terms_enum.seekExact(term_bytes):
                df = terms_enum.docFreq()
                cf = terms_enum.totalTermFreq()
                return max(df, 1), max(cf, 1)

            # Term not found
            return 1, 1

        except Exception as e:
            # Fallback for weird JNI errors or index issues
            # Don't log error on every miss to avoid log spam, use debug
            logger.debug(f"Error getting stats for term '{term}': {e}")
            return 1, 1

    def _compute_tf_pseudo_optimized(
            self,
            term: str,
            pseudo_docs_tokens: List[List[str]]
    ) -> float:
        """
        Compute normalized term frequency using pre-tokenized docs.
        """
        if not pseudo_docs_tokens:
            return 0.0

        term_lower = term.lower()
        total_count = 0
        total_words = 0

        for doc_tokens in pseudo_docs_tokens:
            total_words += len(doc_tokens)
            total_count += doc_tokens.count(term_lower)

        if total_words == 0:
            return 0.0

        return total_count / total_words

    def _compute_coverage_optimized(
            self,
            term: str,
            pseudo_docs_text: List[str]
    ) -> float:
        """
        Compute fraction of pseudo-docs containing the term.
        """
        if not pseudo_docs_text:
            return 0.0

        term_lower = term.lower()
        # Fast string check
        count = sum(1 for doc in pseudo_docs_text if term_lower in doc.lower())

        return count / len(pseudo_docs_text)

    def compute_pseudo_centroid(self, query_text: str, precomputed_docs: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute centroid of pseudo-relevant document embeddings.
        Accepts precomputed docs to avoid re-searching Lucene.
        """
        if precomputed_docs is not None:
            pseudo_docs = precomputed_docs
        else:
            pseudo_docs, _ = self._get_pseudo_relevant_docs_and_tokens(query_text, self.n_pseudo_docs)

        if not pseudo_docs:
            return np.zeros(self.encoder.get_dim(), dtype=np.float32)

        # Encode documents
        pseudo_embeddings = self.encoder.encode(pseudo_docs)

        # Compute centroid
        centroid = np.mean(pseudo_embeddings, axis=0)

        return centroid

    def compute_query_stats(
            self,
            query_text: str,
            precomputed_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute query-level statistics for budget prediction.

        Args:
            query_text: The query
            precomputed_scores: Optional numpy array of top BM25 scores if already retrieved.
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

        # === RETRIEVAL STATS (Shared Search) ===
        # If scores aren't provided, perform ONE search here to get them
        if precomputed_scores is None:
            top_docs = self._perform_lucene_search(query_text, self.n_pseudo_docs)
            if top_docs and top_docs.scoreDocs:
                precomputed_scores = np.array([sd.score for sd in top_docs.scoreDocs])
            else:
                precomputed_scores = np.array([])

        # === ENTROPY & BM25 STATS ===
        entropy = self._compute_retrieval_entropy_from_scores(precomputed_scores)
        avg_bm25, var_bm25 = self._compute_bm25_stats_from_scores(precomputed_scores)

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
        """
        if not query_tokens:
            return 0.0

        clarity = 0.0
        for token in query_tokens:
            df, cf = self._get_term_stats(token)

            # P(t|C) = collection frequency / collection size
            # Assume avg doc length ~100 for normalization if strict count isn't available
            p_t_c = cf / (self.collection_size * 100)

            # P(t|q) = uniform over query terms
            p_t_q = 1.0 / len(query_tokens)

            if p_t_c > 0:
                clarity += p_t_q * np.log(p_t_q / p_t_c)

        return float(clarity)

    def _compute_retrieval_entropy_from_scores(self, scores: np.ndarray) -> float:
        """
        Compute entropy from pre-computed scores to avoid re-searching.
        """
        if scores.size == 0 or scores.sum() == 0:
            return 0.0

        # Normalize to probabilities
        probs = scores / scores.sum()

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(entropy)

    def _compute_bm25_stats_from_scores(self, scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute BM25 stats from pre-computed scores.
        """
        if scores.size == 0:
            return 0.0, 0.0

        avg_bm25 = float(np.mean(scores))
        var_bm25 = float(np.var(scores))
        return avg_bm25, var_bm25

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type using Regex-based heuristics (Robust).

        Returns: "navigational", "informational", or "transactional"
        """
        query_lower = query.lower()

        def has_word(keywords, text):
            # Creates regex like: \b(buy|purchase|price)\b
            pattern = r'\b(' + '|'.join([re.escape(k) for k in keywords]) + r')\b'
            return bool(re.search(pattern, text))

        # Navigational indicators
        nav_keywords = [
            'homepage', 'website', 'official', 'login', 'site',
            'main page', 'home page', 'portal', 'www', '.com', '.org'
        ]
        if has_word(nav_keywords, query_lower):
            return "navigational"

        # Transactional indicators
        trans_keywords = [
            'buy', 'purchase', 'price', 'download', 'order', 'shop',
            'cheap', 'deal', 'discount', 'sale', 'cost', 'rent',
            'book', 'reserve', 'subscribe', 'coupon', 'review'
        ]
        if has_word(trans_keywords, query_lower):
            return "transactional"

        # Informational indicators
        info_keywords = [
            'what', 'how', 'why', 'who', 'when', 'where',
            'definition', 'explain', 'guide', 'tutorial', 'learn',
            'meaning', 'example', 'history', 'difference', 'vs'
        ]
        if has_word(info_keywords, query_lower):
            return "informational"

        # Default fallback
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
    import argparse

    parser = argparse.ArgumentParser(description="Test multi-source candidate extraction")
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--kb-wat-output", type=str, default=None)
    parser.add_argument("--vocab-embeddings", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info("Initializing encoder...")
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    kb_extractor = None
    if args.kb_wat_output:
        kb_extractor = KBCandidateExtractor(wat_output_path=args.kb_wat_output)

    emb_extractor = None
    if args.vocab_embeddings:
        emb_extractor = EmbeddingCandidateExtractor(encoder=encoder, vocab_path=args.vocab_embeddings)

    logger.info("Initializing candidate extractor...")
    extractor = MultiSourceCandidateExtractor(
        index_path=args.index_path,
        encoder=encoder,
        kb_extractor=kb_extractor,
        emb_extractor=emb_extractor,
    )

    logger.info(f"Extracting candidates for: {args.query}")
    candidates = extractor.extract_all_candidates(query_text=args.query)

    print(f"\nQuery: {args.query}")
    print(f"Total candidates: {len(candidates)}")

    # Compute query stats (testing shared retrieval logic internally)
    print("\nQUERY STATISTICS:")
    stats = extractor.compute_query_stats(args.query)
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _main_cli()