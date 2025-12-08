# src/features/feature_extraction.py

"""
Feature Extraction for MS-MEQE

Implements the feature extraction described in Sections 3.3-3.4 of the paper:
  - Value features (18 features): semantic, statistical, query-term interaction,
    retrieval-theoretic, source-specific
  - Weight features (20 features): drift, ambiguity, source risk
  - Query features (10 features): for budget prediction

This module is used by:
  1. msmeqe_expansion.py (at inference time)
  2. create_training_data_msmeqe.py (for generating training instances)
  3. train_value_weight_models.py (for model training)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extract features for value/weight prediction and budget allocation.

    Implements the feature sets described in the paper:
      - Section 3.3.1: Value features (18 features)
      - Section 3.4.1: Weight features (20 features)
      - Section 3.6.1: Query features (10 features)
    """

    def __init__(
            self,
            collection_size: int,
            stopwords: Optional[Set[str]] = None,
    ):
        """
        Initialize feature extractor.

        Args:
            collection_size: Total number of documents in collection (N)
            stopwords: Set of stopwords for filtering
        """
        self.N = int(collection_size)
        self.stopwords = stopwords or self._get_default_stopwords()

        logger.info(f"Initialized FeatureExtractor: N={self.N}")

    # -----------------------------------------------------------------------
    # Value features (Section 3.3.1) - 18 features
    # -----------------------------------------------------------------------

    def extract_value_features(
            self,
            candidate_term: str,
            candidate_source: str,
            candidate_stats: Dict[str, float],
            query_text: str,
            query_embedding: np.ndarray,
            term_embedding: np.ndarray,
            pseudo_centroid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract value features for a single candidate term.

        Args:
            candidate_term: The term string
            candidate_source: Source ("docs", "kb", "emb")
            candidate_stats: Dict with:
                - rm3_score: float
                - tf_pseudo: float
                - coverage_pseudo: float
                - df: int
                - cf: int (optional)
                - native_rank: int
                - native_score: float
            query_text: Original query
            query_embedding: Query embedding vector (d,)
            term_embedding: Term embedding vector (d,)
            pseudo_centroid: Centroid of pseudo-relevant docs (d,) or None

        Returns:
            Feature vector of shape (18,)
        """
        features = []

        # === SEMANTIC FEATURES (3) ===
        cos_sim_q = self._cosine_similarity(query_embedding, term_embedding)
        l2_dist_q = np.linalg.norm(query_embedding - term_embedding)

        if pseudo_centroid is not None:
            cos_sim_pseudo = self._cosine_similarity(pseudo_centroid, term_embedding)
        else:
            cos_sim_pseudo = 0.0

        features.extend([cos_sim_q, cos_sim_pseudo, l2_dist_q])

        # === STATISTICAL FEATURES (5) ===
        df = max(candidate_stats.get('df', 1), 1)
        idf = np.log(self.N / df)
        tf_pseudo = candidate_stats.get('tf_pseudo', 0.0)
        rm3_score = candidate_stats.get('rm3_score', 0.0)
        coverage = candidate_stats.get('coverage_pseudo', 0.0)

        features.extend([idf, tf_pseudo, rm3_score, coverage])

        # === QUERY-TERM INTERACTION FEATURES (3) ===
        in_query = 1.0 if self._term_in_query(candidate_term, query_text) else 0.0
        jaccard = self._jaccard_similarity(candidate_term, query_text)
        pmi_max = self._compute_pmi_max(
            candidate_term,
            query_text,
            candidate_stats
        )

        features.extend([in_query, jaccard, pmi_max])

        # === RETRIEVAL-THEORETIC FEATURES (2) ===
        bm25_delta = self._compute_bm25_delta(
            candidate_stats,
            idf,
            tf_pseudo,
        )
        clarity_delta = self._compute_clarity_delta(
            idf,
            coverage,
            rm3_score,
        )

        features.extend([bm25_delta, clarity_delta])

        # === SOURCE-SPECIFIC FEATURES (5) ===
        source_docs = 1.0 if candidate_source == "docs" else 0.0
        source_kb = 1.0 if candidate_source == "kb" else 0.0
        source_emb = 1.0 if candidate_source == "emb" else 0.0

        native_rank = candidate_stats.get('native_rank', 0)
        native_score = candidate_stats.get('native_score', 0.0)

        # Normalize rank (lower rank = better)
        native_rank_norm = 1.0 / (native_rank + 1.0) if native_rank > 0 else 0.0

        features.extend([
            source_docs,
            source_kb,
            source_emb,
            native_rank_norm,
            native_score,
        ])

        return np.array(features, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Weight features (Section 3.4.1) - 20 features
    # -----------------------------------------------------------------------

    def extract_weight_features(
            self,
            candidate_term: str,
            candidate_source: str,
            candidate_stats: Dict[str, float],
            query_text: str,
            query_embedding: np.ndarray,
            term_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Extract weight (cost/risk) features for a single candidate term.

        Args:
            candidate_term: The term string
            candidate_source: Source ("docs", "kb", "emb")
            candidate_stats: Dict with term statistics
            query_text: Original query
            query_embedding: Query embedding (d,)
            term_embedding: Term embedding (d,)

        Returns:
            Feature vector of shape (20,)
        """
        features = []

        # === DRIFT FEATURES (3) ===
        cos_sim_q = self._cosine_similarity(query_embedding, term_embedding)
        l2_dist_q = np.linalg.norm(query_embedding - term_embedding)
        edit_dist = self._edit_distance_normalized(candidate_term, query_text)

        features.extend([cos_sim_q, l2_dist_q, edit_dist])

        # === AMBIGUITY FEATURES (5) ===
        df = max(candidate_stats.get('df', 1), 1)
        cf = candidate_stats.get('cf', df)

        df_norm = df / self.N
        cf_norm = cf / max(self.N * 100, 1)  # Normalize by collection size * avg doc length
        idf = np.log(self.N / df)

        # Polysemy: number of word senses (simplified: use term complexity as proxy)
        polysemy = self._compute_polysemy(candidate_term)

        coverage = candidate_stats.get('coverage_pseudo', 0.0)

        features.extend([df_norm, cf_norm, idf, polysemy, coverage])

        # === SOURCE RISK FEATURES (7) ===
        source_docs = 1.0 if candidate_source == "docs" else 0.0
        source_kb = 1.0 if candidate_source == "kb" else 0.0
        source_emb = 1.0 if candidate_source == "emb" else 0.0

        native_rank = candidate_stats.get('native_rank', 0)
        native_score = candidate_stats.get('native_score', 0.0)

        # Source confidence (inverse of rank for kb/emb, score for docs)
        source_confidence = self._compute_source_confidence(
            candidate_source,
            native_rank,
            native_score,
        )

        # RM3 score (low RM3 = high risk for docs source)
        rm3_score = candidate_stats.get('rm3_score', 0.0)

        features.extend([
            source_docs,
            source_kb,
            source_emb,
            native_rank / 100.0,  # Normalized rank
            native_score,
            source_confidence,
            rm3_score,
        ])

        # === ADDITIONAL RISK SIGNALS (5) ===
        # Term length (very short or very long = risky)
        term_len = len(candidate_term)
        term_len_norm = min(term_len / 20.0, 1.0)

        # Is stopword (risky)
        is_stopword = 1.0 if candidate_term.lower() in self.stopwords else 0.0

        # In query (not risky if already in query)
        in_query = 1.0 if self._term_in_query(candidate_term, query_text) else 0.0

        # Character diversity (low diversity = risky, e.g., "aaaa")
        char_diversity = len(set(candidate_term)) / max(len(candidate_term), 1)

        # Has numbers (can be risky depending on context)
        has_numbers = 1.0 if any(c.isdigit() for c in candidate_term) else 0.0

        features.extend([
            term_len_norm,
            is_stopword,
            in_query,
            char_diversity,
            has_numbers,
        ])

        return np.array(features, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Query features (Section 3.6.1) - 10 features
    # -----------------------------------------------------------------------

    def extract_query_features(
            self,
            query_text: str,
            query_stats: Dict[str, float],
    ) -> np.ndarray:
        """
        Extract query-level features for budget prediction.

        Args:
            query_text: Query string
            query_stats: Dict with:
                - clarity: float
                - entropy: float
                - avg_idf: float
                - max_idf: float
                - avg_bm25: float (avg score of top-10 docs)
                - var_bm25: float (variance of top-10 scores)
                - q_type: str ("navigational", "informational", "transactional")

        Returns:
            Feature vector of shape (10,)
        """
        features = []

        # === CLARITY AND AMBIGUITY (2) ===
        clarity = query_stats.get('clarity', 0.0)
        entropy = query_stats.get('entropy', 0.0)

        features.extend([clarity, entropy])

        # === STATISTICAL PROPERTIES (3) ===
        q_len = len(query_text.split())
        avg_idf = query_stats.get('avg_idf', 0.0)
        max_idf = query_stats.get('max_idf', 0.0)

        features.extend([float(q_len), avg_idf, max_idf])

        # === RETRIEVAL QUALITY (2) ===
        avg_bm25 = query_stats.get('avg_bm25', 0.0)
        var_bm25 = query_stats.get('var_bm25', 0.0)

        features.extend([avg_bm25, var_bm25])

        # === QUERY TYPE (3) ===
        q_type = query_stats.get('q_type', 'informational').lower()
        type_nav = 1.0 if q_type == 'navigational' else 0.0
        type_info = 1.0 if q_type == 'informational' else 0.0
        type_tran = 1.0 if q_type == 'transactional' else 0.0

        features.extend([type_nav, type_info, type_tran])

        return np.array(features, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Batch processing (for efficiency)
    # -----------------------------------------------------------------------

    def extract_value_features_batch(
            self,
            candidates: List[Tuple[str, str, Dict]],  # [(term, source, stats), ...]
            query_text: str,
            query_embedding: np.ndarray,
            term_embeddings: np.ndarray,  # (m, d)
            pseudo_centroid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract value features for multiple candidates (batch processing).

        Args:
            candidates: List of (term, source, stats) tuples
            query_text: Query string
            query_embedding: Query embedding (d,)
            term_embeddings: Term embeddings (m, d)
            pseudo_centroid: Pseudo-doc centroid (d,) or None

        Returns:
            Feature matrix of shape (m, 18)
        """
        m = len(candidates)
        features_list = []

        for i, (term, source, stats) in enumerate(candidates):
            feat = self.extract_value_features(
                candidate_term=term,
                candidate_source=source,
                candidate_stats=stats,
                query_text=query_text,
                query_embedding=query_embedding,
                term_embedding=term_embeddings[i],
                pseudo_centroid=pseudo_centroid,
            )
            features_list.append(feat)

        return np.vstack(features_list) if features_list else np.zeros((0, 18), dtype=np.float32)

    def extract_weight_features_batch(
            self,
            candidates: List[Tuple[str, str, Dict]],  # [(term, source, stats), ...]
            query_text: str,
            query_embedding: np.ndarray,
            term_embeddings: np.ndarray,  # (m, d)
    ) -> np.ndarray:
        """
        Extract weight features for multiple candidates (batch processing).

        Args:
            candidates: List of (term, source, stats) tuples
            query_text: Query string
            query_embedding: Query embedding (d,)
            term_embeddings: Term embeddings (m, d)

        Returns:
            Feature matrix of shape (m, 20)
        """
        m = len(candidates)
        features_list = []

        for i, (term, source, stats) in enumerate(candidates):
            feat = self.extract_weight_features(
                candidate_term=term,
                candidate_source=source,
                candidate_stats=stats,
                query_text=query_text,
                query_embedding=query_embedding,
                term_embedding=term_embeddings[i],
            )
            features_list.append(feat)

        return np.vstack(features_list) if features_list else np.zeros((0, 20), dtype=np.float32)

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a) + 1e-12
        norm_b = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _term_in_query(term: str, query: str) -> bool:
        """Check if term appears in query (whole-word match)."""
        query_lower = " " + query.lower() + " "
        term_lower = " " + term.lower() + " "
        return term_lower in query_lower

    def _jaccard_similarity(self, term: str, query: str) -> float:
        """Compute Jaccard similarity between term and query tokens."""
        term_tokens = set(term.lower().split())
        query_tokens = set(query.lower().split())

        if not term_tokens or not query_tokens:
            return 0.0

        intersection = len(term_tokens & query_tokens)
        union = len(term_tokens | query_tokens)

        return intersection / union if union > 0 else 0.0

    def _compute_pmi_max(
            self,
            term: str,
            query: str,
            stats: Dict[str, float],
    ) -> float:
        """
        Compute maximum PMI between term and any query token.

        PMI(t, q_i) = log(P(t, q_i) / (P(t) * P(q_i)))

        Simplified: use co-occurrence in pseudo-docs as proxy.
        """
        # If term appears in query, high PMI
        if self._term_in_query(term, query):
            return 2.0

        # Otherwise, use coverage as proxy for co-occurrence
        coverage = stats.get('coverage_pseudo', 0.0)
        rm3_score = stats.get('rm3_score', 0.0)

        # Proxy PMI: higher coverage + RM3 score = higher PMI
        pmi_proxy = np.log1p(coverage * rm3_score * 10)

        return float(pmi_proxy)

    def _compute_bm25_delta(
            self,
            stats: Dict[str, float],
            idf: float,
            tf_pseudo: float,
    ) -> float:
        """
        Estimate BM25 score increase if term is added to query.

        Simplified: BM25_Δ ≈ IDF * TF (in pseudo-docs)
        """
        # Proxy: IDF * normalized TF in pseudo-docs
        bm25_delta = idf * tf_pseudo

        return float(bm25_delta)

    def _compute_clarity_delta(
            self,
            idf: float,
            coverage: float,
            rm3_score: float,
    ) -> float:
        """
        Estimate change in query clarity if term is added.

        Clarity: sum_t P(t|q) * log(P(t|q) / P(t|C))

        Adding a term with high IDF and high P(t|q) increases clarity.
        """
        # Proxy: IDF * RM3 score (P(t|R)) * coverage
        clarity_delta = idf * rm3_score * coverage

        return float(clarity_delta)

    @staticmethod
    def _edit_distance_normalized(term: str, query: str) -> float:
        """
        Compute normalized edit distance between term and query.

        Returns value in [0, 1] where 0 = identical, 1 = completely different.
        """
        term_lower = term.lower()
        query_lower = query.lower()

        # Levenshtein distance
        len_term = len(term_lower)
        len_query = len(query_lower)

        if len_term == 0 or len_query == 0:
            return 1.0

        # Simple character-level distance
        max_len = max(len_term, len_query)
        common_chars = sum(1 for a, b in zip(term_lower, query_lower) if a == b)

        distance = 1.0 - (common_chars / max_len)

        return float(distance)

    def _compute_polysemy(self, term: str) -> float:
        """
        Estimate polysemy (number of senses) for a term.

        Simplified: use heuristics based on term properties.
        Ideally: use WordNet synset count.
        """
        # Heuristic: short common words tend to be more polysemous
        term_lower = term.lower()

        # Very short words are often polysemous
        if len(term_lower) <= 3:
            return 3.0

        # Common words (in stopwords) are polysemous
        if term_lower in self.stopwords:
            return 4.0

        # Longer technical terms tend to be less polysemous
        if len(term_lower) > 10:
            return 1.0

        # Default: moderate polysemy
        return 2.0

    def _compute_source_confidence(
            self,
            source: str,
            rank: int,
            score: float,
    ) -> float:
        """
        Compute source-specific confidence score.

        For KB/emb: higher rank = lower confidence
        For docs: use RM3 score directly
        """
        if source == "docs":
            return score
        elif source == "kb":
            # Entity linking confidence (inverse of rank)
            return 1.0 / (rank + 1.0) if rank > 0 else score
        elif source == "emb":
            # Cosine similarity
            return score
        else:
            return 0.0

    @staticmethod
    def _get_default_stopwords() -> Set[str]:
        """Get default English stopwords."""
        return {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work',
            'first', 'well', 'way', 'even', 'new', 'want', 'because',
            'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was',
            'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having',
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def create_candidate_stats_dict(
        rm3_score: float = 0.0,
        tf_pseudo: float = 0.0,
        coverage_pseudo: float = 0.0,
        df: int = 1,
        cf: int = 1,
        native_rank: int = 0,
        native_score: float = 0.0,
) -> Dict[str, float]:
    """
    Create a candidate stats dictionary with all required fields.

    Convenience function for creating the stats dict expected by feature extraction.
    """
    return {
        'rm3_score': float(rm3_score),
        'tf_pseudo': float(tf_pseudo),
        'coverage_pseudo': float(coverage_pseudo),
        'df': int(df),
        'cf': int(cf),
        'native_rank': int(native_rank),
        'native_score': float(native_score),
    }


def create_query_stats_dict(
        clarity: float = 0.0,
        entropy: float = 0.0,
        avg_idf: float = 0.0,
        max_idf: float = 0.0,
        avg_bm25: float = 0.0,
        var_bm25: float = 0.0,
        q_type: str = 'informational',
) -> Dict[str, float]:
    """
    Create a query stats dictionary with all required fields.

    Convenience function for creating the stats dict for query feature extraction.
    """
    return {
        'clarity': float(clarity),
        'entropy': float(entropy),
        'avg_idf': float(avg_idf),
        'max_idf': float(max_idf),
        'avg_bm25': float(avg_bm25),
        'var_bm25': float(var_bm25),
        'q_type': str(q_type),
    }