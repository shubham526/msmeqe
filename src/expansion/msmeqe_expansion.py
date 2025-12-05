"""
msmeqe/expansion/msmeqe_expansion.py

Core MS-MEQE expansion model.

Implements the pipeline described in the methodology section:

  - Takes multi-source candidate terms (docs / KB / embeddings)
  - Computes value and weight feature vectors
  - Predicts per-term value and weight using trained models
  - Predicts a query-specific expansion budget
  - Solves an unbounded knapsack to select term frequencies
  - Builds a magnitude-encoded enhanced query embedding

Dependencies:
  - msmeqe.reranking.semantic_encoder.SemanticEncoder
  - Regressor objects for value, weight, and budget (with .predict)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

from msmeqe.reranking.semantic_encoder import SemanticEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateTerm:
    """
    Representation of a single term-source candidate.

    Fields:
      - term: lexical term (e.g., "neural", "insulin resistance")
      - source: one of {"docs", "kb", "emb"} (or compatible)

      - rm3_score: RM3 term score from pseudo-relevant docs
      - tf_pseudo: normalized term frequency in pseudo-relevant docs
      - coverage_pseudo: fraction of pseudo docs that contain this term (0–1)

      - df: document frequency in the full collection
      - cf: collection frequency (optional; can be 0)

      - native_rank: rank within its own source (1 = best)
      - native_score: native score within its own source
                      (e.g., entity-linking confidence, cosine similarity)
    """
    term: str
    source: str

    rm3_score: float = 0.0
    tf_pseudo: float = 0.0
    coverage_pseudo: float = 0.0

    df: int = 0
    cf: int = 0

    native_rank: int = 0
    native_score: float = 0.0


@dataclass
class SelectedTerm:
    """
    Output of MS-MEQE after knapsack:

      - term: lexical term
      - source: which source it came from
      - value: predicted value v_{t,i}
      - weight: predicted weight w_{t,i}
      - count: frequency c_k chosen by unbounded knapsack
    """
    term: str
    source: str
    value: float
    weight: float
    count: int


# ---------------------------------------------------------------------------
# Core MS-MEQE model
# ---------------------------------------------------------------------------

class MSMEQEExpansionModel:
    """
    Core MS-MEQE expansion model.

    Assumes:
      - A Sentence-BERT encoder (SemanticEncoder) for embeddings
      - Pretrained regressors for value, weight, and budget
      - Candidates passed in as CandidateTerm instances
      - Pseudo-relevant document centroid embedding is provided (or None)

    Typical usage:

        from msmeqe.reranking.semantic_encoder import SemanticEncoder
        from msmeqe.expansion.msmeqe_expansion import MSMEQEExpansionModel, CandidateTerm

        encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        value_model = joblib.load("value_model.pkl")
        weight_model = joblib.load("weight_model.pkl")
        budget_model = joblib.load("budget_model.pkl")

        msmeqe = MSMEQEExpansionModel(
            encoder=encoder,
            value_model=value_model,
            weight_model=weight_model,
            budget_model=budget_model,
            collection_size=N_docs,
            lambda_interp=0.3,
        )

        selected_terms, q_star = msmeqe.expand(
            query_text=q,
            candidates=candidate_list,
            pseudo_doc_centroid=pseudo_centroid,   # np.ndarray or None
            query_stats=query_stats_dict,          # clarity, entropy, etc.
        )
    """

    def __init__(
        self,
        encoder: SemanticEncoder,
        value_model,
        weight_model,
        budget_model,
        collection_size: int,
        lambda_interp: float = 0.3,
        min_budget: int = 20,
        max_budget: int = 80,
        budget_step: int = 5,
    ) -> None:
        """
        Args:
            encoder: SemanticEncoder instance (Sentence-BERT wrapper)
            value_model: regressor with .predict(X) → values
            weight_model: regressor with .predict(X) → weights
            budget_model: regressor with .predict(X_query_features) → budget
            collection_size: total number of documents N
            lambda_interp: interpolation parameter λ in final query embedding
            min_budget, max_budget, budget_step: budget range & discretization
        """
        self.encoder = encoder
        self.value_model = value_model
        self.weight_model = weight_model
        self.budget_model = budget_model

        self.N = int(collection_size)
        self.lambda_interp = float(lambda_interp)

        self.min_budget = int(min_budget)
        self.max_budget = int(max_budget)
        self.budget_step = int(budget_step)

        if self.min_budget <= 0 or self.max_budget <= 0:
            raise ValueError("Budgets must be positive")
        if self.max_budget <= self.min_budget:
            raise ValueError("max_budget must be > min_budget")
        if self.budget_step <= 0:
            raise ValueError("budget_step must be positive")

        logger.info(
            "Initialized MSMEQEExpansionModel: N=%d, λ=%.3f, budget=[%d,%d], step=%d",
            self.N,
            self.lambda_interp,
            self.min_budget,
            self.max_budget,
            self.budget_step,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(
        self,
        query_text: str,
        candidates: List[CandidateTerm],
        pseudo_doc_centroid: Optional[np.ndarray],
        query_stats: Dict[str, float],
    ) -> Tuple[List[SelectedTerm], np.ndarray]:
        """
        Run full MS-MEQE for one query.

        Args:
            query_text: original query string
            candidates: list of CandidateTerm objects (multi-source)
            pseudo_doc_centroid: centroid embedding of pseudo-relevant docs
                                 (same dim as encoder embeddings) or None
            query_stats: dictionary with query-level stats for budget pred:
                - "clarity": float
                - "entropy": float
                - "avg_idf": float
                - "max_idf": float
                - "avg_bm25": float
                - "var_bm25": float
                - "q_len": int
                - "q_type": "navigational" | "informational" | "transactional"

        Returns:
            selected_terms: list of SelectedTerm with counts (frequencies)
            q_star: enhanced query embedding (np.ndarray)
        """
        if not candidates:
            logger.warning(
                "MSMEQE.expand called with no candidates; returning original query embedding."
            )
            q_emb = self.encoder.encode([query_text])[0]
            return [], q_emb

        # Encode query
        logger.debug("MSMEQE: encoding query")
        q_emb = self.encoder.encode([query_text])[0]  # (d,)

        # Encode candidate terms
        logger.debug("MSMEQE: encoding %d candidate terms", len(candidates))
        term_strings = [c.term for c in candidates]
        term_embs = self.encoder.encode(term_strings)  # (m, d)

        # Build features
        logger.debug("MSMEQE: building value feature matrix")
        X_val = self._build_value_feature_matrix(
            query_text=query_text,
            query_embedding=q_emb,
            pseudo_centroid=pseudo_doc_centroid,
            candidates=candidates,
            term_embeddings=term_embs,
        )

        logger.debug("MSMEQE: building weight feature matrix")
        X_wt = self._build_weight_feature_matrix(
            query_embedding=q_emb,
            candidates=candidates,
            term_embeddings=term_embs,
        )

        # Predict values and weights
        logger.debug("MSMEQE: predicting values and weights")
        v_pred = self._predict_values(X_val)  # (m,)
        w_pred = self._predict_weights(X_wt)  # (m,)
        w_pred = np.maximum(w_pred, 1e-6)     # ensure non-negative / non-zero

        # Predict budget
        logger.debug("MSMEQE: predicting budget")
        W = self._predict_budget(query_stats)

        # Solve unbounded knapsack
        logger.debug("MSMEQE: solving unbounded knapsack (m=%d, W=%d)", len(candidates), W)
        counts = self._solve_unbounded_knapsack(
            values=v_pred,
            weights=w_pred,
            budget=W,
        )

        # Collect selected terms
        selected_terms: List[SelectedTerm] = []
        for cand, v, w, c in zip(candidates, v_pred, w_pred, counts):
            if c <= 0:
                continue
            selected_terms.append(
                SelectedTerm(
                    term=cand.term,
                    source=cand.source,
                    value=float(v),
                    weight=float(w),
                    count=int(c),
                )
            )

        logger.debug("MSMEQE: %d terms selected by knapsack", len(selected_terms))

        # Build enhanced query embedding
        logger.debug("MSMEQE: constructing enhanced query embedding")
        q_star = self._build_enhanced_query_embedding(
            query_embedding=q_emb,
            term_embeddings=term_embs,
            values=v_pred,
            counts=counts,
        )

        return selected_terms, q_star

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _build_value_feature_matrix(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        pseudo_centroid: Optional[np.ndarray],
        candidates: List[CandidateTerm],
        term_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Build value feature matrix (m x d_features) for candidates.

        Features (simplified but aligned with the paper):

          Semantic (SBERT):
            - cos_sim_q: cosine(term_emb, q_emb)
            - cos_sim_pseudo: cosine(term_emb, pseudo_centroid) or 0 if None
            - l2_dist_q: ||term_emb - q_emb||_2

          Statistical:
            - idf = log(N / df)
            - tf_pseudo
            - rm3_score
            - coverage_pseudo

          Query-term interaction:
            - in_query: 1 if term appears in query text (whole-word match)

          Source-specific:
            - source_docs, source_kb, source_emb (one-hot)
            - native_rank_norm
            - native_score
        """
        m = len(candidates)
        if m == 0:
            return np.zeros((0, 1), dtype=np.float32)

        def _normalize(x: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
            return x / norms

        # Normalize embeddings
        q_norm = _normalize(query_embedding[None, :])[0]          # (d,)
        t_norm = _normalize(term_embeddings)                      # (m, d)

        # Cosine similarities and distances
        cos_sim_q = np.sum(t_norm * q_norm[None, :], axis=1)      # (m,)
        l2_dist_q = np.linalg.norm(term_embeddings - q_norm[None, :], axis=1)

        if pseudo_centroid is not None:
            p_norm = _normalize(pseudo_centroid[None, :])[0]
            cos_sim_pseudo = np.sum(t_norm * p_norm[None, :], axis=1)
        else:
            cos_sim_pseudo = np.zeros(m, dtype=np.float32)

        # Basic stats
        df = np.array([max(c.df, 1) for c in candidates], dtype=np.float32)
        idf = np.log(self.N / df)

        tf_pseudo = np.array([c.tf_pseudo for c in candidates], dtype=np.float32)
        rm3 = np.array([c.rm3_score for c in candidates], dtype=np.float32)
        coverage = np.array([c.coverage_pseudo for c in candidates], dtype=np.float32)

        # Query-term interaction: simple whole-word lexical match
        q_lower = " " + query_text.lower() + " "
        in_query = np.array(
            [
                1.0 if f" {c.term.lower()} " in q_lower else 0.0
                for c in candidates
            ],
            dtype=np.float32,
        )

        # Source one-hot
        source_docs = np.array(
            [1.0 if c.source == "docs" else 0.0 for c in candidates],
            dtype=np.float32,
        )
        source_kb = np.array(
            [1.0 if c.source == "kb" else 0.0 for c in candidates],
            dtype=np.float32,
        )
        source_emb = np.array(
            [1.0 if c.source == "emb" else 0.0 for c in candidates],
            dtype=np.float32,
        )

        native_rank = np.array([float(c.native_rank) for c in candidates], dtype=np.float32)
        native_score = np.array([c.native_score for c in candidates], dtype=np.float32)

        max_rank = np.max(native_rank) if np.max(native_rank) > 0 else 1.0
        native_rank_norm = native_rank / max_rank

        features = np.stack(
            [
                cos_sim_q,
                cos_sim_pseudo,
                l2_dist_q,
                idf,
                tf_pseudo,
                rm3,
                coverage,
                in_query,
                source_docs,
                source_kb,
                source_emb,
                native_rank_norm,
                native_score,
            ],
            axis=1,
        ).astype(np.float32)

        return features

    def _build_weight_feature_matrix(
        self,
        query_embedding: np.ndarray,
        candidates: List[CandidateTerm],
        term_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Build weight feature matrix, focusing on drift/ambiguity risk.

        Features:

          Drift:
            - cos_sim_q
            - l2_dist_q

          Ambiguity:
            - df_norm = df / N
            - idf (low idf = more ambiguous)
            - coverage_pseudo

          Source risk:
            - source_docs, source_kb, source_emb
            - native_rank_norm
            - native_score
        """
        m = len(candidates)
        if m == 0:
            return np.zeros((0, 1), dtype=np.float32)

        def _normalize(x: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
            return x / norms

        q_norm = _normalize(query_embedding[None, :])[0]
        t_norm = _normalize(term_embeddings)

        cos_sim_q = np.sum(t_norm * q_norm[None, :], axis=1)
        l2_dist_q = np.linalg.norm(term_embeddings - q_norm[None, :], axis=1)

        df = np.array([max(c.df, 1) for c in candidates], dtype=np.float32)
        df_norm = df / float(self.N)
        idf = np.log(self.N / df)
        coverage = np.array([c.coverage_pseudo for c in candidates], dtype=np.float32)

        source_docs = np.array(
            [1.0 if c.source == "docs" else 0.0 for c in candidates],
            dtype=np.float32,
        )
        source_kb = np.array(
            [1.0 if c.source == "kb" else 0.0 for c in candidates],
            dtype=np.float32,
        )
        source_emb = np.array(
            [1.0 if c.source == "emb" else 0.0 for c in candidates],
            dtype=np.float32,
        )

        native_rank = np.array([float(c.native_rank) for c in candidates], dtype=np.float32)
        native_score = np.array([c.native_score for c in candidates], dtype=np.float32)

        max_rank = np.max(native_rank) if np.max(native_rank) > 0 else 1.0
        native_rank_norm = native_rank / max_rank

        features = np.stack(
            [
                cos_sim_q,
                l2_dist_q,
                df_norm,
                idf,
                coverage,
                source_docs,
                source_kb,
                source_emb,
                native_rank_norm,
                native_score,
            ],
            axis=1,
        ).astype(np.float32)

        return features

    # ------------------------------------------------------------------
    # Model predictions
    # ------------------------------------------------------------------

    def _predict_values(self, X: np.ndarray) -> np.ndarray:
        """Predict value v_{t,i} for each candidate using the value model."""
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        y = self.value_model.predict(X)
        return np.asarray(y, dtype=np.float32).reshape(-1)

    def _predict_weights(self, X: np.ndarray) -> np.ndarray:
        """Predict weight (risk) w_{t,i} for each candidate using the weight model."""
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        y = self.weight_model.predict(X)
        return np.asarray(y, dtype=np.float32).reshape(-1)

    def _predict_budget(self, query_stats: Dict[str, float]) -> int:
        """
        Predict query-specific expansion budget W from query-level features.

        Expected keys in query_stats:
            - clarity
            - entropy
            - avg_idf
            - max_idf
            - avg_bm25
            - var_bm25
            - q_len
            - q_type  ("navigational" / "informational" / "transactional")

        Returns:
            Integer budget in [min_budget, max_budget] snapped to nearest budget_step.
        """
        q_type = str(query_stats.get("q_type", "informational")).lower()
        type_nav = 1.0 if q_type == "navigational" else 0.0
        type_tran = 1.0 if q_type == "transactional" else 0.0
        type_info = 1.0 if q_type == "informational" else 0.0

        x_vec = np.array(
            [
                float(query_stats.get("clarity", 0.0)),
                float(query_stats.get("entropy", 0.0)),
                float(query_stats.get("avg_idf", 0.0)),
                float(query_stats.get("max_idf", 0.0)),
                float(query_stats.get("avg_bm25", 0.0)),
                float(query_stats.get("var_bm25", 0.0)),
                float(query_stats.get("q_len", 0)),
                type_nav,
                type_info,
                type_tran,
            ],
            dtype=np.float32,
        ).reshape(1, -1)

        budget_pred = float(self.budget_model.predict(x_vec)[0])

        # Clip and snap to nearest step
        clipped = max(self.min_budget, min(self.max_budget, budget_pred))
        snapped = self.budget_step * int(round(clipped / self.budget_step))

        logger.debug(
            "Budget prediction: raw=%.2f, clipped=%.2f, snapped=%d",
            budget_pred,
            clipped,
            snapped,
        )
        return snapped

    # ------------------------------------------------------------------
    # Unbounded knapsack
    # ------------------------------------------------------------------

    def _solve_unbounded_knapsack(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        """
        Unbounded knapsack DP:
          max sum c_k * v_k  s.t. sum c_k * w_k <= W, c_k ∈ N

        We discretize weights to integers via a scale factor and use
        standard DP over capacity.

        Returns:
            counts: np.ndarray of length m with optimal c_k frequencies.
        """
        m = len(values)
        if m == 0 or budget <= 0:
            return np.zeros((m,), dtype=np.int32)

        v = np.asarray(values, dtype=np.float32)
        w = np.asarray(weights, dtype=np.float32)

        # Scale weights so that median weight ≈ 5 (keep capacity manageable)
        median_w = float(np.median(w))
        scale = 1.0
        if median_w > 0:
            scale = 5.0 / median_w
        w_scaled = np.maximum(1, np.round(w * scale).astype(np.int32))

        W = int(budget)
        dp = np.zeros(W + 1, dtype=np.float32)
        choice = np.full(W + 1, -1, dtype=np.int32)

        for cap in range(1, W + 1):
            best_val = dp[cap]
            best_item = -1
            for i in range(m):
                wi = w_scaled[i]
                if wi <= cap:
                    cand_val = v[i] + dp[cap - wi]
                    if cand_val > best_val:
                        best_val = cand_val
                        best_item = i
            dp[cap] = best_val
            choice[cap] = best_item

        counts = np.zeros(m, dtype=np.int32)
        cap = W
        while cap > 0 and choice[cap] != -1:
            i = choice[cap]
            counts[i] += 1
            cap -= w_scaled[i]

        return counts

    # ------------------------------------------------------------------
    # Query embedding construction
    # ------------------------------------------------------------------

    def _build_enhanced_query_embedding(
        self,
        query_embedding: np.ndarray,
        term_embeddings: np.ndarray,
        values: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        """
        Build magnitude-encoded enhanced query embedding:

            e_scaled(t_k) = c_k * v_k * e(t_k)

            q* = (1 - λ) e(q) + λ * (sum e_scaled / sum (c_k * v_k))

        If no terms are selected (all counts=0), returns original e(q).
        """
        if term_embeddings.shape[0] == 0 or np.all(counts <= 0):
            return query_embedding

        scaled_vectors = []
        scaled_weights = []

        for emb, v, c in zip(term_embeddings, values, counts):
            if c <= 0 or v <= 0:
                continue
            weight = float(c) * float(v)
            scaled_vectors.append(weight * emb)
            scaled_weights.append(weight)

        if not scaled_vectors:
            return query_embedding

        scaled_vectors = np.stack(scaled_vectors, axis=0)
        scaled_weights = np.array(scaled_weights, dtype=np.float32)

        weighted_avg = np.sum(scaled_vectors, axis=0) / (np.sum(scaled_weights) + 1e-12)
        q_star = (1.0 - self.lambda_interp) * query_embedding + self.lambda_interp * weighted_avg
        return q_star
