# msmeqe/reranking/semantic_encoder.py

from __future__ import annotations
import logging
from typing import List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticEncoder:
    """
    Wrapper around Sentence-BERT that:
      - Loads a transformer model
      - Encodes queries/terms in batches
      - Always returns L2-normalized embeddings
      - Provides a simple top_k_neighbors helper for vocab-based NN search
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32, device: str | None = None):
        """
        Args:
            model_name: SBERT model name from HuggingFace
            batch_size: batch size for encoding
            device: optional (cpu / cuda), SBERT auto-detects if None
        """
        logger.info(f"Loading SemanticEncoder model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    # -------------------------
    # Encoding
    # -------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings → L2-normalized embeddings.

        Args:
            texts: list of text strings

        Returns:
            np.ndarray of shape (n, d)
        """
        if not texts:
            return np.zeros((0, self.get_dim()), dtype=np.float32)

        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )

        return self._normalize(emb)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string → normalized 1D vector.
        """
        emb = self.encode([text])
        return emb[0]

    def get_dim(self) -> int:
        """
        Returns embedding dimensionality.
        """
        return self.model.get_sentence_embedding_dimension()

    # -------------------------
    # Nearest neighbors for vocab (used by embedding_candidates)
    # -------------------------

    def top_k_neighbors(
        self,
        query_text: str,
        vocab_terms: Sequence[str],
        vocab_embs: np.ndarray,
        k: int = 30,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k nearest neighbor terms for a query over a pre-embedded vocab.

        Args:
            query_text: the raw query string
            vocab_terms: list of vocab tokens, same order as vocab_embs
            vocab_embs: np.ndarray of shape (V, d), L2-normalized
            k: number of neighbors

        Returns:
            List of (term, cosine_similarity) sorted by similarity desc
        """
        if len(vocab_terms) == 0:
            return []

        q_emb = self.encode_single(query_text)      # (d,)
        # vocab_embs should already be normalized; safe-guard:
        v_emb = self._normalize(vocab_embs)

        sims = v_emb @ q_emb  # (V,)

        k = min(k, len(vocab_terms))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        return [(vocab_terms[i], float(sims[i])) for i in top_idx]

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
        return v / norm
