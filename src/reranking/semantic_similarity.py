"""
semantic_similarity.py

Sentence-BERT based semantic encoding and cosine similarity utilities.

This module provides a thin, reusable wrapper around SentenceTransformer
that is used consistently across the MS-MEQE codebase for:

  - Query / document encoding
  - Embedding-based similarity for features
  - Embedding-based candidate extraction (e.g., nearest neighbors)
  - Reranking with dense cosine scores

Design goals:
  - Single model load per process (singleton-style)
  - Thread-safe lazy initialization
  - Numpy outputs for easy downstream use
  - No YAML, configuration via function arguments / CLI flags only
"""

import logging
import threading
from typing import List, Sequence, Union, Optional, Tuple

import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover - import error only at runtime
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    """
    Choose a sensible default device.

    Returns:
        "cuda" if available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceBERTEncoder:
    """
    Thin wrapper around SentenceTransformer for consistent encoding
    and cosine similarity computation.

    This is the main workhorse for:
      - encoding queries/documents/terms
      - computing cosine similarity
      - providing embeddings to other modules (features, expansion, reranking)

    Usage (inside codebase):

        from msmeqe.reranking.semantic_similarity import get_default_encoder

        encoder = get_default_encoder()
        q_emb = encoder.encode(["neural information retrieval"])
        d_emb = encoder.encode(["document about neural IR"])
        sims = encoder.cosine_similarity(q_emb, d_emb)

    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            model_name: HuggingFace / SentenceTransformers model name.
            device: "cpu", "cuda", or None to auto-detect.
            normalize: Whether to L2-normalize embeddings by default.
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install via `pip install sentence-transformers`."
            )

        self.model_name = model_name
        self.device = device or _auto_device()
        self.normalize = normalize

        logger.info(
            "Loading SentenceTransformer model '%s' on device '%s'",
            self.model_name,
            self.device,
        )
        self.model = SentenceTransformer(self.model_name, device=self.device)

    # ---------------------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Encode a single string or a list of strings into embeddings.

        Args:
            texts: String or list of strings.
            batch_size: Batch size for model.encode().
            normalize: Override default normalization behavior.

        Returns:
            Numpy array of shape (dim,) for single string,
            or (N, dim) for list of N strings.
        """
        if normalize is None:
            normalize = self.normalize

        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        with torch.no_grad():
            emb = self.model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        emb_np = emb.cpu().numpy()

        if single_input:
            return emb_np[0]
        return emb_np

    # ---------------------------------------------------------------------
    # Similarity utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.

        Args:
            a: Array of shape (N, dim) or (dim,)
            b: Array of shape (M, dim) or (dim,)

        Returns:
            Similarity matrix of shape (N, M) if both inputs are 2D,
            or scalar if both are 1D.
        """
        a = np.asarray(a)
        b = np.asarray(b)

        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)

        # L2-normalize (in case embeddings are not normalized upstream)
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)

        sims = np.dot(a_norm, b_norm.T)

        if sims.size == 1:
            return float(sims[0, 0])  # type: ignore[return-value]
        return sims

    def similarity_texts(
        self,
        text_a: Union[str, Sequence[str]],
        text_b: Union[str, Sequence[str]],
    ) -> np.ndarray:
        """
        Compute cosine similarity between two texts or sets of texts.

        Args:
            text_a: String or list of strings.
            text_b: String or list of strings.

        Returns:
            Similarity matrix or scalar as in cosine_similarity().
        """
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        return self.cosine_similarity(emb_a, emb_b)

    # ---------------------------------------------------------------------
    # Nearest neighbors over a vocabulary (for embedding_candidates)
    # ---------------------------------------------------------------------

    def top_k_neighbors(
        self,
        query_text: str,
        vocab_texts: Sequence[str],
        k: int = 30,
        return_embeddings: bool = False,
    ) -> Union[
        List[Tuple[str, float]],
        Tuple[List[Tuple[str, float]], np.ndarray, np.ndarray],
    ]:
        """
        Compute top-k nearest neighbors of query_text in a list of vocab terms.

        Used by `expansion/embedding_candidates.py` to get SBERT neighbors.

        Args:
            query_text: The query string.
            vocab_texts: List of vocabulary strings.
            k: Number of neighbors to return.
            return_embeddings: If True, also return (q_emb, vocab_embs).

        Returns:
            If return_embeddings=False:
                List of (term, similarity) sorted by similarity desc.
            If return_embeddings=True:
                (neighbors, query_embedding, vocab_embeddings)
        """
        if len(vocab_texts) == 0:
            return [] if not return_embeddings else ([], np.empty(0), np.empty(0))

        q_emb = self.encode(query_text)          # (dim,)
        v_emb = self.encode(list(vocab_texts))   # (V, dim)

        sims = self.cosine_similarity(q_emb, v_emb)  # (V,)
        if isinstance(sims, np.ndarray) and sims.ndim == 2:
            sims = sims[0]

        k = min(k, len(vocab_texts))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        neighbors = [(vocab_texts[i], float(sims[i])) for i in top_idx]

        if return_embeddings:
            return neighbors, q_emb, v_emb
        return neighbors


# -----------------------------------------------------------------------------
# Global singleton for reuse across modules (features, expansion, reranking)
# -----------------------------------------------------------------------------

_global_encoder_lock = threading.Lock()
_global_encoder: Optional[SentenceBERTEncoder] = None


def get_default_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    normalize: bool = True,
) -> SentenceBERTEncoder:
    """
    Get a process-wide singleton SentenceBERTEncoder.

    This is what other modules should call, e.g.:

        from msmeqe.reranking.semantic_similarity import get_default_encoder

        encoder = get_default_encoder()
        q_emb = encoder.encode("neural retrieval")
        d_emb = encoder.encode(["doc about neural IR", "other doc"])

    Args:
        model_name: Model name (ignored if encoder already created with same).
        device: Optional device override.
        normalize: Default normalization preference.

    Returns:
        A shared SentenceBERTEncoder instance.
    """
    global _global_encoder

    if _global_encoder is not None:
        return _global_encoder

    with _global_encoder_lock:
        if _global_encoder is None:
            _global_encoder = SentenceBERTEncoder(
                model_name=model_name,
                device=device,
                normalize=normalize,
            )
    return _global_encoder


# -----------------------------------------------------------------------------
# CLI utility for quick manual testing
# -----------------------------------------------------------------------------

def _main_cli() -> None:
    """
    Simple CLI for sanity-checking the encoder and similarity.

    Example:
        python -m msmeqe.reranking.semantic_similarity \\
            --text1 "neural retrieval" \\
            --text2 "dense passage retrieval"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Sentence-BERT semantic similarity utility"
    )
    parser.add_argument(
        "--text1",
        type=str,
        required=True,
        help="First text",
    )
    parser.add_argument(
        "--text2",
        type=str,
        required=True,
        help="Second text",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-BERT model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu or cuda (default: auto-detect)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    encoder = get_default_encoder(
        model_name=args.model_name,
        device=args.device,
    )

    sim = encoder.similarity_texts(args.text1, args.text2)
    if isinstance(sim, np.ndarray):
        sim_val = float(sim)
    else:
        sim_val = sim

    print(f"Text 1: {args.text1}")
    print(f"Text 2: {args.text2}")
    print(f"Cosine similarity: {sim_val:.4f}")


if __name__ == "__main__":  # pragma: no cover
    _main_cli()
