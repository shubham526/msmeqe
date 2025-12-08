# scripts/create_training_data_msmeqe.py

"""
Generate training data for MS-MEQE value/weight models.

For each query:
  1. Extract candidates from all sources
  2. For each candidate term:
     - Expand query: q' = q + term
     - Retrieve documents using enhanced query embedding
     - Evaluate: Δ_MAP = MAP(q') - MAP(q)
     - Extract features using FeatureExtractor
     - Store: (features, Δ_MAP) as training instance

This generates the training data described in Sections 3.3.2 and 3.4.2.

Usage:
    # Step 0: Precompute document embeddings FIRST
    python scripts/precompute_doc_embeddings.py \\
        --collection msmarco-passage \\
        --output-dir data/msmarco_index \\
        --batch-size 512

    # Step 1: Generate training data
    python scripts/create_training_data_msmeqe.py \\
        --training-queries data/train_queries.json \\
        --qrels data/train_qrels.txt \\
        --index-path data/msmarco_index \\
        --output-dir data/training_data \\
        --max-queries 5000 \\
        --max-candidates-per-query 20
"""

import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pickle
from tqdm import tqdm
import json
import time



# FIX: Corrected import paths from src.* to msmeqe.*
from src.reranking.semantic_encoder import SemanticEncoder
from src.features.feature_extraction import FeatureExtractor, create_candidate_stats_dict
from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from src.expansion.kb_expansion import KBCandidateExtractor
from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
from src.expansion.msmeqe_expansion import CandidateTerm
from src.retrieval.evaluator import TRECEvaluator
from src.utils.file_utils import load_json, load_qrels

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """
    Generate training data for value/weight models.
    """

    def __init__(
            self,
            encoder: SemanticEncoder,
            feature_extractor: FeatureExtractor,
            candidate_extractor: MultiSourceCandidateExtractor,
            index_path: str,
            collection_size: int,
    ):
        """
        Initialize training data generator.
        """
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.candidate_extractor = candidate_extractor
        self.index_path = index_path
        self.N = collection_size

        self.evaluator = TRECEvaluator(metrics=['map'])

        # Initialize dense retrieval system
        self._init_retrieval_system()

        logger.info("Initialized TrainingDataGenerator")

    def _init_retrieval_system(self):
        """
        Initialize dense retrieval system with validation.
        """
        if not self.index_path:
            raise ValueError("index_path is required for training data generation")

        # Load pre-computed document embeddings
        # Handle cases where index_path is the dir or the index file itself
        base_path = Path(self.index_path)
        if base_path.is_file():
            base_path = base_path.parent

        doc_embeddings_path = base_path / "doc_embeddings.npy"
        doc_ids_path = base_path / "doc_ids.json"

        if not doc_embeddings_path.exists() or not doc_ids_path.exists():
            raise FileNotFoundError(
                f"Document embeddings not found at {doc_embeddings_path}. "
                f"Please run scripts/precompute_doc_embeddings.py first."
            )

        logger.info(f"Loading pre-computed document embeddings from {doc_embeddings_path}")

        # FIX: Added try/except block for loading validation
        try:
            self.doc_embeddings = np.load(str(doc_embeddings_path))  # Shape: (N, d)
        except Exception as e:
            raise ValueError(f"Corrupt or invalid .npy file: {e}")

        with open(doc_ids_path, 'r') as f:
            self.doc_ids = json.load(f)  # List of doc IDs

        # FIX: Added strict validation of shapes
        if self.doc_embeddings.shape[0] != len(self.doc_ids):
            raise ValueError(
                f"Data Mismatch: Embeddings has {self.doc_embeddings.shape[0]} rows "
                f"but doc_ids has {len(self.doc_ids)} entries."
            )

        logger.info(
            f"Loaded {len(self.doc_ids)} document embeddings, "
            f"dim={self.doc_embeddings.shape[1]}"
        )

        # Optimization: Normalize once at load time for efficiency
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        self.doc_embeddings = self.doc_embeddings / norms

    def generate_training_data(
            self,
            queries: Dict[str, str],
            qrels: Dict[str, Dict[str, int]],
            output_dir: str,
            max_queries: int = None,
            max_candidates_per_query: int = 20,
            checkpoint_every: int = 100
    ):
        """
        Generate training data for all queries with checkpointing.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # FIX: Added Progress Checkpointing logic
        value_checkpoint_path = output_dir / "value_checkpoint.pkl"
        weight_checkpoint_path = output_dir / "weight_checkpoint.pkl"

        value_features = []
        value_targets = []
        weight_features = []
        weight_targets = []
        processed_ids = set()

        # Resume from checkpoint if exists
        if value_checkpoint_path.exists() and weight_checkpoint_path.exists():
            logger.info("Found checkpoints. Resuming generation...")
            with open(value_checkpoint_path, 'rb') as f:
                val_data = pickle.load(f)
                value_features = list(val_data['features'])
                value_targets = list(val_data['targets'])
                processed_ids = val_data['processed_ids']

            with open(weight_checkpoint_path, 'rb') as f:
                wgt_data = pickle.load(f)
                weight_features = list(wgt_data['features'])
                weight_targets = list(wgt_data['targets'])
            logger.info(f"Resumed: {len(processed_ids)} queries already processed.")

        # Filter queries
        remaining_queries = {k: v for k, v in queries.items() if k not in processed_ids}
        if max_queries:
            limit = max(0, max_queries - len(processed_ids))
            remaining_queries = dict(list(remaining_queries.items())[:limit])

        logger.info(f"Generating training data for {len(remaining_queries)} new queries")

        processed_count = 0

        # FIX: Wrapped in try...finally to ensure data isn't lost on crash
        try:
            for query_id, query_text in tqdm(remaining_queries.items(), desc="Processing queries"):
                if query_id not in qrels:
                    continue

                try:
                    # === 1. EXTRACT CANDIDATES ===
                    candidates = self.candidate_extractor.extract_all_candidates(
                        query_text=query_text,
                        query_id=query_id,
                    )

                    if not candidates:
                        continue

                    # Subsample candidates (for efficiency)
                    if len(candidates) > max_candidates_per_query:
                        candidates = self._subsample_candidates(
                            candidates,
                            max_candidates_per_query
                        )

                    # === 2. GET EMBEDDINGS ===
                    query_emb = self.encoder.encode([query_text])[0]
                    term_embs = self.encoder.encode([c.term for c in candidates])
                    pseudo_centroid = self.candidate_extractor.compute_pseudo_centroid(query_text)

                    # === 3. EVALUATE BASELINE (original query) ===
                    baseline_map = self._evaluate_query_embedding(
                        query_embedding=query_emb,
                        query_id=query_id,
                        qrels=qrels,
                    )

                    # === 4. FOR EACH CANDIDATE, COMPUTE GROUND TRUTH ===
                    for i, cand in enumerate(candidates):
                        # --- Feature Extraction ---
                        stats = create_candidate_stats_dict(
                            rm3_score=cand.rm3_score,
                            tf_pseudo=cand.tf_pseudo,
                            coverage_pseudo=cand.coverage_pseudo,
                            df=cand.df,
                            cf=cand.cf,
                            native_rank=cand.native_rank,
                            native_score=cand.native_score,
                        )

                        val_features = self.feature_extractor.extract_value_features(
                            candidate_term=cand.term,
                            candidate_source=cand.source,
                            candidate_stats=stats,
                            query_text=query_text,
                            query_embedding=query_emb,
                            term_embedding=term_embs[i],
                            pseudo_centroid=pseudo_centroid,
                        )

                        wt_features = self.feature_extractor.extract_weight_features(
                            candidate_term=cand.term,
                            candidate_source=cand.source,
                            candidate_stats=stats,
                            query_text=query_text,
                            query_embedding=query_emb,
                            term_embedding=term_embs[i],
                        )

                        # --- Evaluate Expansion ---
                        # Use weighted average q' = 0.7*q + 0.3*term
                        expanded_query_emb = 0.7 * query_emb + 0.3 * term_embs[i]

                        expanded_map = self._evaluate_query_embedding(
                            query_embedding=expanded_query_emb,
                            query_id=query_id,
                            qrels=qrels,
                        )

                        delta_map = expanded_map - baseline_map

                        # Store Instances
                        value_features.append(val_features)
                        value_targets.append(delta_map)

                        weight_features.append(wt_features)
                        # Risk is positive if map decreases
                        weight_targets.append(max(0.0, -delta_map))

                    # Mark done
                    processed_ids.add(query_id)
                    processed_count += 1

                    # Save Checkpoint
                    if processed_count % checkpoint_every == 0:
                        self._save_checkpoints(
                            output_dir,
                            value_features, value_targets,
                            weight_features, weight_targets,
                            processed_ids
                        )

                except Exception as e:
                    logger.warning(f"Error processing query {query_id}: {e}")
                    continue

        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Saving progress...")
        finally:
            # Final save of raw data arrays
            self._save_final_data(
                output_dir,
                value_features, value_targets,
                weight_features, weight_targets
            )
            # Remove checkpoints on clean exit if desired, or keep them
            logger.info("Training data generation stopped/finished.")

    def _save_checkpoints(self, out_dir, vf, vt, wf, wt, pids):
        """Helper to save intermediate state."""
        try:
            with open(out_dir / "value_checkpoint.pkl", 'wb') as f:
                pickle.dump({'features': vf, 'targets': vt, 'processed_ids': pids}, f)
            with open(out_dir / "weight_checkpoint.pkl", 'wb') as f:
                pickle.dump({'features': wf, 'targets': wt}, f)
            logger.info(f"Checkpoint saved. Processed: {len(pids)}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _save_final_data(self, out_dir, vf, vt, wf, wt):
        """Convert lists to numpy and save final .pkl files."""
        if not vf:
            return

        logger.info("Saving final numpy arrays...")
        X_val = np.vstack(vf)
        y_val = np.array(vt, dtype=np.float32)

        X_wgt = np.vstack(wf)
        y_wgt = np.array(wt, dtype=np.float32)

        with open(out_dir / "value_training_data.pkl", 'wb') as f:
            pickle.dump({'features': X_val, 'targets': y_val}, f)

        with open(out_dir / "weight_training_data.pkl", 'wb') as f:
            pickle.dump({'features': X_wgt, 'targets': y_wgt}, f)

        logger.info(f"Saved {len(X_val)} instances to {out_dir}")

    def _subsample_candidates(
            self,
            candidates: List[CandidateTerm],
            max_k: int
    ) -> List[CandidateTerm]:
        """
        Subsample candidates to max_k.
        Strategy: Take top candidates from each source, then fill remainder.
        """
        from collections import defaultdict
        by_source = defaultdict(list)

        for cand in candidates:
            by_source[cand.source].append(cand)

        # Sort each source by native_score (descending)
        for source in by_source:
            by_source[source].sort(key=lambda c: c.native_score, reverse=True)

        num_sources = len(by_source)
        if num_sources == 0:
            return []

        k_per_source = max(1, max_k // num_sources)

        subsampled = []
        for source in by_source:
            subsampled.extend(by_source[source][:k_per_source])

        # If we have room, add more from top scoring remaining
        if len(subsampled) < max_k:
            remaining = max_k - len(subsampled)
            all_remaining = [
                c for source in by_source
                for c in by_source[source][k_per_source:]
            ]
            all_remaining.sort(key=lambda c: c.native_score, reverse=True)
            subsampled.extend(all_remaining[:remaining])

        return subsampled

    def _evaluate_query_embedding(
            self,
            query_embedding: np.ndarray,
            query_id: str,
            qrels: Dict[str, Dict[str, int]],
    ) -> float:
        """
        Evaluate a query embedding using REAL dense retrieval.
        FIX: Updated to perform correct matrix multiplication retrieval.
        """
        try:
            # Normalize query embedding
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

            # Compute cosine similarity with all documents
            # self.doc_embeddings is already normalized in _init_retrieval_system
            similarities = np.dot(self.doc_embeddings, q_norm)  # Shape: (N,)

            # Get top-1000 documents efficiently
            top_k = min(1000, len(similarities))

            # Use argpartition for efficiency on large arrays
            if len(similarities) > top_k:
                top_indices_unsorted = np.argpartition(-similarities, top_k)[:top_k]
                # Sort only the top k
                top_indices = top_indices_unsorted[np.argsort(-similarities[top_indices_unsorted])]
            else:
                top_indices = np.argsort(-similarities)

            # Build run results for TREC evaluator
            retrieved_docs = [
                (self.doc_ids[idx], float(similarities[idx]))
                for idx in top_indices
            ]

            run_results = {query_id: retrieved_docs}

            # Evaluate
            if query_id not in qrels or not qrels[query_id]:
                return 0.0

            metrics = self.evaluator.evaluate_run(
                run_results=run_results,
                qrels={query_id: qrels[query_id]}
            )

            map_score = metrics.get('map', 0.0)
            return float(map_score)

        except Exception as e:
            logger.warning(f"Evaluation failed for query {query_id}: {e}")
            return 0.0


def main():
    """Main entry point for training data generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate MS-MEQE training data with REAL retrieval"
    )

    # Data paths
    parser.add_argument("--training-queries", type=str, required=True,
                        help="Path to training queries JSON")
    parser.add_argument("--qrels", type=str, required=True,
                        help="Path to qrels file")
    parser.add_argument("--index-path", type=str, required=True,
                        help="Path to dir containing doc_embeddings.npy")

    # Optional extractors
    parser.add_argument("--kb-wat-output", type=str, default=None,
                        help="Path to WAT entity linking output (optional)")
    parser.add_argument("--kb-desc-tsv", type=str, default=None,
                        help="Path to wikiid->description TSV (optional)")
    parser.add_argument("--vocab-embeddings", type=str, default=None,
                        help="Path to vocabulary embeddings (optional)")

    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for training data")

    # Processing parameters
    parser.add_argument("--max-queries", type=int, default=5000,
                        help="Maximum number of queries to process")
    parser.add_argument("--max-candidates-per-query", type=int, default=20,
                        help="Maximum candidates per query (subsample)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save checkpoint every N queries")

    # Collection size
    parser.add_argument("--collection-size", type=int, default=8841823,
                        help="Collection size (default: MS MARCO passage)")

    # Encoder
    parser.add_argument("--model-name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-BERT model name")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    )

    # === INITIALIZE COMPONENTS ===

    logger.info("Initializing components...")

    # Encoder
    logger.info(f"Loading encoder: {args.model_name}")
    encoder = SemanticEncoder(model_name=args.model_name)

    # Feature extractor
    feature_extractor = FeatureExtractor(collection_size=args.collection_size)

    # Initialize KB extractor (optional)
    kb_extractor = None
    if args.kb_wat_output:
        logger.info("Initializing KB extractor...")
        kb_extractor = KBCandidateExtractor(
            wat_output_path=args.kb_wat_output,
            wikiid_desc_path=args.kb_desc_tsv,
        )

    # Initialize embedding extractor (optional)
    emb_extractor = None
    if args.vocab_embeddings:
        logger.info("Initializing embedding extractor...")
        emb_extractor = EmbeddingCandidateExtractor(
            encoder=encoder,
            vocab_path=args.vocab_embeddings,
        )

    # Initialize candidate extractor
    logger.info("Initializing candidate extractor...")
    candidate_extractor = MultiSourceCandidateExtractor(
        index_path=args.index_path,
        encoder=encoder,
        kb_extractor=kb_extractor,
        emb_extractor=emb_extractor,
    )

    # Initialize training data generator
    logger.info("Initializing training data generator...")
    generator = TrainingDataGenerator(
        encoder=encoder,
        feature_extractor=feature_extractor,
        candidate_extractor=candidate_extractor,
        index_path=args.index_path,
        collection_size=args.collection_size,
    )

    # === LOAD QUERIES AND QRELS ===

    logger.info(f"Loading queries from {args.training_queries}")
    queries = load_json(args.training_queries)

    logger.info(f"Loading qrels from {args.qrels}")
    qrels = load_qrels(args.qrels)

    logger.info(f"Loaded {len(queries)} queries, {len(qrels)} qrels")

    # === GENERATE TRAINING DATA ===

    logger.info("Generating training data with REAL retrieval...")
    logger.info("This may take a while (several hours for 5000 queries)...")

    generator.generate_training_data(
        queries=queries,
        qrels=qrels,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        max_candidates_per_query=args.max_candidates_per_query,
        checkpoint_every=args.checkpoint_every
    )

    logger.info("=" * 60)
    logger.info("TRAINING DATA GENERATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Files created:")
    logger.info(f"  - value_training_data.pkl")
    logger.info(f"  - weight_training_data.pkl")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Train value/weight models:")
    logger.info(f"     python scripts/train_value_weight_models.py \\")
    logger.info(f"       --value-data {args.output_dir}/value_training_data.pkl \\")
    logger.info(f"       --weight-data {args.output_dir}/weight_training_data.pkl \\")
    logger.info(f"       --output-dir models/")


if __name__ == "__main__":
    main()