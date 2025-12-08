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
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pickle
from tqdm import tqdm
import json

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

    For each query:
      - Extract candidates from all sources
      - For each candidate:
          * Compute features
          * Expand query with this candidate
          * Retrieve and evaluate (REAL retrieval!)
          * Store (features, Δ_MAP)
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

        Args:
            encoder: SemanticEncoder instance
            feature_extractor: FeatureExtractor instance
            candidate_extractor: MultiSourceCandidateExtractor instance
            index_path: Path to Lucene index (or dir with doc_embeddings.npy)
            collection_size: Total number of documents
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
        Initialize dense retrieval system.

        Loads pre-computed document embeddings from:
          - {index_path}/doc_embeddings.npy
          - {index_path}/doc_ids.json
        """
        if not self.index_path:
            raise ValueError("index_path is required for training data generation")

        # Load pre-computed document embeddings
        doc_embeddings_path = Path(self.index_path).parent / "doc_embeddings.npy"
        doc_ids_path = Path(self.index_path).parent / "doc_ids.json"

        # Try with parent directory structure
        if not doc_embeddings_path.exists():
            doc_embeddings_path = Path(self.index_path) / "doc_embeddings.npy"
            doc_ids_path = Path(self.index_path) / "doc_ids.json"

        if not doc_embeddings_path.exists() or not doc_ids_path.exists():
            raise FileNotFoundError(
                f"Document embeddings not found at {doc_embeddings_path}. "
                f"Please run scripts/precompute_doc_embeddings.py first."
            )

        logger.info(f"Loading pre-computed document embeddings from {doc_embeddings_path}")
        self.doc_embeddings = np.load(str(doc_embeddings_path))  # Shape: (N, d)

        with open(doc_ids_path, 'r') as f:
            self.doc_ids = json.load(f)  # List of doc IDs

        logger.info(
            f"Loaded {len(self.doc_ids)} document embeddings, "
            f"dim={self.doc_embeddings.shape[1]}"
        )

    def generate_training_data(
        self,
        queries: Dict[str, str],  # {query_id: query_text}
        qrels: Dict[str, Dict[str, int]],  # {query_id: {doc_id: relevance}}
        output_dir: str,
        max_queries: int = None,
        max_candidates_per_query: int = 20,
    ):
        """
        Generate training data for all queries.

        Args:
            queries: Query ID -> query text
            qrels: Query ID -> doc ID -> relevance
            output_dir: Where to save training data
            max_queries: Limit number of queries (for testing)
            max_candidates_per_query: Subsample candidates (for efficiency)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Limit queries if requested
        if max_queries and len(queries) > max_queries:
            queries = dict(list(queries.items())[:max_queries])

        # Storage for training instances
        value_features = []
        value_targets = []
        weight_features = []
        weight_targets = []

        logger.info(f"Generating training data for {len(queries)} queries")

        for query_id, query_text in tqdm(queries.items(), desc="Processing queries"):
            if query_id not in qrels:
                logger.debug(f"Skipping query {query_id} (no qrels)")
                continue  # Skip queries without relevance judgments

            try:
                # === 1. EXTRACT CANDIDATES ===
                candidates = self.candidate_extractor.extract_all_candidates(
                    query_text=query_text,
                    query_id=query_id,
                )

                if not candidates:
                    logger.debug(f"No candidates for query {query_id}")
                    continue

                # Subsample candidates (for efficiency)
                if len(candidates) > max_candidates_per_query:
                    candidates = self._subsample_candidates(
                        candidates,
                        max_candidates_per_query
                    )

                logger.debug(f"Query {query_id}: {len(candidates)} candidates")

                # === 2. GET EMBEDDINGS ===
                query_emb = self.encoder.encode([query_text])[0]  # (d,)
                term_embs = self.encoder.encode([c.term for c in candidates])  # (m, d)
                pseudo_centroid = self.candidate_extractor.compute_pseudo_centroid(query_text)

                # === 3. EVALUATE BASELINE (original query) ===
                baseline_map = self._evaluate_query_embedding(
                    query_embedding=query_emb,
                    query_id=query_id,
                    qrels=qrels,
                )

                logger.debug(f"Query {query_id}: baseline MAP={baseline_map:.4f}")

                # === 4. FOR EACH CANDIDATE, COMPUTE GROUND TRUTH ===
                for i, cand in enumerate(candidates):
                    try:
                        # === VALUE FEATURES & TARGET ===

                        # Extract features
                        val_features = self.feature_extractor.extract_value_features(
                            candidate_term=cand.term,
                            candidate_source=cand.source,
                            candidate_stats=create_candidate_stats_dict(
                                rm3_score=cand.rm3_score,
                                tf_pseudo=cand.tf_pseudo,
                                coverage_pseudo=cand.coverage_pseudo,
                                df=cand.df,
                                cf=cand.cf,
                                native_rank=cand.native_rank,
                                native_score=cand.native_score,
                            ),
                            query_text=query_text,
                            query_embedding=query_emb,
                            term_embedding=term_embs[i],
                            pseudo_centroid=pseudo_centroid,
                        )

                        # Compute expanded query embedding (simple: weighted average)
                        # For training, we use a simple expansion:
                        # q' = 0.7 * q + 0.3 * term
                        expanded_query_emb = 0.7 * query_emb + 0.3 * term_embs[i]

                        # Evaluate expanded query
                        expanded_map = self._evaluate_query_embedding(
                            query_embedding=expanded_query_emb,
                            query_id=query_id,
                            qrels=qrels,
                        )

                        # Compute ground truth: Δ_MAP
                        delta_map = expanded_map - baseline_map

                        # Store value training instance
                        value_features.append(val_features)
                        value_targets.append(delta_map)

                        # === WEIGHT FEATURES & TARGET ===

                        wt_features = self.feature_extractor.extract_weight_features(
                            candidate_term=cand.term,
                            candidate_source=cand.source,
                            candidate_stats=create_candidate_stats_dict(
                                rm3_score=cand.rm3_score,
                                tf_pseudo=cand.tf_pseudo,
                                coverage_pseudo=cand.coverage_pseudo,
                                df=cand.df,
                                cf=cand.cf,
                                native_rank=cand.native_rank,
                                native_score=cand.native_score,
                            ),
                            query_text=query_text,
                            query_embedding=query_emb,
                            term_embedding=term_embs[i],
                        )

                        # Weight target: risk = -Δ_MAP if negative, 0 if positive
                        # Normalize to [0, 1] range
                        weight_target = max(0.0, -delta_map)

                        # Store weight training instance
                        weight_features.append(wt_features)
                        weight_targets.append(weight_target)

                        logger.debug(
                            f"  Candidate '{cand.term}' ({cand.source}): "
                            f"Δ_MAP={delta_map:.4f}"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Error processing candidate '{cand.term}' for query {query_id}: {e}"
                        )
                        continue

            except Exception as e:
                logger.warning(f"Error processing query {query_id}: {e}", exc_info=True)
                continue

        # Convert to arrays
        if not value_features:
            raise ValueError("No training instances generated! Check your data.")

        value_features = np.vstack(value_features)
        value_targets = np.array(value_targets, dtype=np.float32)
        weight_features = np.vstack(weight_features)
        weight_targets = np.array(weight_targets, dtype=np.float32)

        logger.info(f"Generated {len(value_features)} training instances")
        logger.info(f"Value target stats: mean={value_targets.mean():.4f}, std={value_targets.std():.4f}")
        logger.info(f"Weight target stats: mean={weight_targets.mean():.4f}, std={weight_targets.std():.4f}")

        # Save
        logger.info(f"Saving training data to {output_dir}")

        with open(output_dir / "value_training_data.pkl", 'wb') as f:
            pickle.dump({
                'features': value_features,
                'targets': value_targets,
            }, f)

        with open(output_dir / "weight_training_data.pkl", 'wb') as f:
            pickle.dump({
                'features': weight_features,
                'targets': weight_targets,
            }, f)

        logger.info(f"Training data saved to {output_dir}")

        # Save statistics
        stats = {
            'num_instances': len(value_features),
            'num_queries': len(queries),
            'value_stats': {
                'mean': float(value_targets.mean()),
                'std': float(value_targets.std()),
                'min': float(value_targets.min()),
                'max': float(value_targets.max()),
            },
            'weight_stats': {
                'mean': float(weight_targets.mean()),
                'std': float(weight_targets.std()),
                'min': float(weight_targets.min()),
                'max': float(weight_targets.max()),
            },
        }

        with open(output_dir / "training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("Training data generation complete!")

    def _subsample_candidates(
        self,
        candidates: List[CandidateTerm],
        max_k: int
    ) -> List[CandidateTerm]:
        """
        Subsample candidates to max_k.

        Strategy:
          - Take top candidates from each source
          - Prioritize by native_score within each source

        Args:
            candidates: List of CandidateTerm objects
            max_k: Maximum number to keep

        Returns:
            Subsampled list of candidates
        """
        from collections import defaultdict

        by_source = defaultdict(list)

        for cand in candidates:
            by_source[cand.source].append(cand)

        # Sort each source by native_score (descending)
        for source in by_source:
            by_source[source].sort(key=lambda c: c.native_score, reverse=True)

        # Take top-k/num_sources from each source
        num_sources = len(by_source)
        k_per_source = max(1, max_k // num_sources)

        subsampled = []
        for source in by_source:
            subsampled.extend(by_source[source][:k_per_source])

        # If we have room, add more from top source
        if len(subsampled) < max_k:
            remaining = max_k - len(subsampled)
            all_remaining = [
                c for source in by_source
                for c in by_source[source][k_per_source:]
            ]
            all_remaining.sort(key=lambda c: c.native_score, reverse=True)
            subsampled.extend(all_remaining[:remaining])

        logger.debug(
            f"Subsampled {len(candidates)} → {len(subsampled)} candidates "
            f"(target={max_k})"
        )

        return subsampled

    def _evaluate_query_embedding(
        self,
        query_embedding: np.ndarray,
        query_id: str,
        qrels: Dict[str, Dict[str, int]],
    ) -> float:
        """
        Evaluate a query embedding using REAL dense retrieval.

        Pipeline:
          1. Normalize query embedding
          2. Compute cosine similarity with all documents
          3. Retrieve top-1000 documents
          4. Evaluate against qrels using MAP

        Args:
            query_embedding: Query embedding vector (d,)
            query_id: Query ID
            qrels: Relevance judgments

        Returns:
            MAP score
        """
        try:
            # Normalize query embedding
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

            # Compute cosine similarity with all documents
            similarities = self.doc_embeddings @ q_norm  # Shape: (N,)

            # Get top-1000 documents
            top_k = min(1000, len(similarities))
            top_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

            # Build run results
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
                        help="Path to Lucene index (or dir with doc_embeddings.npy)")

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
    )

    logger.info("=" * 60)
    logger.info("TRAINING DATA GENERATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Files created:")
    logger.info(f"  - value_training_data.pkl")
    logger.info(f"  - weight_training_data.pkl")
    logger.info(f"  - training_stats.json")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Train value/weight models:")
    logger.info(f"     python scripts/train_value_weight_models.py \\")
    logger.info(f"       --value-data {args.output_dir}/value_training_data.pkl \\")
    logger.info(f"       --weight-data {args.output_dir}/weight_training_data.pkl \\")
    logger.info(f"       --output-dir models/")


if __name__ == "__main__":
    main()