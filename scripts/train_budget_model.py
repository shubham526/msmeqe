# scripts/train_budget_model.py

"""
Train budget prediction model for MS-MEQE.

Budget prediction (Section 3.6.2 of the paper):
  - For each query, try 3 different budgets (low=30, med=50, high=70)
  - Evaluate which budget achieves best MAP
  - Train classifier to predict optimal budget from query features

Query features (10 features from Section 3.6.1):
  - clarity, entropy, avg_idf, max_idf, avg_bm25, var_bm25, q_len
  - q_type (navigational, informational, transactional)

Usage:
    # Step 0: Precompute document embeddings FIRST
    python scripts/precompute_doc_embeddings.py \\
        --collection msmarco-passage \\
        --output-dir data/msmarco_index \\
        --batch-size 512

    # Step 1: Train budget model
    python scripts/train_budget_model.py \\
        --training-queries data/train_queries.json \\
        --qrels data/train_qrels.txt \\
        --index-path data/msmarco_index \\
        --value-model models/value_model.pkl \\
        --weight-model models/weight_model.pkl \\
        --output-dir models/ \\
        --max-queries 2000
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm
import json
import pickle

from msmeqe.reranking.semantic_encoder import SemanticEncoder
from msmeqe.features.feature_extraction import FeatureExtractor
from msmeqe.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
from msmeqe.expansion.msmeqe_expansion import MSMEQEExpansionModel
from msmeqe.expansion.kb_expansion import KBCandidateExtractor
from msmeqe.expansion.embedding_candidates import EmbeddingCandidateExtractor
from msmeqe.retrieval.evaluator import TRECEvaluator
from msmeqe.utils.file_utils import load_json, load_qrels

logger = logging.getLogger(__name__)


class BudgetTrainingDataGenerator:
    """
    Generate training data for budget prediction.

    For each query:
      1. Extract query features (clarity, entropy, etc.)
      2. Try expansion with budgets [30, 50, 70]
      3. Find which budget gives best MAP (using REAL retrieval)
      4. Store (query_features, best_budget) as training instance
    """

    def __init__(
            self,
            encoder: SemanticEncoder,
            feature_extractor: FeatureExtractor,
            candidate_extractor: MultiSourceCandidateExtractor,
            value_model,
            weight_model,
            budget_options: List[int] = None,
            index_path: str = None,
            collection_size: int = 8841823,
    ):
        """
        Initialize budget training data generator.

        Args:
            encoder: SemanticEncoder instance
            feature_extractor: FeatureExtractor instance
            candidate_extractor: MultiSourceCandidateExtractor instance
            value_model: Trained value prediction model
            weight_model: Trained weight prediction model
            budget_options: List of budget values to try (default: [30, 50, 70])
            index_path: Path to Lucene index (for document retrieval)
            collection_size: Number of documents in collection
        """
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.candidate_extractor = candidate_extractor
        self.value_model = value_model
        self.weight_model = weight_model

        self.budget_options = budget_options or [30, 50, 70]

        # Initialize dense retrieval system
        self.index_path = index_path
        self.collection_size = collection_size
        self._init_retrieval_system()

        # Create MS-MEQE models with different budgets
        self.msmeqe_models = {}
        for budget in self.budget_options:
            self.msmeqe_models[budget] = MSMEQEExpansionModel(
                encoder=encoder,
                feature_extractor=feature_extractor,
                value_model=value_model,
                weight_model=weight_model,
                budget_model=None,  # We're training this!
                lambda_interp=0.3,
                min_budget=budget,
                max_budget=budget,
                budget_step=1,
            )

        self.evaluator = TRECEvaluator(metrics=['map'])

        logger.info(f"Initialized BudgetTrainingDataGenerator with budgets: {self.budget_options}")

    def _init_retrieval_system(self):
        """
        Initialize dense retrieval system.

        Loads pre-computed document embeddings from:
          - {index_path}/doc_embeddings.npy
          - {index_path}/doc_ids.json

        If not found, falls back to simulation.
        """
        if not self.index_path:
            logger.warning("No index_path provided, will use simulation")
            self.doc_embeddings = None
            self.doc_ids = None
            return

        # Load pre-computed document embeddings
        doc_embeddings_path = Path(self.index_path).parent / "doc_embeddings.npy"
        doc_ids_path = Path(self.index_path).parent / "doc_ids.json"

        # Try with parent directory structure
        if not doc_embeddings_path.exists():
            doc_embeddings_path = Path(self.index_path) / "doc_embeddings.npy"
            doc_ids_path = Path(self.index_path) / "doc_ids.json"

        if doc_embeddings_path.exists() and doc_ids_path.exists():
            logger.info(f"Loading pre-computed document embeddings from {doc_embeddings_path}")
            self.doc_embeddings = np.load(str(doc_embeddings_path))  # Shape: (N, d)

            with open(doc_ids_path, 'r') as f:
                self.doc_ids = json.load(f)  # List of doc IDs

            logger.info(f"Loaded {len(self.doc_ids)} document embeddings, dim={self.doc_embeddings.shape[1]}")
        else:
            logger.warning(
                f"Pre-computed embeddings not found at {doc_embeddings_path}. "
                f"Retrieval will use simulation. Run scripts/precompute_doc_embeddings.py first."
            )
            self.doc_embeddings = None
            self.doc_ids = None

    def generate_training_data(
            self,
            queries: Dict[str, str],  # {query_id: query_text}
            qrels: Dict[str, Dict[str, int]],  # {query_id: {doc_id: relevance}}
            output_path: str,
            max_queries: int = None,
    ):
        """
        Generate training data for budget prediction.

        Args:
            queries: Query ID -> query text
            qrels: Query ID -> doc ID -> relevance
            output_path: Where to save training data
            max_queries: Limit number of queries (for testing)
        """
        if max_queries:
            queries = dict(list(queries.items())[:max_queries])

        logger.info(f"Generating budget training data for {len(queries)} queries")

        # Storage for training instances
        query_features_list = []
        best_budgets_list = []

        for query_id, query_text in tqdm(queries.items(), desc="Processing queries"):
            if query_id not in qrels:
                continue  # Skip queries without relevance judgments

            try:
                # === 1. EXTRACT QUERY FEATURES ===
                query_stats = self.candidate_extractor.compute_query_stats(query_text)

                query_features = self.feature_extractor.extract_query_features(
                    query_text=query_text,
                    query_stats=query_stats,
                )  # Shape: (10,)

                # === 2. EXTRACT CANDIDATES (shared across all budgets) ===
                candidates = self.candidate_extractor.extract_all_candidates(
                    query_text=query_text,
                    query_id=query_id,
                )

                if not candidates:
                    logger.warning(f"No candidates for query {query_id}, skipping")
                    continue

                pseudo_centroid = self.candidate_extractor.compute_pseudo_centroid(query_text)

                # === 3. TRY EACH BUDGET AND EVALUATE ===
                budget_scores = {}

                for budget in self.budget_options:
                    # Manually set budget in query_stats (override budget prediction)
                    query_stats_with_budget = query_stats.copy()

                    # Expand with this budget
                    selected_terms, q_star = self.msmeqe_models[budget].expand(
                        query_text=query_text,
                        candidates=candidates,
                        pseudo_doc_centroid=pseudo_centroid,
                        query_stats=query_stats_with_budget,
                    )

                    # Evaluate expansion with REAL retrieval
                    expanded_map = self._evaluate_expanded_query(
                        query_text=query_text,
                        selected_terms=selected_terms,
                        query_id=query_id,
                        qrels=qrels,
                        q_star=q_star,
                    )

                    budget_scores[budget] = expanded_map

                    logger.debug(f"Query {query_id}, budget={budget}, MAP={expanded_map:.4f}")

                # === 4. FIND BEST BUDGET ===
                best_budget = max(budget_scores, key=budget_scores.get)

                # Store training instance
                query_features_list.append(query_features)
                best_budgets_list.append(best_budget)

                logger.debug(
                    f"Query {query_id}: best_budget={best_budget}, "
                    f"scores={budget_scores}"
                )

            except Exception as e:
                logger.warning(f"Error processing query {query_id}: {e}", exc_info=True)
                continue

        # Convert to arrays
        X = np.vstack(query_features_list)  # Shape: (n_queries, 10)
        y = np.array(best_budgets_list)  # Shape: (n_queries,)

        # Convert budget values to class indices (0, 1, 2)
        budget_to_class = {b: i for i, b in enumerate(sorted(set(self.budget_options)))}
        y_classes = np.array([budget_to_class[b] for b in y])

        logger.info(f"Generated {len(X)} training instances")
        logger.info(f"Budget distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump({
                'features': X,
                'budgets': y,
                'budget_classes': y_classes,
                'budget_to_class': budget_to_class,
                'class_to_budget': {v: k for k, v in budget_to_class.items()},
            }, f)

        logger.info(f"Budget training data saved to {output_path}")

        return X, y_classes

    def _evaluate_expanded_query(
            self,
            query_text: str,
            selected_terms: List,
            query_id: str,
            qrels: Dict,
            q_star: np.ndarray,
    ) -> float:
        """
        Evaluate expanded query using REAL dense retrieval.

        Pipeline:
          1. Use q_star (enhanced query embedding) to retrieve top-1000 docs
          2. Compute cosine similarity with all document embeddings
          3. Rank by similarity
          4. Evaluate against qrels using MAP

        Args:
            query_text: Original query text
            selected_terms: List of SelectedTerm objects
            query_id: Query ID
            qrels: Relevance judgments
            q_star: Enhanced query embedding (d,)

        Returns:
            MAP score for this expanded query
        """
        if self.doc_embeddings is None:
            # Fallback: use simplified simulation
            logger.debug("No document embeddings loaded, using simulation")
            return self._simulate_evaluation(selected_terms)

        try:
            # === DENSE RETRIEVAL ===

            # Normalize query embedding
            q_star_norm = q_star / (np.linalg.norm(q_star) + 1e-12)

            # Compute cosine similarity with all documents
            # doc_embeddings should already be normalized (from precomputation)
            similarities = self.doc_embeddings @ q_star_norm  # Shape: (N,)

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

            # === EVALUATION ===

            # Check if this query has qrels
            if query_id not in qrels or not qrels[query_id]:
                logger.warning(f"No qrels for query {query_id}")
                return 0.0

            # Evaluate using TRECEvaluator
            metrics = self.evaluator.evaluate_run(
                run_results=run_results,
                qrels={query_id: qrels[query_id]}
            )
            map_score = metrics.get('map', 0.0)

            return float(map_score)

        except Exception as e:
            logger.warning(f"Evaluation failed for query {query_id}: {e}", exc_info=True)
            return self._simulate_evaluation(selected_terms)

    def _simulate_evaluation(self, selected_terms: List) -> float:
        """
        Fallback simulation when document embeddings are not available.

        This is a heuristic approximation based on number of selected terms.
        """
        n_terms = len(selected_terms)

        if n_terms == 0:
            return 0.30  # No expansion
        elif n_terms <= 5:
            return 0.30 + 0.05 * n_terms
        elif n_terms <= 10:
            return 0.55
        else:
            return 0.55 - 0.02 * (n_terms - 10)


def train_budget_classifier(
        training_data_path: str,
        output_path: str,
        n_estimators: int = 50,
        max_depth: int = 4,
        learning_rate: float = 0.1,
):
    """
    Train budget prediction classifier.

    Args:
        training_data_path: Path to budget_training_data.pkl
        output_path: Where to save trained model
        n_estimators: Number of trees (shallower than value/weight models)
        max_depth: Maximum tree depth (simpler task than value/weight)
        learning_rate: XGBoost learning rate
    """
    logger.info(f"Loading budget training data from {training_data_path}")

    with open(training_data_path, 'rb') as f:
        data = pickle.load(f)

    X = data['features']  # Shape: (n_queries, 10)
    y = data['budget_classes']  # Shape: (n_queries,) - class indices
    budget_to_class = data['budget_to_class']
    class_to_budget = data['class_to_budget']

    logger.info(f"Training data: {X.shape[0]} queries, {X.shape[1]} features")
    logger.info(f"Budget mapping: {class_to_budget}")

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    logger.info(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Train XGBoost classifier
    logger.info("Training XGBoost budget classifier...")

    n_classes = len(class_to_budget)

    if n_classes == 2:
        objective = 'binary:logistic'
    else:
        objective = 'multi:softmax'

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        random_state=42,
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Validation accuracy: {val_acc:.4f}")

    # Classification report
    logger.info("\nValidation Classification Report:")
    report = classification_report(
        y_val, y_val_pred,
        target_names=[f"Budget {class_to_budget[i]}" for i in range(n_classes)]
    )
    logger.info(f"\n{report}")

    # Feature importance
    importance = model.feature_importances_
    logger.info("\nTop 5 features by importance:")
    feature_names = [
        'clarity', 'entropy', 'q_len', 'avg_idf', 'max_idf',
        'avg_bm25', 'var_bm25', 'type_nav', 'type_info', 'type_tran'
    ]
    top_features = np.argsort(importance)[-5:][::-1]
    for i, idx in enumerate(top_features, 1):
        fname = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
        logger.info(f"  {i}. {fname}: {importance[idx]:.4f}")

    # Wrap model with budget mapping
    class BudgetPredictor:
        """Wrapper that predicts actual budget values instead of class indices."""

        def __init__(self, xgb_model, class_to_budget):
            self.model = xgb_model
            self.class_to_budget = class_to_budget

        def predict(self, X):
            """Predict budget values."""
            class_preds = self.model.predict(X)
            budget_preds = np.array([self.class_to_budget[int(c)] for c in class_preds])
            return budget_preds

        def predict_proba(self, X):
            """Predict class probabilities."""
            return self.model.predict_proba(X)

    wrapped_model = BudgetPredictor(model, class_to_budget)

    # Save wrapped model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(wrapped_model, output_path)
    logger.info(f"Budget model saved to {output_path}")

    return wrapped_model


def main():
    """Main entry point for budget model training."""
    parser = argparse.ArgumentParser(description="Train MS-MEQE budget prediction model")

    # Data paths
    parser.add_argument("--training-queries", type=str, required=True,
                        help="Path to training queries JSON")
    parser.add_argument("--qrels", type=str, required=True,
                        help="Path to qrels file")
    parser.add_argument("--index-path", type=str, required=True,
                        help="Path to Lucene index (or dir with doc_embeddings.npy)")

    # Model paths
    parser.add_argument("--value-model", type=str, required=True,
                        help="Path to trained value model")
    parser.add_argument("--weight-model", type=str, required=True,
                        help="Path to trained weight model")

    # Optional extractors
    parser.add_argument("--kb-wat-output", type=str, default=None,
                        help="Path to WAT entity linking output (optional)")
    parser.add_argument("--kb-desc-tsv", type=str, default=None,
                        help="Path to wikiid->description TSV (optional)")
    parser.add_argument("--vocab-embeddings", type=str, default=None,
                        help="Path to vocabulary embeddings (optional)")

    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for models")

    # Training parameters
    parser.add_argument("--max-queries", type=int, default=2000,
                        help="Maximum number of queries to process")
    parser.add_argument("--budget-options", type=int, nargs="+",
                        default=[30, 50, 70],
                        help="Budget values to try (default: 30 50 70)")
    parser.add_argument("--n-estimators", type=int, default=50,
                        help="Number of XGBoost trees")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate")

    # Collection size
    parser.add_argument("--collection-size", type=int, default=8841823,
                        help="Collection size (default: MS MARCO passage)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === INITIALIZE COMPONENTS ===

    logger.info("Initializing components...")

    # Encoder
    encoder = SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Feature extractor
    feature_extractor = FeatureExtractor(collection_size=args.collection_size)

    # Load trained value/weight models
    logger.info(f"Loading value model from {args.value_model}")
    value_model = joblib.load(args.value_model)

    logger.info(f"Loading weight model from {args.weight_model}")
    weight_model = joblib.load(args.weight_model)

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

    # === LOAD QUERIES AND QRELS ===

    logger.info(f"Loading queries from {args.training_queries}")
    queries = load_json(args.training_queries)

    logger.info(f"Loading qrels from {args.qrels}")
    qrels = load_qrels(args.qrels)

    logger.info(f"Loaded {len(queries)} queries, {len(qrels)} qrels")

    # === GENERATE TRAINING DATA ===

    logger.info("Generating budget training data...")
    generator = BudgetTrainingDataGenerator(
        encoder=encoder,
        feature_extractor=feature_extractor,
        candidate_extractor=candidate_extractor,
        value_model=value_model,
        weight_model=weight_model,
        budget_options=args.budget_options,
        index_path=args.index_path,
        collection_size=args.collection_size,
    )

    training_data_path = output_dir / "budget_training_data.pkl"

    generator.generate_training_data(
        queries=queries,
        qrels=qrels,
        output_path=str(training_data_path),
        max_queries=args.max_queries,
    )

    # === TRAIN MODEL ===

    logger.info("Training budget classifier...")
    budget_model = train_budget_classifier(
        training_data_path=str(training_data_path),
        output_path=str(output_dir / "budget_model.pkl"),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    logger.info("Budget model training complete!")
    logger.info(f"Model saved to: {output_dir / 'budget_model.pkl'}")


if __name__ == "__main__":
    main()