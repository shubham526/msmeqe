#!/usr/bin/env python3
# scripts/run_msmeqe_evaluation.py

"""
End-to-end MS-MEQE evaluation with full configuration support.

Complete implementation matching paper requirements:
  - Multi-source candidate extraction (RM3/KB/Embeddings)
  - Value/weight/budget prediction
  - Unbounded knapsack optimization
  - Dense retrieval with enhanced query embeddings
  - TREC run file output
  - Comprehensive evaluation and analysis

Usage:
    # Full evaluation
    python scripts/run_msmeqe_evaluation.py \\
        --index data/msmarco_index \\
        --topics data/queries.tsv \\
        --qrels data/qrels.txt \\
        --value-model models/value_model.pkl \\
        --weight-model models/weight_model.pkl \\
        --budget-model models/budget_model.pkl \\
        --sbert-model sentence-transformers/all-MiniLM-L6-v2 \\
        --emb-vocab data/vocab_embeddings.pkl \\
        --kb-candidates data/kb_candidates.jsonl \\
        --output runs/msmeqe.msmarco.txt \\
        --run-name MS-MEQE \\
        --log-per-query results/per_query_stats.jsonl \\
        --collection-size 8841823 \\
        --lucene-path /path/to/lucene/*
"""

import logging
import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import joblib
from src.utils.lucene_utils import initialize_lucene

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_queries_from_file(
        topics_file: str,
        max_queries: Optional[int] = None,
) -> Dict[str, str]:
    """
    Load queries from file.

    Supports:
      - TSV: qid\tquery_text
      - JSON: {"qid": "query_text"}
      - JSONL: one JSON per line

    Args:
        topics_file: Path to topics file
        max_queries: Limit number of queries

    Returns:
        Dictionary {query_id: query_text}
    """
    logger.info(f"Loading queries from: {topics_file}")

    topics_path = Path(topics_file)
    queries = {}

    if topics_path.suffix == '.json':
        # JSON format
        with open(topics_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                queries = data
            elif isinstance(data, list):
                queries = {str(i): q for i, q in enumerate(data)}

    elif topics_path.suffix == '.jsonl':
        # JSONL format
        with open(topics_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                qid = item.get('qid') or item.get('query_id') or item.get('id')
                qtext = item.get('query') or item.get('text')
                queries[str(qid)] = qtext

    else:
        # TSV format (default)
        with open(topics_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid, qtext = parts[0], parts[1]
                    queries[qid] = qtext

    if max_queries:
        queries = dict(list(queries.items())[:max_queries])

    logger.info(f"Loaded {len(queries)} queries")
    return queries


def load_qrels_from_file(qrels_file: str) -> Dict[str, Dict[str, int]]:
    """
    Load qrels from TREC format file.

    Format: qid iteration docid relevance

    Args:
        qrels_file: Path to qrels file

    Returns:
        Dictionary {query_id: {doc_id: relevance}}
    """
    logger.info(f"Loading qrels from: {qrels_file}")

    qrels = defaultdict(dict)

    with open(qrels_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                docid = parts[2]
                rel = int(parts[3])
                qrels[qid][docid] = rel

    qrels = dict(qrels)
    logger.info(f"Loaded qrels for {len(qrels)} queries")

    return qrels


def load_kb_candidates_from_file(
        kb_candidates_file: str
) -> Dict[str, List[Dict]]:
    """
    Load precomputed KB candidates from JSONL.

    Format per line: {"qid": "...", "candidates": [...]}

    Args:
        kb_candidates_file: Path to KB candidates JSONL

    Returns:
        Dictionary {query_id: [candidate_dicts]}
    """
    logger.info(f"Loading KB candidates from: {kb_candidates_file}")

    kb_candidates = {}

    with open(kb_candidates_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item['qid']
            candidates = item.get('candidates', [])
            kb_candidates[qid] = candidates

    logger.info(f"Loaded KB candidates for {len(kb_candidates)} queries")

    return kb_candidates


# ---------------------------------------------------------------------------
# TREC run file writer
# ---------------------------------------------------------------------------

def write_trec_run_file(
        run_results: Dict[str, List[Tuple[str, float]]],
        output_path: str,
        run_name: str = "MS-MEQE",
):
    """
    Write results in TREC run format.

    Format: qid Q0 docid rank score run_name

    Args:
        run_results: {query_id: [(doc_id, score), ...]}
        output_path: Output file path
        run_name: Run identifier
    """
    logger.info(f"Writing TREC run file to: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for qid in sorted(run_results.keys()):
            results = run_results[qid]
            for rank, (docid, score) in enumerate(results, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")

    logger.info(f"Wrote {len(run_results)} queries to run file")


# ---------------------------------------------------------------------------
# Dense retrieval system
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense retrieval using pre-computed document embeddings."""

    def __init__(self, index_path: str, encoder):
        self.encoder = encoder
        self.index_path = index_path

        # Load pre-computed document embeddings
        # Handle cases where index_path is file or directory
        base_path = Path(index_path)
        if base_path.is_file():
            base_path = base_path.parent

        doc_emb_path = base_path / "doc_embeddings.npy"
        doc_ids_path = base_path / "doc_ids.json"

        # Try parent if not found
        if not doc_emb_path.exists():
            doc_emb_path = base_path.parent / "doc_embeddings.npy"
            doc_ids_path = base_path.parent / "doc_ids.json"

        if not doc_emb_path.exists():
            raise FileNotFoundError(
                f"Document embeddings not found. "
                f"Run scripts/precompute_doc_embeddings.py first."
            )

        logger.info(f"Loading document embeddings from {doc_emb_path}")
        self.doc_embeddings = np.load(str(doc_emb_path))

        with open(doc_ids_path, 'r') as f:
            self.doc_ids = json.load(f)

        logger.info(
            f"Loaded {len(self.doc_ids)} document embeddings, "
            f"dim={self.doc_embeddings.shape[1]}"
        )

    def retrieve(
            self,
            query_embedding: np.ndarray,
            k: int = 1000,
    ) -> List[Tuple[str, float]]:
        """Retrieve documents using dense similarity."""
        # Normalize query
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

        # Compute cosine similarity
        similarities = self.doc_embeddings @ q_norm

        # Get top-k
        top_k = min(k, len(similarities))
        top_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Build results
        results = [
            (self.doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results


# ---------------------------------------------------------------------------
# MS-MEQE evaluation pipeline
# ---------------------------------------------------------------------------

class MSMEQEEvaluationPipeline:
    """Complete MS-MEQE evaluation pipeline."""

    def __init__(
            self,
            encoder,
            candidate_extractor,
            msmeqe_model,
            dense_retriever,
            evaluator,
            kb_candidates_map: Optional[Dict] = None,
            save_features_dir: Optional[str] = None,
    ):
        """
        Initialize evaluation pipeline.

        Args:
            encoder: SemanticEncoder instance
            candidate_extractor: MultiSourceCandidateExtractor instance
            msmeqe_model: MSMEQEExpansionModel instance
            dense_retriever: DenseRetriever instance
            evaluator: TRECEvaluator instance
            kb_candidates_map: Optional precomputed KB candidates
            save_features_dir: Optional directory for feature debugging
        """
        self.encoder = encoder
        self.candidate_extractor = candidate_extractor
        self.msmeqe_model = msmeqe_model
        self.dense_retriever = dense_retriever
        self.evaluator = evaluator
        self.kb_candidates_map = kb_candidates_map or {}
        self.save_features_dir = save_features_dir

        if self.save_features_dir:
            Path(self.save_features_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Initialized MSMEQEEvaluationPipeline")

    def run_evaluation(
            self,
            queries: Dict[str, str],
            qrels: Dict[str, Dict[str, int]],
            topk: int = 1000,
            log_per_query_path: Optional[str] = None,
    ) -> Dict:
        """
        Run complete MS-MEQE evaluation.

        Args:
            queries: Query ID -> query text
            qrels: Query ID -> doc ID -> relevance
            topk: Number of documents to retrieve
            log_per_query_path: Path for per-query logging (optional)

        Returns:
            Dictionary with results
        """
        logger.info(f"Running MS-MEQE evaluation on {len(queries)} queries")

        # Storage
        run_results = {}
        per_query_stats = []

        start_time = time.time()

        for qid, qtext in tqdm(queries.items(), desc="MS-MEQE evaluation"):
            try:
                # === 1. EXTRACT CANDIDATES (WITH KB OVERRIDE) ===

                # Get precomputed KB candidates if available
                kb_override = self.kb_candidates_map.get(qid) if self.kb_candidates_map else None

                if kb_override:
                    logger.debug(f"Using {len(kb_override)} precomputed KB candidates for {qid}")

                # Extract candidates with kb_override
                candidates = self.candidate_extractor.extract_all_candidates(
                    query_text=qtext,
                    query_id=qid,
                    kb_override=kb_override,
                )

                if not candidates:
                    logger.warning(f"No candidates for query {qid}, using original query")
                    q_emb = self.encoder.encode([qtext])[0]
                    results = self.dense_retriever.retrieve(q_emb, k=topk)
                    run_results[qid] = results
                    continue

                # Count candidates by source
                source_counts = defaultdict(int)
                for cand in candidates:
                    source_counts[cand.source] += 1

                # === 2. COMPUTE QUERY STATS ===
                query_stats = self.candidate_extractor.compute_query_stats(qtext)
                pseudo_centroid = self.candidate_extractor.compute_pseudo_centroid(qtext)

                # === 3. RUN MS-MEQE EXPANSION ===
                selected_terms, q_star = self.msmeqe_model.expand(
                    query_text=qtext,
                    candidates=candidates,
                    pseudo_doc_centroid=pseudo_centroid,
                    query_stats=query_stats,
                )

                # === 4. RETRIEVE WITH ENHANCED QUERY ===
                results = self.dense_retriever.retrieve(q_star, k=topk)
                run_results[qid] = results

                # === 5. LOG PER-QUERY STATS ===
                selected_source_counts = defaultdict(int)
                total_count = 0
                for term in selected_terms:
                    selected_source_counts[term.source] += term.count
                    total_count += term.count

                query_stat = {
                    'qid': qid,
                    'query_text': qtext,
                    'num_candidates': len(candidates),
                    'num_candidates_by_source': dict(source_counts),
                    'num_selected_terms': len(selected_terms),
                    'total_term_count': total_count,
                    'selected_by_source': dict(selected_source_counts),
                    'clarity': float(query_stats.get('clarity', 0)),
                    'entropy': float(query_stats.get('entropy', 0)),
                    'q_type': query_stats.get('q_type', 'unknown'),
                    'q_len': query_stats.get('q_len', 0),
                    'selected_terms': [
                        {
                            'term': t.term,
                            'source': t.source,
                            'value': float(t.value),
                            'weight': float(t.weight),
                            'count': int(t.count),
                        }
                        for t in selected_terms
                    ],
                }
                per_query_stats.append(query_stat)

            except Exception as e:
                logger.error(f"Failed to process query {qid}: {e}")
                # Fall back to original query
                q_emb = self.encoder.encode([qtext])[0]
                results = self.dense_retriever.retrieve(q_emb, k=topk)
                run_results[qid] = results
                continue

        elapsed = time.time() - start_time
        logger.info(f"MS-MEQE evaluation completed in {elapsed:.1f}s")

        # === EVALUATE ===
        metrics = self.evaluator.evaluate_run(run_results, qrels)

        logger.info("\nMS-MEQE Results:")
        logger.info(f"  MAP:         {metrics.get('map', 0):.4f}")
        logger.info(f"  nDCG@10:     {metrics.get('ndcg_cut_10', 0):.4f}")
        logger.info(f"  MRR:         {metrics.get('recip_rank', 0):.4f}")
        logger.info(f"  Recall@100:  {metrics.get('recall_100', 0):.4f}")
        logger.info(f"  Recall@1000: {metrics.get('recall_1000', 0):.4f}")
        logger.info(f"  P@10:        {metrics.get('P_10', 0):.4f}")

        # === SAVE PER-QUERY STATS ===
        if log_per_query_path:
            self._save_per_query_stats(per_query_stats, log_per_query_path)

        return {
            'metrics': metrics,
            'run_results': run_results,
            'per_query_stats': per_query_stats,
            'elapsed_time': elapsed,
        }

    def _save_per_query_stats(self, stats: List[Dict], output_path: str):
        """Save per-query statistics to JSONL."""
        logger.info(f"Saving per-query stats to: {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for stat in stats:
                f.write(json.dumps(stat) + '\n')

        logger.info(f"Saved stats for {len(stats)} queries")


# ---------------------------------------------------------------------------
# Helper Class for Ablations
# ---------------------------------------------------------------------------

class FixedBudgetModel:
    """Mock model for Fixed Budget ablation."""

    def predict(self, X):
        # Always return budget 50
        return np.array([50] * X.shape[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MS-MEQE end-to-end evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === CORE ===
    parser.add_argument("--index", type=str, required=True,
                        help="Lucene index path")
    parser.add_argument("--topics", type=str, required=True,
                        help="Query file (TSV/JSON/JSONL)")
    parser.add_argument("--qrels", type=str, required=True,
                        help="Qrels file (TREC format)")
    parser.add_argument("--field", type=str, default="contents",
                        help="Index field for text")
    parser.add_argument("--topk", type=int, default=1000,
                        help="Final ranking depth")
    parser.add_argument("--run-name", type=str, default="MS-MEQE",
                        help="Run identifier for TREC file")
    parser.add_argument("--lucene-path", type=str, required=True,
                        help="Path to Lucene JAR files")

    # === RM3 COMPONENT ===
    parser.add_argument("--rm3-depth", type=int, default=20,
                        help="Number of docs for pseudo-relevance feedback")
    parser.add_argument("--rm3-terms", type=int, default=40,
                        help="Number of terms from RM3")
    parser.add_argument("--rm3-alpha", type=float, default=0.5,
                        help="RM3 interpolation parameter")

    # === KB COMPONENT ===
    parser.add_argument("--kb-candidates", type=str, default=None,
                        help="Path to precomputed KB candidates JSONL (OVERRIDES dynamic extraction)")
    parser.add_argument("--kb-wat-output", type=str, default=None,
                        help="Path to WAT output (alternative to --kb-candidates)")

    # === EMBEDDING COMPONENT ===
    parser.add_argument("--sbert-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-BERT model name")
    parser.add_argument("--emb-vocab", type=str, default=None,
                        help="Path to pre-embedded vocabulary")

    # === MS-MEQE MODELS ===
    parser.add_argument("--value-model", type=str, required=True,
                        help="Path to trained value model (.pkl)")
    parser.add_argument("--weight-model", type=str, required=True,
                        help="Path to trained weight model (.pkl)")
    parser.add_argument("--budget-model", type=str, required=True,
                        help="Path to trained budget model (.pkl)")

    # === OUTPUT ===
    parser.add_argument("--output", type=str, required=True,
                        help="Output TREC run file path")

    # === ABLATION STUDY ===
    parser.add_argument("--ablation", type=str,
                        choices=['no_kb', 'no_emb', 'fixed_budget'],
                        help="Run ablation study configuration")

    # === MISC / ANALYSIS ===
    parser.add_argument("--save-features", type=str, default=None,
                        help="Directory to save feature matrices (currently a stub)")
    parser.add_argument("--log-per-query", type=str, default=None,
                        help="Path for per-query statistics JSONL")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of queries (for testing)")
    parser.add_argument("--collection-size", type=int, default=8841823,
                        help="Collection size (for MS-MEQE model)")

    # === LOGGING ===
    parser.add_argument("--log-level", type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    )

    logger.info("=" * 60)
    logger.info("MS-MEQE END-TO-END EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Index:       {args.index}")
    logger.info(f"Topics:      {args.topics}")
    logger.info(f"Qrels:       {args.qrels}")
    logger.info(f"Output:      {args.output}")
    logger.info(f"Run name:    {args.run_name}")

    if args.ablation:
        logger.info(f"ABLATION MODE: {args.ablation}")

    # Initialize Lucene
    if not initialize_lucene(args.lucene_path):
        logger.error("Failed to initialize Lucene")
        sys.exit(1)

    # Load data
    queries = load_queries_from_file(args.topics, args.max_queries)
    qrels = load_qrels_from_file(args.qrels)

    # Load KB candidates if provided
    kb_candidates_map = None
    if args.kb_candidates:
        kb_candidates_map = load_kb_candidates_from_file(args.kb_candidates)

    # Initialize components
    from src.reranking.semantic_encoder import SemanticEncoder
    from src.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor
    from src.expansion.msmeqe_expansion import MSMEQEExpansionModel
    from src.expansion.kb_expansion import KBCandidateExtractor
    from src.expansion.embedding_candidates import EmbeddingCandidateExtractor
    from src.retrieval.evaluator import TRECEvaluator

    logger.info(f"Loading encoder: {args.sbert_model}")
    encoder = SemanticEncoder(model_name=args.sbert_model)

    # Load trained models
    logger.info("Loading trained models...")
    try:
        value_model = joblib.load(args.value_model)
        weight_model = joblib.load(args.weight_model)

        # Ablation: Fixed Budget
        if args.ablation == 'fixed_budget':
            logger.info("ABLATION: Overriding Budget Model with Fixed Budget (50)")
            budget_model = FixedBudgetModel()
        else:
            budget_model = joblib.load(args.budget_model)

        # Validation
        if not (value_model and weight_model and budget_model):
            raise ValueError("One or more models failed to load properly.")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)

    # Initialize KB extractor
    kb_extractor = None
    if args.kb_wat_output:
        # Ablation: No KB
        if args.ablation == 'no_kb':
            logger.info("ABLATION: Disabling KB Extractor")
            kb_extractor = None
            # Also clear any precomputed map to be safe
            kb_candidates_map = None
        else:
            logger.info("Initializing KB extractor...")
            kb_extractor = KBCandidateExtractor(wat_output_path=args.kb_wat_output)

    # Initialize embedding extractor
    emb_extractor = None
    if args.emb_vocab:
        # Ablation: No Embeddings
        if args.ablation == 'no_emb':
            logger.info("ABLATION: Disabling Embedding Extractor")
            emb_extractor = None
        else:
            logger.info("Initializing embedding extractor...")
            emb_extractor = EmbeddingCandidateExtractor(
                encoder=encoder,
                vocab_path=args.emb_vocab,
            )

    # Initialize candidate extractor
    logger.info("Initializing candidate extractor...")
    candidate_extractor = MultiSourceCandidateExtractor(
        index_path=args.index,
        encoder=encoder,
        kb_extractor=kb_extractor,
        emb_extractor=emb_extractor,
        n_docs_rm3=args.rm3_terms,
        n_pseudo_docs=args.rm3_depth,
    )

    # Initialize MS-MEQE model
    # Note: feature_extractor is internal to MSMEQEExpansionModel
    logger.info("Initializing MS-MEQE model...")
    msmeqe_model = MSMEQEExpansionModel(
        encoder=encoder,
        value_model=value_model,
        weight_model=weight_model,
        budget_model=budget_model,
        collection_size=args.collection_size,
        lambda_interp=0.3,
        # Optional: expose these as CLI args if needed
        # min_budget=20,
        # max_budget=80,
        # budget_step=5,
    )

    # Initialize dense retriever
    logger.info("Initializing dense retriever...")
    dense_retriever = DenseRetriever(
        index_path=args.index,
        encoder=encoder,
    )

    # Initialize evaluator
    evaluator = TRECEvaluator(
        metrics=['map', 'ndcg_cut_10', 'recip_rank', 'recall_100', 'recall_1000', 'P_10']
    )

    # Create evaluation pipeline
    pipeline = MSMEQEEvaluationPipeline(
        encoder=encoder,
        candidate_extractor=candidate_extractor,
        msmeqe_model=msmeqe_model,
        dense_retriever=dense_retriever,
        evaluator=evaluator,
        kb_candidates_map=kb_candidates_map,
        save_features_dir=args.save_features,
    )

    # Run evaluation
    results = pipeline.run_evaluation(
        queries=queries,
        qrels=qrels,
        topk=args.topk,
        log_per_query_path=args.log_per_query,
    )

    # Write TREC run file
    write_trec_run_file(
        run_results=results['run_results'],
        output_path=args.output,
        run_name=args.run_name,
    )

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Run file: {args.output}")
    logger.info(f"Metrics:")
    for metric, value in results['metrics'].items():
        logger.info(f"  {metric:15s}: {value:.4f}")


if __name__ == "__main__":
    main()