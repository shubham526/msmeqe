import logging
import tempfile
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import ir_datasets
from src.evaluation.metrics import get_metric

logger = logging.getLogger(__name__)


class TRECEvaluator:
    """
    Evaluator for TREC-style retrieval experiments.
    Handles qrels, runs, and computes standard IR metrics.
    """

    def __init__(self, metrics: List[str] = None):
        """
        Initialize evaluator with specified metrics.

        Args:
            metrics: List of metrics to compute (e.g., ['map', 'ndcg_cut_10', 'recip_rank'])
        """
        if metrics is None:
            self.metrics = ['map', 'ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_10', 'recall_100']
        else:
            self.metrics = metrics

        logger.info(f"TRECEvaluator initialized with metrics: {self.metrics}")

    def evaluate_run(self, run_results: Dict[str, List[Tuple[str, float]]],
                     qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """
        Evaluate a single run against qrels.

        Args:
            run_results: {query_id: [(doc_id, score), ...]}
            qrels: {query_id: {doc_id: relevance}}

        Returns:
            Dictionary of metric scores
        """
        if not run_results or not qrels:
            logger.warning("Empty run_results or qrels provided")
            return {metric: 0.0 for metric in self.metrics}

        # Create temporary files for evaluation
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.qrel') as qrel_file, \
                tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.run') as run_file:

            try:
                # Write qrels
                self._write_qrels(qrels, qrel_file.name)

                # Write run
                self._write_run(run_results, run_file.name)

                # Evaluate using your existing get_metric function
                results = {}
                for metric in self.metrics:
                    try:
                        results[metric] = float(get_metric(qrel_file.name, run_file.name, metric))
                    except Exception as e:
                        logger.warning(f"Failed to compute {metric}: {e}")
                        results[metric] = 0.0

                return results

            finally:
                # Cleanup temporary files
                try:
                    os.unlink(qrel_file.name)
                    os.unlink(run_file.name)
                except OSError:
                    pass

    def evaluate_multiple_runs(self, runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
                               qrels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple runs against qrels.

        Args:
            runs: {run_name: {query_id: [(doc_id, score), ...]}}
            qrels: {query_id: {doc_id: relevance}}

        Returns:
            {run_name: {metric: score}}
        """
        logger.info(f"Evaluating {len(runs)} runs on {len(qrels)} queries")

        results = {}
        for run_name, run_results in runs.items():
            logger.info(f"Evaluating run: {run_name}")
            results[run_name] = self.evaluate_run(run_results, qrels)

        return results

    def compare_runs(self, runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
                     qrels: Dict[str, Dict[str, int]],
                     baseline_run: str = None) -> Dict[str, Any]:
        """
        Compare multiple runs and compute improvements over baseline.

        Args:
            runs: {run_name: {query_id: [(doc_id, score), ...]}}
            qrels: {query_id: {doc_id: relevance}}
            baseline_run: Name of baseline run for computing improvements

        Returns:
            Comprehensive comparison results
        """
        # Evaluate all runs
        evaluations = self.evaluate_multiple_runs(runs, qrels)

        # Find baseline
        if baseline_run is None:
            baseline_run = list(runs.keys())[0]
            logger.info(f"Using '{baseline_run}' as baseline")

        if baseline_run not in evaluations:
            logger.error(f"Baseline run '{baseline_run}' not found in evaluations")
            baseline_run = list(evaluations.keys())[0]
            logger.info(f"Using '{baseline_run}' as baseline instead")

        baseline_scores = evaluations[baseline_run]

        # Compute improvements
        comparison = {
            'evaluations': evaluations,
            'baseline': baseline_run,
            'improvements': {}
        }

        for run_name, scores in evaluations.items():
            if run_name == baseline_run:
                continue

            improvements = {}
            for metric, score in scores.items():
                baseline_score = baseline_scores[metric]
                if baseline_score > 0:
                    improvement_pct = (score - baseline_score) / baseline_score * 100
                    improvements[f"{metric}_improvement_pct"] = improvement_pct
                    improvements[f"{metric}_improvement_abs"] = score - baseline_score
                else:
                    improvements[f"{metric}_improvement_pct"] = 0.0
                    improvements[f"{metric}_improvement_abs"] = score

            comparison['improvements'][run_name] = improvements

        return comparison

    def create_results_table(self, comparison_results: Dict[str, Any]) -> str:
        """
        Create a formatted results table for paper inclusion.
        """
        evaluations = comparison_results['evaluations']
        baseline = comparison_results['baseline']
        improvements = comparison_results['improvements']

        # Table header
        table = "Method" + "\t" + "\t".join(self.metrics) + "\n"
        table += "-" * (len("Method") + sum(len(m) + 1 for m in self.metrics)) + "\n"

        # Baseline row
        baseline_scores = evaluations[baseline]
        table += f"{baseline}"
        for metric in self.metrics:
            table += f"\t{baseline_scores[metric]:.4f}"
        table += "\n"

        # Other runs with improvements
        for run_name, scores in evaluations.items():
            if run_name == baseline:
                continue

            table += f"{run_name}"
            for metric in self.metrics:
                score = scores[metric]
                if run_name in improvements:
                    improvement = improvements[run_name].get(f"{metric}_improvement_abs", 0.0)
                    if improvement > 0:
                        table += f"\t{score:.4f} (+{improvement:.4f})"
                    else:
                        table += f"\t{score:.4f} ({improvement:.4f})"
                else:
                    table += f"\t{score:.4f}"
            table += "\n"

        return table

    def _write_qrels(self, qrels: Dict[str, Dict[str, int]], filename: str):
        """Write qrels to TREC format file."""
        with open(filename, 'w') as f:
            for query_id, docs in qrels.items():
                for doc_id, relevance in docs.items():
                    f.write(f"{query_id} 0 {doc_id} {relevance}\n")

    def _write_run(self, run_results: Dict[str, List[Tuple[str, float]]], filename: str):
        """Write run results to TREC format file."""
        with open(filename, 'w') as f:
            for query_id, docs in run_results.items():
                for rank, (doc_id, score) in enumerate(docs, 1):
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} run\n")


class ExpansionEvaluator(TRECEvaluator):
    """
    Specialized evaluator for query expansion experiments.
    """

    def __init__(self, metrics: List[str] = None):
        super().__init__(metrics)

    def evaluate_expansion_with_precomputed_data(self,
                                                 baseline_run: Dict[str, List[Tuple[str, float]]],
                                                 expansion_models: Dict[str, Any],
                                                 queries: Dict[str, str],
                                                 qrels: Dict[str, Dict[str, int]],
                                                 first_stage_runs: Dict[str, List[Tuple[str, float]]],
                                                 expansion_terms_dict: Dict[str, List[Tuple[str, float]]],
                                                 pseudo_relevant_docs_dict: Dict[str, List[str]],
                                                 pseudo_relevant_scores_dict: Dict[str, List[float]],
                                                 reranker) -> Dict[str, Any]:
        """
        Evaluate expansion models with pre-computed expansion data.

        Args:
            baseline_run: Results without expansion
            expansion_models: {model_name: expansion_model}
            queries: {query_id: query_text}
            qrels: {query_id: {doc_id: relevance}}
            first_stage_runs: {query_id: [(doc_id, score), ...]}
            expansion_terms_dict: {query_id: [(term, rm_weight), ...]}
            pseudo_relevant_docs_dict: {query_id: [doc_text, ...]}
            pseudo_relevant_scores_dict: {query_id: [score, ...]}
            reranker: MultiVectorReranker instance

        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Running expansion evaluation with {len(expansion_models)} models")

        # Prepare runs dictionary
        runs = {'Baseline (No Expansion)': baseline_run}

        # Evaluate each expansion model
        for model_name, expansion_model in expansion_models.items():
            logger.info(f"Evaluating expansion model: {model_name}")

            # Compute importance weights for all queries
            importance_weights_dict = {}
            for query_id, query_text in queries.items():
                try:
                    if (query_id in pseudo_relevant_docs_dict and
                            query_id in pseudo_relevant_scores_dict):

                        importance_weights = expansion_model.expand_query(
                            query=query_text,
                            pseudo_relevant_docs=pseudo_relevant_docs_dict[query_id],
                            pseudo_relevant_scores=pseudo_relevant_scores_dict[query_id],
                            reference_doc_id=None  # For BM25 scoring - could be improved
                        )
                        importance_weights_dict[query_id] = importance_weights
                    else:
                        logger.warning(f"Missing expansion data for query {query_id}")
                        importance_weights_dict[query_id] = {}

                except Exception as e:
                    logger.warning(f"Failed to compute importance for query {query_id}: {e}")
                    importance_weights_dict[query_id] = {}

            # Rerank using this expansion model
            try:
                reranked_results = reranker.rerank_trec_dl_run(
                    queries=queries,
                    first_stage_runs=first_stage_runs,
                    expansion_terms_dict=expansion_terms_dict,
                    importance_weights_dict=importance_weights_dict,
                    top_k=100
                )
                runs[model_name] = reranked_results

            except Exception as e:
                logger.error(f"Failed to rerank with model {model_name}: {e}")
                runs[model_name] = baseline_run  # Fallback to baseline

        # Compare all runs
        comparison = self.compare_runs(runs, qrels, 'Baseline (No Expansion)')

        return comparison

    def evaluate_expansion_ablation(self,
                                    baseline_run: Dict[str, List[Tuple[str, float]]],
                                    expansion_models: Dict[str, Any],
                                    queries: Dict[str, str],
                                    qrels: Dict[str, Dict[str, int]],
                                    first_stage_runs: Dict[str, List[Tuple[str, float]]],
                                    reranker,
                                    rm_expansion=None,
                                    document_collection: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Evaluate ablation study for different expansion models.
        This version generates expansion data on-the-fly.

        Args:
            baseline_run: Results without expansion
            expansion_models: {model_name: expansion_model}
            queries: {query_id: query_text}
            qrels: {query_id: {doc_id: relevance}}
            first_stage_runs: {query_id: [(doc_id, score), ...]}
            reranker: MultiVectorReranker instance
            rm_expansion: RM expansion instance (optional)
            document_collection: {doc_id: doc_text} (optional)

        Returns:
            Comprehensive ablation results
        """
        logger.info(f"Running expansion ablation with {len(expansion_models)} models")

        if rm_expansion is None or document_collection is None:
            logger.warning("Missing RM expansion or document collection - using simplified expansion")
            return self._evaluate_with_simple_expansion(
                baseline_run, expansion_models, queries, qrels, first_stage_runs, reranker
            )

        # Generate expansion data for all queries
        expansion_terms_dict = {}
        pseudo_relevant_docs_dict = {}
        pseudo_relevant_scores_dict = {}

        for query_id, query_text in queries.items():
            try:
                if query_id in first_stage_runs:
                    # Use top-k documents as pseudo-relevant
                    top_docs = first_stage_runs[query_id][:10]  # Top 10 for PRF

                    pseudo_docs = []
                    pseudo_scores = []

                    for doc_id, score in top_docs:
                        if doc_id in document_collection:
                            pseudo_docs.append(document_collection[doc_id])
                            pseudo_scores.append(score)

                    if pseudo_docs:
                        # Generate RM expansion
                        rm_terms = rm_expansion.expand_query(
                            query=query_text,
                            documents=pseudo_docs,
                            scores=pseudo_scores,
                            num_expansion_terms=15,
                            rm_type="rm3"
                        )

                        expansion_terms_dict[query_id] = rm_terms
                        pseudo_relevant_docs_dict[query_id] = pseudo_docs
                        pseudo_relevant_scores_dict[query_id] = pseudo_scores
                    else:
                        logger.warning(f"No valid pseudo-relevant docs for query {query_id}")
                        expansion_terms_dict[query_id] = []
                        pseudo_relevant_docs_dict[query_id] = []
                        pseudo_relevant_scores_dict[query_id] = []

            except Exception as e:
                logger.warning(f"Failed to generate expansion for query {query_id}: {e}")
                expansion_terms_dict[query_id] = []
                pseudo_relevant_docs_dict[query_id] = []
                pseudo_relevant_scores_dict[query_id] = []

        # Use the main evaluation function with pre-computed data
        return self.evaluate_expansion_with_precomputed_data(
            baseline_run=baseline_run,
            expansion_models=expansion_models,
            queries=queries,
            qrels=qrels,
            first_stage_runs=first_stage_runs,
            expansion_terms_dict=expansion_terms_dict,
            pseudo_relevant_docs_dict=pseudo_relevant_docs_dict,
            pseudo_relevant_scores_dict=pseudo_relevant_scores_dict,
            reranker=reranker
        )

    def _evaluate_with_simple_expansion(self,
                                        baseline_run, expansion_models, queries,
                                        qrels, first_stage_runs, reranker):
        """Fallback evaluation with simple query term expansion."""
        logger.info("Using simple expansion fallback")

        runs = {'Baseline (No Expansion)': baseline_run}

        # Simple expansion: use query terms
        expansion_terms_dict = {}
        for query_id, query_text in queries.items():
            terms = query_text.lower().split()[:5]  # First 5 terms
            expansion_terms_dict[query_id] = [(term, 1.0) for term in terms]

        # Evaluate each model with uniform importance weights
        for model_name, expansion_model in expansion_models.items():
            logger.info(f"Evaluating model {model_name} with simple expansion")

            # Create uniform importance weights
            importance_weights_dict = {}
            for query_id in queries.keys():
                importance_weights_dict[query_id] = {
                    term: 1.0 for term, _ in expansion_terms_dict.get(query_id, [])
                }

            try:
                reranked_results = reranker.rerank_trec_dl_run(
                    queries=queries,
                    first_stage_runs=first_stage_runs,
                    expansion_terms_dict=expansion_terms_dict,
                    importance_weights_dict=importance_weights_dict,
                    top_k=100
                )
                runs[model_name] = reranked_results

            except Exception as e:
                logger.error(f"Failed to rerank with model {model_name}: {e}")
                runs[model_name] = baseline_run

        return self.compare_runs(runs, qrels, 'Baseline (No Expansion)')


def create_trec_dl_evaluator(year: str = "2019") -> ExpansionEvaluator:
    """
    Factory function to create evaluator for TREC DL datasets.

    Args:
        year: "2019" or "2020"

    Returns:
        Configured ExpansionEvaluator
    """
    # TREC DL specific metrics
    trec_dl_metrics = ['map', 'ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_100']

    evaluator = ExpansionEvaluator(metrics=trec_dl_metrics)

    logger.info(f"Created TREC DL {year} evaluator")
    return evaluator


# Example usage for your paper
def run_paper_evaluation_example():
    """
    Example evaluation pipeline for your SIGIR paper.
    Shows how to integrate all components.
    """
    logger.info("Starting paper evaluation example")

    try:
        # This is just an example - adapt to your actual pipeline
        from src.models.multivector_reranking import TRECDLReranker
        from src.models.expansion_models import create_baseline_comparison_models

        # Load TREC DL data
        evaluator = create_trec_dl_evaluator("2019")

        # Load data using ir_datasets
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2019")
        queries = {q.query_id: q.text for q in dataset.queries_iter()}

        qrels = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        # Load first-stage runs
        first_stage_runs = defaultdict(list)
        for scoreddoc in dataset.scoreddocs_iter():
            first_stage_runs[scoreddoc.query_id].append((scoreddoc.doc_id, scoreddoc.score))

        print(f"Loaded {len(queries)} queries, {len(qrels)} qrels")
        print("Example evaluation pipeline ready!")

        return {
            'queries': queries,
            'qrels': dict(qrels),
            'first_stage_runs': dict(first_stage_runs),
            'evaluator': evaluator
        }

    except ImportError as e:
        logger.warning(f"Could not import all modules for example: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("TREC Evaluator Module")
    print("=" * 30)

    # Run example
    result = run_paper_evaluation_example()
    if result:
        print("✓ Example evaluation pipeline created successfully")
    else:
        print("⚠ Could not create full example due to missing dependencies")