import logging
import sys
import os
from typing import Optional
from pathlib import Path
from datetime import datetime


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def setup_logging(log_level: str = "INFO",
                  log_file: Optional[str] = None,
                  log_to_console: bool = True,
                  experiment_name: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the project.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_to_console: Whether to log to console
        experiment_name: Name of experiment for log file naming

    Returns:
        Configured logger
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file or experiment_name:
        if not log_file and experiment_name:
            # Generate log file name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/{experiment_name}_{timestamp}.log"

        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        print(f"Logging to file: {log_file}")

    # Log initial message
    logger = logging.getLogger("setup")
    logger.info(f"Logging configured - Level: {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_experiment_info(logger: logging.Logger, **kwargs):
    """
    Log experiment configuration and parameters.

    Args:
        logger: Logger instance
        **kwargs: Experiment parameters to log
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 50)

    for key, value in kwargs.items():
        logger.info(f"{key}: {value}")

    logger.info("=" * 50)


def log_results(logger: logging.Logger, results: dict, title: str = "RESULTS"):
    """
    Log experimental results in a formatted way.

    Args:
        logger: Logger instance
        results: Dictionary of results to log
        title: Title for the results section
    """
    logger.info("=" * 50)
    logger.info(title)
    logger.info("=" * 50)

    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        logger.info(f"  {sub_key}: {sub_value:.4f}")
                    else:
                        logger.info(f"  {sub_key}: {sub_value}")
            else:
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value}")
    else:
        logger.info(str(results))

    logger.info("=" * 50)


def log_model_info(logger: logging.Logger, model_name: str, **model_params):
    """
    Log model information and parameters.

    Args:
        logger: Logger instance
        model_name: Name of the model
        **model_params: Model parameters
    """
    logger.info(f"Model: {model_name}")
    for param, value in model_params.items():
        logger.info(f"  {param}: {value}")


def log_dataset_info(logger: logging.Logger, dataset_name: str, num_queries: int,
                     num_docs: int, num_qrels: int):
    """
    Log dataset information.

    Args:
        logger: Logger instance
        dataset_name: Name of dataset
        num_queries: Number of queries
        num_docs: Number of documents
        num_qrels: Number of relevance judgments
    """
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"  Queries: {num_queries:,}")
    logger.info(f"  Documents: {num_docs:,}")
    logger.info(f"  Qrels: {num_qrels:,}")


def log_weight_optimization(logger: logging.Logger, initial_weights: tuple,
                            final_weights: tuple, initial_score: float,
                            final_score: float, iterations: int):
    """
    Log weight optimization progress.

    Args:
        logger: Logger instance
        initial_weights: Starting weights (alpha, beta, gamma)
        final_weights: Final optimized weights
        initial_score: Initial performance score
        final_score: Final performance score
        iterations: Number of optimization iterations
    """
    logger.info("WEIGHT OPTIMIZATION COMPLETED")
    logger.info(
        f"Initial weights: alpha={initial_weights[0]:.3f}, beta={initial_weights[1]:.3f}, gamma={initial_weights[2]:.3f}")
    logger.info(
        f"Final weights:   alpha={final_weights[0]:.3f}, beta={final_weights[1]:.3f}, gamma={final_weights[2]:.3f}")
    logger.info(f"Initial score: {initial_score:.4f}")
    logger.info(f"Final score:   {final_score:.4f}")
    logger.info(
        f"Improvement:   +{final_score - initial_score:.4f} ({((final_score - initial_score) / initial_score * 100):+.2f}%)")
    logger.info(f"Iterations: {iterations}")


def log_expansion_stats(logger: logging.Logger, query_id: str, original_terms: list,
                        expansion_terms: list, importance_scores: dict):
    """
    Log query expansion statistics.

    Args:
        logger: Logger instance
        query_id: Query identifier
        original_terms: Original query terms
        expansion_terms: Expansion terms
        importance_scores: Importance scores for expansion terms
    """
    logger.debug(f"Query {query_id} expansion:")
    logger.debug(f"  Original terms: {original_terms}")
    logger.debug(f"  Expansion terms: {len(expansion_terms)}")

    if importance_scores:
        top_terms = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.debug(f"  Top expansion terms:")
        for term, score in top_terms:
            logger.debug(f"    {term}: {score:.3f}")


def log_evaluation_progress(logger: logging.Logger, current: int, total: int,
                            current_item: str = None):
    """
    Log evaluation progress.

    Args:
        logger: Logger instance
        current: Current item number
        total: Total number of items
        current_item: Name/ID of current item being processed
    """
    progress = (current / total) * 100
    if current_item:
        logger.info(f"Progress: {current}/{total} ({progress:.1f}%) - Processing: {current_item}")
    else:
        logger.info(f"Progress: {current}/{total} ({progress:.1f}%)")


# Example usage functions for your specific project
def setup_experiment_logging(experiment_name: str,
                             log_level: str = "INFO",
                             log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging specifically for expansion weight learning experiments.

    Args:
        experiment_name: Name of the experiment
        log_level: Logging level
        log_file: Optional path to log file. If None, generated automatically.

    Returns:
        Configured logger
    """
    # If a specific log file isn't provided, generate one automatically.
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = ensure_dir(f"logs/{experiment_name}") # Helper to create dir
        log_file = log_dir / f"{timestamp}.log"

    logger = setup_logging(
        log_level=log_level,
        log_file=str(log_file), # Pass the final path to the general setup function
        log_to_console=True
    )

    # Log system info
    import platform
    system_logger = get_logger("system")
    system_logger.info(f"Python version: {platform.python_version()}")
    system_logger.info(f"Platform: {platform.platform()}")

    return logger


def log_trec_dl_experiment(logger: logging.Logger, year: str, num_queries: int,
                           models: list, metrics: list):
    """
    Log TREC DL experiment setup.

    Args:
        logger: Logger instance
        year: TREC DL year (2019/2020)
        num_queries: Number of queries
        models: List of model names being evaluated
        metrics: List of evaluation metrics
    """
    log_experiment_info(
        logger,
        dataset=f"TREC DL {year}",
        queries=num_queries,
        models=models,
        metrics=metrics
    )


# Context manager for timed operations
class TimedOperation:
    """Context manager for timing operations with logging."""

    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.logger.info(f"Starting: {self.operation_name}")
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(f"Completed: {self.operation_name} (took {duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation_name} after {duration:.2f}s - {exc_val}")


# Example usage
if __name__ == "__main__":
    # Setup logging for an experiment
    logger = setup_experiment_logging("test_experiment", "DEBUG")

    # Log experiment info
    log_experiment_info(
        logger,
        model="ImportanceWeightedExpansion",
        alpha=1.2,
        beta=0.8,
        gamma=1.5,
        dataset="TREC DL 2019"
    )

    # Time an operation
    with TimedOperation(logger, "Model training"):
        import time

        time.sleep(2)  # Simulate work

    # Log results
    results = {
        "map": 0.4560,
        "ndcg_cut_10": 0.5420,
        "improvements": {
            "map": 0.031,
            "ndcg_cut_10": 0.024
        }
    }
    log_results(logger, results)