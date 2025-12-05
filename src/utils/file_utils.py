import os
import json
import pickle
import gzip
import shutil
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2,
              compress: bool = False) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
        compress: Whether to gzip compress the file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if compress:
        filepath = filepath.with_suffix(filepath.suffix + '.gz')
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.info(f"Saved JSON to: {filepath}")


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    filepath = Path(filepath)

    if filepath.suffix == '.gz' or str(filepath).endswith('.json.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

    logger.debug(f"Loaded JSON from: {filepath}")
    return data


def save_pickle(data: Any, filepath: Union[str, Path], compress: bool = False) -> None:
    """
    Save data using pickle.

    Args:
        data: Data to save
        filepath: Output file path
        compress: Whether to gzip compress the file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if compress:
        filepath = filepath.with_suffix(filepath.suffix + '.gz')
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    logger.info(f"Saved pickle to: {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    filepath = Path(filepath)

    if filepath.suffix == '.gz' or str(filepath).endswith('.pkl.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

    logger.debug(f"Loaded pickle from: {filepath}")
    return data


def save_trec_run(run_results: Dict[str, List[tuple]], filepath: Union[str, Path],
                  run_name: str = "run") -> None:
    """
    Save results in TREC run format.

    Args:
        run_results: {query_id: [(doc_id, score), ...]}
        filepath: Output file path
        run_name: Run identifier
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w') as f:
        for query_id, docs in run_results.items():
            for rank, (doc_id, score) in enumerate(docs, 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

    logger.info(f"Saved TREC run to: {filepath}")


def load_trec_run(filepath: Union[str, Path]) -> Dict[str, List[tuple]]:
    """
    Load TREC run file.

    Args:
        filepath: Input file path

    Returns:
        {query_id: [(doc_id, score), ...]}
    """
    filepath = Path(filepath)
    run_results = {}

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id = parts[0]
                doc_id = parts[2]
                score = float(parts[4])

                if query_id not in run_results:
                    run_results[query_id] = []
                run_results[query_id].append((doc_id, score))

    logger.debug(f"Loaded TREC run from: {filepath} - {len(run_results)} queries")
    return run_results


def save_qrels(qrels: Dict[str, Dict[str, int]], filepath: Union[str, Path]) -> None:
    """
    Save qrels in TREC format.

    Args:
        qrels: {query_id: {doc_id: relevance}}
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w') as f:
        for query_id, docs in qrels.items():
            for doc_id, relevance in docs.items():
                f.write(f"{query_id} 0 {doc_id} {relevance}\n")

    logger.info(f"Saved qrels to: {filepath}")


def load_qrels(filepath: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """
    Load qrels file.

    Args:
        filepath: Input file path

    Returns:
        {query_id: {doc_id: relevance}}
    """
    filepath = Path(filepath)
    qrels = {}

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])

                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance

    logger.debug(f"Loaded qrels from: {filepath} - {len(qrels)} queries")
    return qrels


def save_training_data(training_data: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    """
    Save complete training dataset with proper organization.

    Args:
        training_data: Dictionary containing all training components
        output_dir: Output directory
    """
    output_dir = ensure_dir(output_dir)

    # Save different components
    if 'queries' in training_data:
        save_json(training_data['queries'], output_dir / 'queries.json')

    if 'qrels' in training_data:
        save_qrels(training_data['qrels'], output_dir / 'qrels.txt')
        save_json(training_data['qrels'], output_dir / 'qrels.json')  # Also save as JSON

    if 'documents' in training_data:
        save_pickle(training_data['documents'], output_dir / 'documents.pkl', compress=True)

    if 'features' in training_data:
        save_json(training_data['features'], output_dir / 'features.json', compress=True)

    if 'expansion_terms' in training_data:
        save_json(training_data['expansion_terms'], output_dir / 'expansion_terms.json')

    if 'statistics' in training_data:
        save_json(training_data['statistics'], output_dir / 'statistics.json')

    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'components': list(training_data.keys()),
        'num_queries': len(training_data.get('queries', {})),
        'num_documents': len(training_data.get('documents', {})),
        'num_qrels': sum(len(docs) for docs in training_data.get('qrels', {}).values())
    }
    save_json(metadata, output_dir / 'metadata.json')

    logger.info(f"Saved complete training dataset to: {output_dir}")


# In src/utils/file_utils.py

def load_training_data(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load complete training dataset. Checks for compressed versions of files
    like features and documents.

    Args:
        data_dir: Data directory

    Returns:
        Dictionary containing all training components
    """
    data_dir = Path(data_dir)
    training_data = {}

    # Define the base filenames
    components = {
        'queries': 'queries.json',
        'qrels': 'qrels.json',
        'documents': 'documents.pkl',
        'features': 'features.json',
        'statistics': 'statistics.json',
        'metadata': 'metadata.json'
    }

    for component, base_filename in components.items():
        filepath = data_dir / base_filename
        compressed_filepath = data_dir / (base_filename + '.gz')

        # Prioritize loading the compressed file if it exists
        if compressed_filepath.exists():
            final_path = compressed_filepath
        elif filepath.exists():
            final_path = filepath
        else:
            continue # Skip if neither file exists

        if str(final_path).endswith('.json') or str(final_path).endswith('.json.gz'):
            training_data[component] = load_json(final_path)
        elif str(final_path).endswith('.pkl') or str(final_path).endswith('.pkl.gz'):
            training_data[component] = load_pickle(final_path)

    logger.info(f"Loaded training dataset from: {data_dir}")
    return training_data


def save_experiment_results(results: Dict[str, Any], output_dir: Union[str, Path],
                            experiment_name: str) -> None:
    """
    Save experiment results with timestamp and organization.

    Args:
        results: Experiment results
        output_dir: Output directory
        experiment_name: Name of experiment
    """
    output_dir = ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment subdirectory
    exp_dir = ensure_dir(output_dir / f"{experiment_name}_{timestamp}")

    # Save main results
    save_json(results, exp_dir / 'results.json')

    # Save individual components if they exist
    if 'evaluations' in results:
        save_json(results['evaluations'], exp_dir / 'evaluations.json')

    if 'runs' in results:
        for run_name, run_data in results['runs'].items():
            save_trec_run(run_data, exp_dir / f'{run_name}.txt', run_name)

    if 'learned_weights' in results:
        save_json(results['learned_weights'], exp_dir / 'learned_weights.json')

    # Save summary
    summary = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'num_runs': len(results.get('runs', {})),
        'metrics': list(results.get('evaluations', {}).get(list(results.get('evaluations', {}).keys())[0],
                                                           {}).keys()) if results.get('evaluations') else []
    }
    save_json(summary, exp_dir / 'summary.json')

    logger.info(f"Saved experiment results to: {exp_dir}")
    return exp_dir


def backup_file(filepath: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create backup of a file with timestamp.

    Args:
        filepath: File to backup
        backup_dir: Backup directory (default: same directory as file)

    Returns:
        Path to backup file
    """
    filepath = Path(filepath)

    if backup_dir is None:
        backup_dir = filepath.parent / 'backups'
    else:
        backup_dir = Path(backup_dir)

    ensure_dir(backup_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"

    shutil.copy2(filepath, backup_path)
    logger.info(f"Created backup: {backup_path}")

    return backup_path


def get_file_size(filepath: Union[str, Path]) -> str:
    """
    Get human-readable file size.

    Args:
        filepath: File path

    Returns:
        Formatted file size string
    """
    filepath = Path(filepath)
    size_bytes = filepath.stat().st_size

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} PB"


def list_files_by_extension(directory: Union[str, Path], extension: str) -> List[Path]:
    """
    List all files with specific extension in directory.

    Args:
        directory: Directory to search
        extension: File extension (e.g., '.json', '.pkl')

    Returns:
        List of file paths
    """
    directory = Path(directory)
    files = list(directory.glob(f"*{extension}"))
    logger.debug(f"Found {len(files)} {extension} files in {directory}")
    return files


def clean_temp_files(temp_dir: Union[str, Path], max_age_hours: int = 24) -> None:
    """
    Clean old temporary files.

    Args:
        temp_dir: Temporary directory
        max_age_hours: Maximum age in hours before deletion
    """
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return

    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    deleted_count = 0
    for file in temp_dir.iterdir():
        if file.is_file():
            file_age = current_time - file.stat().st_mtime
            if file_age > max_age_seconds:
                file.unlink()
                deleted_count += 1

    logger.info(f"Cleaned {deleted_count} old temporary files from {temp_dir}")


# Context manager for temporary files
class TemporaryDirectory:
    """Context manager for temporary directory with automatic cleanup."""

    def __init__(self, prefix: str = "expansion_", cleanup: bool = True):
        self.prefix = prefix
        self.cleanup = cleanup
        self.temp_dir = None

    def __enter__(self) -> Path:
        self.temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix))
        logger.debug(f"Created temporary directory: {self.temp_dir}")
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup and self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")


# Utility functions for your specific use case
def load_msmarco_training_data(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load MSMARCO training data created by your training data script.

    Args:
        data_dir: Directory containing training data

    Returns:
        Training data dictionary
    """
    return load_training_data(data_dir)


def save_learned_weights(weights: tuple, filepath: Union[str, Path],
                         experiment_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Save learned weights with metadata.

    Args:
        weights: (alpha, beta, gamma) tuple
        filepath: Output file path
        experiment_info: Additional experiment information
    """
    alpha, beta, gamma = weights

    data = {
        'weights': {
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma)
        },
        'created_at': datetime.now().isoformat()
    }

    if experiment_info:
        data['experiment_info'] = experiment_info

    save_json(data, filepath)


def load_learned_weights(filepath: Union[str, Path]) -> tuple:
    """
    Load learned weights.

    Args:
        filepath: Input file path

    Returns:
        (alpha, beta, gamma) tuple
    """
    data = load_json(filepath)
    weights = data['weights']
    return (weights['alpha'], weights['beta'], weights['gamma'])


# Example usage
if __name__ == "__main__":
    # Example usage of the file utilities

    # Save some example data
    data = {
        'queries': {'q1': 'machine learning', 'q2': 'neural networks'},
        'results': {'q1': [('doc1', 0.9), ('doc2', 0.8)]}
    }

    with TemporaryDirectory() as temp_dir:
        # Save training data
        save_json(data['queries'], temp_dir / 'queries.json')
        save_trec_run(data['results'], temp_dir / 'run.txt')

        # Load it back
        loaded_queries = load_json(temp_dir / 'queries.json')
        loaded_run = load_trec_run(temp_dir / 'run.txt')

        print(f"Loaded {len(loaded_queries)} queries")
        print(f"Loaded run with {len(loaded_run)} queries")