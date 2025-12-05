#!/usr/bin/env python3
"""
Create BM25 index with BERT tokenization.

This script creates a BM25 index that aligns with BERT tokenization for proper
importance score computation. Uses BERTTokenBM25Indexer to build the index
from a document collection.

Usage:
    python create_index.py --collection msmarco-passage --model_name bert-base-uncased --output_dir ./indexes --lucene_path /path/to/lucene/*
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
from collections import defaultdict
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import ir_datasets
from tqdm import tqdm

# Import existing indexing infrastructure
try:
    from src.core.bm25_scorer import BERTTokenBM25Indexer
    from src.utils.lucene_utils import initialize_lucene

    INDEXER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"BM25 indexer not available: {e}")
    INDEXER_AVAILABLE = False

from src.utils.file_utils import ensure_dir
from src.utils.logging_utils import (setup_experiment_logging, log_experiment_info,
                                 TimedOperation, log_dataset_info)

logger = logging.getLogger(__name__)


class IndexCreator:
    """
    Creates BM25 indexes with BERT tokenization alignment.
    """

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 k1: float = 1.2,
                 b: float = 0.75,
                 max_length: int = 512):
        """
        Initialize index creator.

        Args:
            model_name: HuggingFace model name for tokenization
            k1: BM25 k1 parameter
            b: BM25 b parameter
            max_length: Maximum token length for documents
        """
        self.model_name = model_name
        self.k1 = k1
        self.b = b
        self.max_length = max_length

        logger.info(f"IndexCreator initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  BM25 parameters: k1={k1}, b={b}")
        logger.info(f"  Max length: {max_length}")

    def load_document_collection_generator(self, collection_name: str) -> Iterator[Tuple[str, str]]:
        """
        Creates a generator that yields documents one by one.
        """
        logger.info(f"Setting up document generator for collection: {collection_name}")
        dataset = ir_datasets.load(collection_name)

        for doc in dataset.docs_iter():
            if hasattr(doc, 'text'):
                doc_content = doc.text
            elif hasattr(doc, 'body'):
                doc_content = doc.body
            else:
                continue
            yield (doc.doc_id, doc_content)

    def create_index(self, collection_name: str, output_dir: str,
                     max_docs: Optional[int] = None) -> str:
        """
        Create BM25 index for document collection.

        Args:
            collection_name: IR dataset collection name
            output_dir: Output directory for index
            max_docs: Maximum documents to index (for testing)

        Returns:
            Path to created index
        """
        if not INDEXER_AVAILABLE:
            raise RuntimeError("BERTTokenBM25Indexer not available. Check bert_bm25_indexer.py import.")

        # Ensure output directory exists
        output_dir = ensure_dir(output_dir)
        index_path = output_dir / f"{collection_name.replace('/', '_')}_{self.model_name.replace('/', '_')}"

        logger.info(f"Creating index at: {index_path}")

        # Initialize indexer
        logger.info("Initializing BERTTokenBM25Indexer...")
        indexer = BERTTokenBM25Indexer(
            index_path=str(index_path),
            model_name=self.model_name
        )

        with TimedOperation(logger, "Document indexing"):
            # Get the total number of documents for the progress bar, if possible
            try:
                num_docs = ir_datasets.load(collection_name).docs_count()
            except Exception:
                num_docs = None  # Fallback if count is not available

            # Create the generator
            doc_generator = self.load_document_collection_generator(collection_name)

            # If max_docs is set, create a limited generator
            if max_docs is not None:
                from itertools import islice
                doc_generator = islice(doc_generator, max_docs)
                num_docs = max_docs

            # Call the indexer's create_index method with the generator
            indexer.create_index(doc_generator, num_docs=num_docs)

        logger.info(f"Index creation completed.")
        return str(index_path)


    def validate_index(self, index_path: str, sample_queries: List[str] = None) -> bool:
        """
        Validate the created index by running sample queries.

        Args:
            index_path: Path to the index
            sample_queries: List of test queries

        Returns:
            True if validation successful
        """
        logger.info("Validating created index...")

        if sample_queries is None:
            sample_queries = [
                "machine learning",
                "neural networks",
                "information retrieval",
                "natural language processing",
                "deep learning algorithms"
            ]

        try:
            # Import scorer to test the index
            from src.core.bm25_scorer import TokenBM25Scorer

            # Initialize scorer with the new index
            scorer = TokenBM25Scorer(index_path)

            # Test with sample queries
            validation_passed = True
            for query in sample_queries:
                try:
                    # This should not fail if index is properly created
                    token_scores = scorer.get_token_scores(query, "dummy_doc_id", max_length=50)
                    if token_scores is not None:
                        logger.debug(f"Validation query '{query}': OK")
                    else:
                        logger.warning(f"Validation query '{query}': No scores returned")
                        validation_passed = False
                except Exception as e:
                    logger.error(f"Validation query '{query}' failed: {e}")
                    validation_passed = False

            if validation_passed:
                logger.info("Index validation: PASSED")
            else:
                logger.warning("Index validation: FAILED (some queries failed)")

            return validation_passed

        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False

    def get_index_statistics(self, index_path: str) -> Dict[str, Any]:
        """
        Get statistics about the created index.

        Args:
            index_path: Path to the index

        Returns:
            Dictionary with index statistics
        """
        try:
            from src.core.bm25_scorer import  TokenBM25Scorer

            scorer = TokenBM25Scorer(index_path)

            # Basic statistics (these would need to be implemented in your scorer)
            stats = {
                'index_path': index_path,
                'model_name': self.model_name,
                'bm25_parameters': {'k1': self.k1, 'b': self.b},
                'max_length': self.max_length,
                'index_size_mb': self._get_directory_size(index_path) / (1024 * 1024)
            }

            # You might want to add more statistics here based on your indexer
            # For example: number of documents, average document length, vocabulary size, etc.

            return stats

        except Exception as e:
            logger.error(f"Failed to get index statistics: {e}")
            return {'error': str(e)}

    def _get_directory_size(self, path: str) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size


def main():
    parser = argparse.ArgumentParser(description="Create BM25 index with BERT tokenization")

    # Required parameters
    parser.add_argument('--collection', type=str, required=True,
                        help='IR dataset collection name (e.g., msmarco-passage)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for the index')
    parser.add_argument('--lucene-path', type=str, required=True,
                        help='Path to Lucene JAR files')

    # Model parameters
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                        help='HuggingFace model name for tokenization (default: bert-base-uncased)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum token length for documents (default: 512)')

    # BM25 parameters
    parser.add_argument('--k1', type=float, default=1.2,
                        help='BM25 k1 parameter (default: 1.2)')
    parser.add_argument('--b', type=float, default=0.75,
                        help='BM25 b parameter (default: 0.75)')

    # Indexing parameters
    parser.add_argument('--max-docs', type=int, default=None,
                        help='Maximum number of documents to index (for testing)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate index after creation')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing index')

    # Logging parameters
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup logging
    logger = setup_experiment_logging("create_index", args.log_level)

    # Log experiment configuration
    log_experiment_info(
        logger,
        collection=args.collection,
        model_name=args.model_name,
        output_dir=args.output_dir,
        k1=args.k1,
        b=args.b,
        max_length=args.max_length,
        max_docs=args.max_docs,
        validate=args.validate,
        force=args.force
    )

    try:
        # Check if indexer is available
        if not INDEXER_AVAILABLE:
            logger.error("BERTTokenBM25Indexer not available. Check bert_bm25_indexer.py")
            sys.exit(1)

        # Initialize Lucene JVM
        logger.info(f"Initializing Lucene from: {args.lucene_path}")
        if not initialize_lucene(args.lucene_path):
            logger.error("Failed to initialize Lucene")
            sys.exit(1)
        logger.info("Lucene initialized successfully")

        # Check if index already exists
        output_dir = Path(args.output_dir)
        index_name = f"{args.collection.replace('/', '_')}_{args.model_name.replace('/', '_')}"
        index_path = output_dir / index_name

        if index_path.exists() and not args.force:
            logger.error(f"Index already exists at {index_path}. Use --force to overwrite.")
            sys.exit(1)
        elif index_path.exists() and args.force:
            logger.warning(f"Overwriting existing index at {index_path}")
            import shutil
            shutil.rmtree(index_path)

        # Create index creator
        creator = IndexCreator(
            model_name=args.model_name,
            k1=args.k1,
            b=args.b,
            max_length=args.max_length
        )

        # Log dataset info
        try:
            dataset = ir_datasets.load(args.collection)
            # Count documents for logging (this might be slow for large collections)
            if args.max_docs and args.max_docs < 100000:
                doc_count = sum(1 for _ in dataset.docs_iter())
                log_dataset_info(logger, args.collection, 0, doc_count, 0)
        except Exception as e:
            logger.warning(f"Could not get dataset info: {e}")

        # Create index
        final_index_path = creator.create_index(
            collection_name=args.collection,
            output_dir=args.output_dir,
            max_docs=args.max_docs
        )

        # Validate index if requested
        if args.validate:
            validation_success = creator.validate_index(final_index_path)
            if not validation_success:
                logger.warning("Index validation failed, but index was created")

        # Get and log index statistics
        stats = creator.get_index_statistics(final_index_path)
        logger.info("Index Statistics:")
        for key, value in stats.items():
            if key == 'index_size_mb':
                logger.info(f"  {key}: {value:.2f} MB")
            else:
                logger.info(f"  {key}: {value}")

        # Final success message
        logger.info("=" * 60)
        logger.info("INDEX CREATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Collection: {args.collection}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Index path: {final_index_path}")
        logger.info(f"Index size: {stats.get('index_size_mb', 0):.2f} MB")

        # Usage instructions
        logger.info("")
        logger.info("Usage in other scripts:")
        logger.info(f"  --index_path {final_index_path}")
        logger.info(f"  --lucene_path {args.lucene_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()