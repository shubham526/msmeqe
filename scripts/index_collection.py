#!/usr/bin/env python3
# scripts/index_collection.py

"""
Build Lucene index from document collection for MS-MEQE.

Creates a BM25 index with:
  - Document text storage (for RM3 pseudo-relevance feedback)
  - Term statistics (DF, CF)
  - Document lengths
  - Collection statistics

The index is used for:
  1. BM25 retrieval (first-stage ranking)
  2. RM3 query expansion (pseudo-doc analysis)
  3. Term statistics for feature extraction
  4. Pseudo-document centroid computation

Usage:
    # Index MS MARCO passage collection
    python scripts/index_collection.py \\
        --collection msmarco-passage \\
        --output data/msmarco_index \\
        --lucene-path /path/to/lucene-9.x.x/* \\
        --analyzer english \\
        --ram-buffer 4096

    # Index from JSONL file
    python scripts/index_collection.py \\
        --input data/docs.jsonl \\
        --output data/my_index \\
        --lucene-path /path/to/lucene/* \\
        --format jsonl \\
        --analyzer english

    # Index with validation
    python scripts/index_collection.py \\
        --collection msmarco-passage \\
        --output data/msmarco_index \\
        --lucene-path /path/to/lucene/* \\
        --validate \\
        --max-docs 10000
"""

import logging
import argparse
import json
import sys
from pathlib import Path as PathlibPath  # Avoid name collision!
from typing import Iterator, Dict, Optional, Tuple
from tqdm import tqdm
import time
from src.utils.lucene_utils import initialize_lucene, get_lucene_classes

logger = logging.getLogger(__name__)




# ---------------------------------------------------------------------------
# Document iterators
# ---------------------------------------------------------------------------

class DocumentIterator:
    """Abstract document iterator interface."""

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError


class IRDatasetIterator(DocumentIterator):
    """Iterate over documents from ir-datasets."""

    def __init__(self, collection_name: str):
        try:
            import ir_datasets
        except ImportError:
            raise ImportError(
                "ir_datasets required. Install with: pip install ir-datasets"
            )

        self.dataset = ir_datasets.load(collection_name)
        self.collection_name = collection_name

        logger.info(f"Loaded ir-datasets collection: {collection_name}")

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        for doc in self.dataset.docs_iter():
            # Handle different document types
            if hasattr(doc, 'doc_id'):
                docid = doc.doc_id
            elif hasattr(doc, 'id'):
                docid = doc.id
            else:
                docid = str(doc[0])

            if hasattr(doc, 'text'):
                text = doc.text
            elif hasattr(doc, 'body'):
                text = doc.body
            else:
                text = str(doc[1])

            yield docid, text


class JSONLIterator(DocumentIterator):
    """Iterate over documents from JSONL file."""

    def __init__(
            self,
            file_path: str,
            docid_field: str = 'docid',
            text_field: str = 'text',
    ):
        self.file_path = file_path
        self.docid_field = docid_field
        self.text_field = text_field

        logger.info(f"Loading JSONL file: {file_path}")

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    doc = json.loads(line)
                    docid = str(doc[self.docid_field])
                    text = str(doc[self.text_field])
                    yield docid, text
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Line {line_no}: Failed to parse: {e}")
                    continue


class TSVIterator(DocumentIterator):
    """Iterate over documents from TSV file."""

    def __init__(
            self,
            file_path: str,
            docid_col: int = 0,
            text_col: int = 1,
            delimiter: str = '\t',
    ):
        self.file_path = file_path
        self.docid_col = docid_col
        self.text_col = text_col
        self.delimiter = delimiter

        logger.info(f"Loading TSV file: {file_path}")

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    fields = line.rstrip('\n').split(self.delimiter)
                    docid = fields[self.docid_col]
                    text = fields[self.text_col]
                    yield docid, text
                except IndexError as e:
                    logger.warning(f"Line {line_no}: Failed to parse: {e}")
                    continue


# ---------------------------------------------------------------------------
# Lucene indexer
# ---------------------------------------------------------------------------

class LuceneIndexer:
    """
    Build Lucene index with BM25 similarity.

    Index structure:
      - Field 'docid': stored, not indexed (for retrieval)
      - Field 'contents': stored, indexed, analyzed (for BM25 + RM3)

    Stores:
      - Document text (for RM3 pseudo-doc analysis)
      - Term vectors (for term statistics)
      - BM25 similarity parameters
    """

    def __init__(
            self,
            index_dir: str,
            analyzer: str = 'english',
            ram_buffer_size_mb: int = 4096,
            k1: float = 1.2,
            b: float = 0.75,
    ):
        """
        Initialize Lucene indexer.

        Args:
            index_dir: Output directory for index
            analyzer: Analyzer type ('standard', 'english', 'whitespace')
            ram_buffer_size_mb: RAM buffer size for indexing (larger = faster)
            k1: BM25 k1 parameter (term saturation)
            b: BM25 b parameter (length normalization)
        """
        self.index_dir = index_dir
        self.analyzer_name = analyzer.lower()
        self.ram_buffer_size_mb = ram_buffer_size_mb
        self.k1 = k1
        self.b = b

        # Get Lucene classes
        classes = get_lucene_classes()

        self.IndexWriter = classes['IndexWriter']
        self.IndexWriterConfig = classes['IndexWriterConfig']
        self.FSDirectory = classes['FSDirectory']
        self.JavaPaths = classes['JavaPaths']
        self.Document = classes['Document']
        self.Field = classes['Field']
        self.FieldType = classes['FieldType']
        self.StringField = classes['StringField']
        self.TextField = classes['TextField']
        self.StandardAnalyzer = classes['StandardAnalyzer']
        self.EnglishAnalyzer = classes['EnglishAnalyzer']
        self.WhitespaceAnalyzer = classes['WhitespaceAnalyzer']
        self.BM25Similarity = classes['BM25Similarity']
        self.IndexOptions = classes['IndexOptions']

        # Create analyzer
        self.analyzer = self._create_analyzer()

        # Create directory
        PathlibPath(index_dir).mkdir(parents=True, exist_ok=True)
        directory = self.FSDirectory.open(self.JavaPaths.get(index_dir))

        # Create config
        config = self.IndexWriterConfig(self.analyzer)
        config.setRAMBufferSizeMB(float(ram_buffer_size_mb))

        # Set BM25 similarity with custom parameters
        similarity = self.BM25Similarity(float(k1), float(b))
        config.setSimilarity(similarity)

        # Create writer
        self.writer = self.IndexWriter(directory, config)

        logger.info(f"Lucene indexer initialized:")
        logger.info(f"  Index dir:    {index_dir}")
        logger.info(f"  Analyzer:     {analyzer}")
        logger.info(f"  RAM buffer:   {ram_buffer_size_mb} MB")
        logger.info(f"  BM25:         k1={k1}, b={b}")

    def _create_analyzer(self):
        """Create analyzer based on name."""
        if self.analyzer_name == 'standard':
            return self.StandardAnalyzer()
        elif self.analyzer_name == 'english':
            return self.EnglishAnalyzer()
        elif self.analyzer_name == 'whitespace':
            return self.WhitespaceAnalyzer()
        else:
            raise ValueError(f"Unknown analyzer: {self.analyzer_name}")

    def index_documents(
            self,
            doc_iterator: DocumentIterator,
            field_name: str = 'contents',
            commit_every: int = 10000,
            max_docs: Optional[int] = None,
    ) -> int:
        """
        Index documents from iterator.

        Args:
            doc_iterator: DocumentIterator instance
            field_name: Field name for document text
            commit_every: Commit frequency (number of documents)
            max_docs: Maximum documents to index (for testing)

        Returns:
            Number of documents indexed
        """
        logger.info("Starting indexing...")

        count = 0
        start_time = time.time()

        # Create custom FieldType for contents field
        # We need: stored=True (for RM3), indexed=True, term vectors
        contents_field_type = self.FieldType()
        contents_field_type.setStored(True)  # CRITICAL: Store text for RM3
        contents_field_type.setIndexOptions(self.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        contents_field_type.setStoreTermVectors(True)
        contents_field_type.setStoreTermVectorPositions(True)
        contents_field_type.setTokenized(True)

        for docid, text in tqdm(doc_iterator, desc="Indexing documents"):
            if max_docs and count >= max_docs:
                logger.info(f"Reached max_docs limit: {max_docs}")
                break

            try:
                # Create document
                doc = self.Document()

                # Add docid field (stored, not indexed)
                doc.add(self.StringField("docid", docid, self.Field.Store.YES))

                # Add text field with custom FieldType
                contents_field = self.Field(field_name, text, contents_field_type)
                doc.add(contents_field)

                # Add to index
                self.writer.addDocument(doc)

                count += 1

                # Periodic commit
                if count % commit_every == 0:
                    self.writer.commit()
                    elapsed = time.time() - start_time
                    rate = count / elapsed
                    logger.info(
                        f"Indexed {count:,} documents "
                        f"({rate:.1f} docs/sec)"
                    )

            except Exception as e:
                logger.warning(f"Failed to index document {docid}: {e}")
                continue

        # Final commit
        logger.info("Committing index...")
        self.writer.commit()

        # Force merge (optional, improves query performance)
        logger.info("Optimizing index (force merge)...")
        self.writer.forceMerge(1)

        # Close writer
        logger.info("Closing index writer...")
        self.writer.close()

        elapsed = time.time() - start_time
        rate = count / elapsed if elapsed > 0 else 0

        logger.info(
            f"Indexing complete: {count:,} documents in {elapsed:.1f}s "
            f"({rate:.1f} docs/sec)"
        )

        return count


# ---------------------------------------------------------------------------
# Index statistics
# ---------------------------------------------------------------------------

def print_index_statistics(index_dir: str):
    """
    Print statistics about the index.

    IMPORTANT: Uses PathlibPath (not Lucene Path) for file operations.

    Args:
        index_dir: Path to Lucene index
    """
    logger.info("\n" + "=" * 60)
    logger.info("INDEX STATISTICS")
    logger.info("=" * 60)

    classes = get_lucene_classes()

    DirectoryReader = classes['IndexReader']
    FSDirectory = classes['FSDirectory']
    JavaPaths = classes['JavaPaths']  # Use this for Lucene

    # Open index (use Lucene JavaPaths)
    directory = FSDirectory.open(JavaPaths.get(index_dir))
    reader = DirectoryReader.open(directory)

    # Basic stats
    num_docs = reader.numDocs()
    max_doc = reader.maxDoc()
    num_deleted = max_doc - num_docs

    logger.info(f"Number of documents:    {num_docs:,}")
    logger.info(f"Number of deleted docs: {num_deleted:,}")
    logger.info(f"Max doc ID:             {max_doc:,}")

    # Field stats
    fields = list(reader.getFieldNames())
    logger.info(f"Fields:                 {', '.join(fields)}")

    # Term stats for 'contents' field
    if 'contents' in fields:
        terms = reader.terms('contents')
        if terms:
            num_terms = terms.size()
            logger.info(f"Number of unique terms: {num_terms:,}")

            # Get total term frequency
            sum_total_term_freq = reader.getSumTotalTermFreq('contents')
            if sum_total_term_freq > 0:
                logger.info(f"Total term occurrences: {sum_total_term_freq:,}")
                avg_doc_len = sum_total_term_freq / num_docs if num_docs > 0 else 0
                logger.info(f"Average document length:{avg_doc_len:.2f} terms")

    # Index size (use PathlibPath for file operations!)
    index_path_obj = PathlibPath(index_dir)  # NOT Lucene Path!
    total_size = sum(
        f.stat().st_size
        for f in index_path_obj.rglob('*')
        if f.is_file()
    )
    size_mb = total_size / (1024 * 1024)
    logger.info(f"Index size on disk:     {size_mb:.2f} MB")

    reader.close()

    logger.info("=" * 60 + "\n")


def validate_index(index_dir: str, sample_queries: list = None) -> bool:
    """
    Validate index by running sample queries.

    Args:
        index_dir: Path to Lucene index
        sample_queries: List of test queries

    Returns:
        True if validation successful
    """
    logger.info("Validating index...")

    if sample_queries is None:
        sample_queries = [
            "machine learning",
            "neural networks",
            "information retrieval",
        ]

    try:
        classes = get_lucene_classes()

        DirectoryReader = classes['IndexReader']
        FSDirectory = classes['FSDirectory']
        JavaPaths = classes['JavaPaths']
        IndexSearcher = classes['IndexSearcher']
        QueryParser = classes['QueryParser']
        EnglishAnalyzer = classes['EnglishAnalyzer']

        # Open index
        directory = FSDirectory.open(JavaPaths.get(index_dir))
        reader = DirectoryReader.open(directory)
        searcher = IndexSearcher(reader)
        analyzer = EnglishAnalyzer()

        validation_passed = True

        for query_text in sample_queries:
            try:
                # Parse and search
                parser = QueryParser("contents", analyzer)
                query = parser.parse(query_text)

                # Get top 10 results
                top_docs = searcher.search(query, 10)

                if top_docs.totalHits.value > 0:
                    logger.info(
                        f"  Query '{query_text}': "
                        f"{top_docs.totalHits.value} hits ✓"
                    )
                else:
                    logger.warning(f"  Query '{query_text}': No results")
                    validation_passed = False

            except Exception as e:
                logger.error(f"  Query '{query_text}' failed: {e}")
                validation_passed = False

        reader.close()

        if validation_passed:
            logger.info("Index validation: PASSED ✓")
        else:
            logger.warning("Index validation: FAILED ✗")

        return validation_passed

    except Exception as e:
        logger.error(f"Index validation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Lucene index for MS-MEQE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--collection",
        type=str,
        help="IR dataset collection name (e.g., 'msmarco-passage')",
    )
    source_group.add_argument(
        "--input",
        type=str,
        help="Input file path (for JSONL or TSV)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for Lucene index",
    )

    # CRITICAL: Lucene JAR path
    parser.add_argument(
        "--lucene-path",
        type=str,
        required=True,
        help="Path to Lucene JAR files (e.g., /path/to/lucene-9.x.x/*)",
    )

    # Input format (for --input)
    parser.add_argument(
        "--format",
        type=str,
        choices=['jsonl', 'tsv'],
        default='jsonl',
        help="Input file format (used with --input)",
    )

    # JSONL options
    parser.add_argument(
        "--docid-field",
        type=str,
        default='docid',
        help="Field name for document ID (JSONL mode)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default='text',
        help="Field name for document text (JSONL mode)",
    )

    # TSV options
    parser.add_argument(
        "--docid-col",
        type=int,
        default=0,
        help="Column index for document ID (TSV mode)",
    )
    parser.add_argument(
        "--text-col",
        type=int,
        default=1,
        help="Column index for document text (TSV mode)",
    )

    # Indexing options
    parser.add_argument(
        "--analyzer",
        type=str,
        choices=['standard', 'english', 'whitespace'],
        default='english',
        help="Lucene analyzer",
    )
    parser.add_argument(
        "--ram-buffer",
        type=int,
        default=4096,
        help="RAM buffer size (MB) - larger is faster",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.2,
        help="BM25 k1 parameter (term saturation)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (length normalization)",
    )
    parser.add_argument(
        "--field-name",
        type=str,
        default='contents',
        help="Field name for document text in index",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=10000,
        help="Commit frequency (number of documents)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to index (for testing)",
    )

    # Validation
    parser.add_argument(
        "--validate",
        action='store_true',
        help="Validate index after creation",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    )

    logger.info("=" * 60)
    logger.info("LUCENE INDEXING FOR MS-MEQE")
    logger.info("=" * 60)

    # Initialize Lucene
    if not initialize_lucene(args.lucene_path):
        logger.error("Failed to initialize Lucene")
        sys.exit(1)

    # Create document iterator
    if args.collection:
        logger.info(f"Source: IR dataset collection '{args.collection}'")
        doc_iterator = IRDatasetIterator(args.collection)

    elif args.input:
        if args.format == 'jsonl':
            logger.info(f"Source: JSONL file '{args.input}'")
            doc_iterator = JSONLIterator(
                file_path=args.input,
                docid_field=args.docid_field,
                text_field=args.text_field,
            )

        elif args.format == 'tsv':
            logger.info(f"Source: TSV file '{args.input}'")
            doc_iterator = TSVIterator(
                file_path=args.input,
                docid_col=args.docid_col,
                text_col=args.text_col,
            )

    # Create indexer
    indexer = LuceneIndexer(
        index_dir=args.output,
        analyzer=args.analyzer,
        ram_buffer_size_mb=args.ram_buffer,
        k1=args.k1,
        b=args.b,
    )

    # Index documents
    num_indexed = indexer.index_documents(
        doc_iterator=doc_iterator,
        field_name=args.field_name,
        commit_every=args.commit_every,
        max_docs=args.max_docs,
    )

    # Print statistics
    print_index_statistics(args.output)

    # Validate if requested
    if args.validate:
        validate_index(args.output)

    logger.info("\n" + "=" * 60)
    logger.info("INDEXING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Index location: {args.output}")
    logger.info(f"Documents indexed: {num_indexed:,}")
    logger.info("")
    logger.info("Usage in MS-MEQE:")
    logger.info("  from msmeqe.expansion.candidate_extraction_pipeline import MultiSourceCandidateExtractor")
    logger.info("  extractor = MultiSourceCandidateExtractor(")
    logger.info(f"      index_path='{args.output}',")
    logger.info("      encoder=encoder,")
    logger.info("  )")


if __name__ == "__main__":
    main()