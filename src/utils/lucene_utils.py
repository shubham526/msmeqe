import os
import logging
import jnius_config

logger = logging.getLogger(__name__)

def setup_lucene(lucene_path: str):
    """Setup JVM and Lucene classpath"""
    jnius_config.set_classpath(lucene_path)

def get_lucene_classes():
    """Lazy load Lucene classes only when needed"""
    _lucene_classes = None
    if _lucene_classes is None:
        try:
            from jnius import autoclass
            _lucene_classes = {
                'FSDirectory': autoclass('org.apache.lucene.store.FSDirectory'),
                'Path': autoclass('java.nio.file.Paths'),
                'Document': autoclass('org.apache.lucene.document.Document'),
                'Field': autoclass('org.apache.lucene.document.Field'),
                'TextField': autoclass('org.apache.lucene.document.TextField'),
                'StringField': autoclass('org.apache.lucene.document.StringField'),
                'StoredField': autoclass('org.apache.lucene.document.StoredField'),
                'EnglishAnalyzer': autoclass('org.apache.lucene.analysis.en.EnglishAnalyzer'),
                'IndexWriterConfig': autoclass('org.apache.lucene.index.IndexWriterConfig'),
                'IndexWriter': autoclass('org.apache.lucene.index.IndexWriter'),
                'FieldStore': autoclass('org.apache.lucene.document.Field$Store'),
                'IndexReader': autoclass('org.apache.lucene.index.DirectoryReader'),
                'IndexSearcher': autoclass('org.apache.lucene.search.IndexSearcher'),
                'Term': autoclass('org.apache.lucene.index.Term'),
                'TermQuery': autoclass('org.apache.lucene.search.TermQuery'),
                'BooleanQuery' : autoclass('org.apache.lucene.search.BooleanQuery'),
                'BooleanQueryBuilder' : autoclass('org.apache.lucene.search.BooleanQuery$Builder'),
                'BooleanClause': autoclass('org.apache.lucene.search.BooleanClause'),
                'BooleanClauseOccur': autoclass('org.apache.lucene.search.BooleanClause$Occur'),
                'BM25Similarity' : autoclass('org.apache.lucene.search.similarities.BM25Similarity'),
                'ConstantScoreQuery': autoclass('org.apache.lucene.search.ConstantScoreQuery'),
                'DirectoryReader' : autoclass('org.apache.lucene.index.DirectoryReader'),
                'CharTermAttribute': autoclass('org.apache.lucene.analysis.tokenattributes.CharTermAttribute')

            }
        except Exception as e:
            logger.error(f"Error loading Lucene classes: {e}")
            raise
    return _lucene_classes


def initialize_lucene(lucene_path: str) -> bool:
    """
    Initialize Lucene with proper classpath settings

    Args:
        lucene_path: Path to Lucene JAR files

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        # Prevent automatic JVM startup
        if 'CLASSPATH' in os.environ:
            del os.environ['CLASSPATH']

        # Build classpath
        lucene_path = os.path.abspath(lucene_path)
        required_jars = [
            'lucene-core-10.1.0.jar',
            'lucene-analysis-common-10.1.0.jar',
            'lucene-queryparser-10.1.0.jar',
            'lucene-memory-10.1.0.jar'
        ]

        jar_paths = []
        for jar in required_jars:
            full_path = os.path.join(lucene_path, jar)
            if not os.path.exists(full_path):
                raise ValueError(f"Required JAR not found: {full_path}")
            jar_paths.append(full_path)
            logger.info(f"Adding to classpath: {full_path}")

        # Join paths with OS-specific separator
        classpath = os.pathsep.join(jar_paths)

        # Set options and classpath before any JVM-related imports
        jnius_config.add_options(
            '-Xmx4096m',
            '-Xms1024m'
        )
        jnius_config.set_classpath(classpath)

        # Now try to verify the setup
        try:
            from jnius import autoclass
            FSDirectory = autoclass('org.apache.lucene.store.FSDirectory')
            logger.info("Successfully verified Lucene class loading")
            return True
        except Exception as e:
            logger.error(f"Failed to verify Lucene setup: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Failed to initialize Lucene: {str(e)}")
        return False