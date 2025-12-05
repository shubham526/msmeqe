import traceback

import torch
import logging
import json
from typing import Dict, Iterator, Tuple, Optional, List
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import jnius_config
from src.utils.lucene_utils import get_lucene_classes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)




class BERTTokenBM25Indexer:
    """Creates a Lucene index with BERT token mapping and BM25 scoring capability"""

    def __init__(self, model_name="bert-base-uncased", index_path: str = None):
        """Initialize the indexer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.index_path = index_path

            # Get Lucene classes lazily
            classes = get_lucene_classes()
            for name, cls in classes.items():
                setattr(self, name, cls)

        except Exception as e:
            logger.error(f"Error initializing BERTTokenBM25Indexer: {e}")
            raise

    def create_index(self, documents_iterator: Iterator[Tuple[str, str]], num_docs: Optional[int] = None):
        """
        Create a Lucene index from an iterator of documents.

        Args:
            documents_iterator: An iterator that yields (doc_id, doc_text) tuples.
            num_docs: The total number of documents, for the tqdm progress bar.
        """
        # The index_path is now taken from the object's state
        if self.index_path is None:
            raise ValueError("Indexer must be initialized with an index_path.")

        writer = None
        try:
            directory = self.FSDirectory.open(self.Path.get(self.index_path))
            analyzer = self.EnglishAnalyzer()
            config = self.IndexWriterConfig(analyzer)
            writer = self.IndexWriter(directory, config)

            logger.info(f"Creating index at {self.index_path}")

            # Use the iterator directly in tqdm. It no longer needs .items()
            doc_stream = tqdm(documents_iterator, total=num_docs, desc="Indexing documents")

            for doc_id, doc_text in doc_stream:
                try:
                    tokens = self.tokenizer.tokenize(doc_text)
                    # Build full words and track positions
                    words = []
                    word_positions = []  # List of (word, start_pos, end_pos)
                    current_word = ""
                    word_start_pos = 0

                    for i, token in enumerate(tokens):
                        if token.startswith('##'):
                            current_word += token[2:]
                        else:
                            if current_word:
                                words.append(current_word)
                                word_positions.append((len(words) - 1, word_start_pos, i - 1))
                            current_word = token
                            word_start_pos = i

                    # Add last word
                    if current_word:
                        words.append(current_word)
                        word_positions.append((len(words) - 1, word_start_pos, len(tokens) - 1))

                    # Create and add Lucene document
                    lucene_doc = self.Document()
                    lucene_doc.add(self.StringField("id", doc_id, self.FieldStore.YES))
                    lucene_doc.add(self.TextField("contents", " ".join(words), self.FieldStore.YES))
                    lucene_doc.add(self.StoredField("positions", json.dumps(word_positions)))

                    writer.addDocument(lucene_doc)

                    # # Use counter instead of doc_id for progress logging
                    # if count % 1000 == 0:
                    #     logger.info(f"Indexed {count} documents")

                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {e}")
                    continue

            writer.commit()
            logger.info("Indexing completed successfully")

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as e:
                    logger.error(f"Error closing index writer: {e}")


class TokenBM25Scorer:
    def __init__(self, index_path: str, k1: float = 1.2, b: float = 0.75):
        try:
            # Get Lucene classes lazily
            classes = get_lucene_classes()
            for name, cls in classes.items():
                setattr(self, name, cls)

            # Get additional required Lucene classes
            from jnius import autoclass
            self.BM25Similarity = autoclass('org.apache.lucene.search.similarities.BM25Similarity')

            # # Open index and setup searcher with BM25
            directory = self.FSDirectory.open(self.Path.get(index_path))
            self.reader = self.DirectoryReader.open(directory)
            self.searcher = self.IndexSearcher(self.reader.getContext())
            self.searcher.setSimilarity(self.BM25Similarity(k1, b))


        except Exception as e:
            logger.error(f"Error initializing TokenBM25Scorer: {e}")
            raise

    def compute_bm25_term_weight(self, docid: str, terms: list) -> dict:
        """
        Compute BM25 weights for multiple terms in a specific document.

        Args:
            docid: Document ID
            terms: List of terms to compute weights for

        Returns:
            dict: Dictionary mapping terms to their BM25 scores
        """
        try:
            # First verify document exists
            id_term = self.Term("id", str(docid))
            id_query = self.TermQuery(id_term)
            doc_hits = self.searcher.search(id_query, 1)

            if doc_hits.totalHits.value() == 0:
                # logger.warning(f"Document {docid} not found")
                return {term: 0.0 for term in terms}

            # Create analyzer - same as used during indexing
            analyzer = self.EnglishAnalyzer()

            # Get individual scores for each term
            term_scores = {}
            for term in terms:
                # Use analyzer to process the term
                stream = analyzer.tokenStream("contents", term)
                charTermAtt = stream.addAttribute(self.CharTermAttribute)
                stream.reset()

                analyzed_terms = []
                while stream.incrementToken():
                    analyzed_terms.append(charTermAtt.toString())
                stream.end()
                stream.close()

                # logger.info(f"Original term: '{term}', Analyzed terms: {analyzed_terms}")

                if not analyzed_terms:  # Skip if term was entirely stopwords
                    term_scores[term] = 0.0
                    continue

                # Build query for all analyzed terms
                term_builder = self.BooleanQueryBuilder()
                for analyzed_term in analyzed_terms:
                    term_builder.add(
                        self.TermQuery(self.Term("contents", analyzed_term)),
                        self.BooleanClauseOccur.MUST
                    )

                # Combine with document filter
                query_builder = self.BooleanQueryBuilder()
                query_builder.add(id_query, self.BooleanClauseOccur.FILTER)
                query_builder.add(term_builder.build(), self.BooleanClauseOccur.MUST)
                final_query = query_builder.build()

                # Search and get score
                hits = self.searcher.search(final_query, 1)
                if hits.totalHits.value() > 0:
                    term_scores[term] = hits.scoreDocs[0].score
                    # logger.info(f"Found term '{term}' with score {term_scores[term]}")
                else:
                    term_scores[term] = 0.0
                    # logger.info(f"Term '{term}' not found in document")

            return term_scores

        except Exception as e:
            logger.error(f"Error computing BM25 score: {e}")
            return {term: 0.0 for term in terms}

    def compute_collection_level_bm25(self, terms: List[str], max_docs: int = 100) -> Dict[str, float]:
        """
        Score terms against entire collection instead of single document.

        Args:
            terms: Terms to score
            max_docs: Maximum documents to consider per term

        Returns:
            Dict mapping terms to collection-level BM25 scores
        """
        term_scores = {}

        for term in terms:
            try:
                # Create query for this term
                term_query = self.TermQuery(self.Term("contents", term))

                # Search across collection
                hits = self.searcher.search(term_query, max_docs)

                if hits.totalHits.value() > 0:
                    # Use average of top document scores
                    top_scores = [hit.score for hit in hits.scoreDocs[:min(10, len(hits.scoreDocs))]]
                    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
                    term_scores[term] = avg_score
                else:
                    term_scores[term] = 0.0

            except Exception as e:
                logger.debug(f"Error scoring term '{term}': {e}")
                term_scores[term] = 0.0

        return term_scores

    def get_token_scores(self, query: str, doc_id: str, max_length: int = 512) -> torch.Tensor:
        """
        Get BM25 scores for each token position in a document based on a query.

        Args:
            query: Search query
            doc_id: Document ID
            max_length: Maximum sequence length to consider

        Returns:
            torch.Tensor: Tensor of token scores
        """
        if not isinstance(query, str) or not query.strip():
            return torch.zeros(max_length)

        try:
            # Search for document by ID
            id_term = self.Term("id", str(doc_id))
            id_query = self.TermQuery(id_term)
            hits = self.searcher.search(id_query, 1)

            if hits.totalHits.value() == 0:
                return torch.zeros(max_length)

            # Get document content and positions
            doc = self.searcher.storedFields().document(hits.scoreDocs[0].doc)
            doc_content = doc.get("contents")
            positions_str = doc.get("positions")

            if not doc_content or not positions_str:
                return torch.zeros(max_length)

            # Parse positions and content
            word_positions = json.loads(positions_str)
            doc_words = doc_content.split()

            # Get scores for all query terms at once
            query_terms = query.lower().split()
            term_scores = self.compute_bm25_term_weight(doc_id, query_terms)

            # Initialize score array
            token_scores = np.zeros(max_length)

            # Apply scores to token positions
            for word_idx, start_pos, end_pos in word_positions:
                if not (0 <= word_idx < len(doc_words)):
                    continue

                if not (0 <= start_pos < max_length):
                    continue

                end_pos = min(end_pos, max_length - 1)
                word = doc_words[word_idx].lower()

                # Look up score for this word
                score = term_scores.get(word, 0.0)
                if score > 0:
                    token_scores[start_pos:end_pos + 1] = score
            # print(token_scores)

            return torch.FloatTensor(token_scores)

        except Exception as e:
            logger.error(f"Error in get_token_scores: {e}", exc_info=True)
            print(traceback.format_exc())
            return torch.zeros(max_length)

    def __del__(self):
        try:
            if hasattr(self, 'reader'):
                self.reader.close()
        except Exception as e:
            logger.error(f"Error closing index reader: {e}")