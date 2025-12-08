# src/expansion/kb_expansion.py

"""
Knowledge Base Expansion Module

Extracts expansion candidates from knowledge base entities using WAT entity linking.
Integrates with the MS-MEQE pipeline to provide structured entity information.

Usage:
    from msmeqe.expansion.kb_expansion import KBCandidateExtractor

    extractor = KBCandidateExtractor(
        wat_output_path="data/wat_entities.jsonl",
        wikiid_desc_path="data/wikiid_descriptions.tsv"
    )

    candidates = extractor.extract_candidates(
        query_text="transformer architecture attention mechanism",
        query_id="q123"
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path

from msmeqe.expansion.wat_adapter import wat_json_to_kb_entities
from msmeqe.utils.wiki_desc_utils import load_wikiid_to_desc_tsv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KBEntity:
    """
    Representation of a knowledge base entity.

    Fields:
        uri: Entity URI (e.g., DBpedia resource URI)
        label: Primary label/title of the entity
        score: Confidence score from entity linker (0-1)
        start: Character start position in source text (optional)
        end: Character end position in source text (optional)
        description: Entity description text (optional)
    """
    uri: str
    label: str
    score: float
    start: Optional[int] = None
    end: Optional[int] = None
    description: Optional[str] = None


@dataclass
class KBCandidate:
    """
    A candidate expansion term derived from a KB entity.

    Fields:
        term: The actual term string (could be label, alias, or description term)
        source_entity: The entity this term came from
        term_type: Type of term ("label", "alias", "description", "related")
        confidence: Combined confidence score
    """
    term: str
    source_entity: KBEntity
    term_type: str  # "label", "alias", "description", "related"
    confidence: float


# ---------------------------------------------------------------------------
# Main KB Candidate Extractor
# ---------------------------------------------------------------------------

class KBCandidateExtractor:
    """
    Extract expansion candidates from knowledge base entities.

    This class:
      1. Loads pre-computed WAT entity linking results
      2. Enriches entities with Wikipedia descriptions
      3. Extracts candidate terms from entity labels, aliases, and descriptions
      4. Filters and ranks candidates by confidence

    The extractor follows the methodology in Section 3.2 of the paper:
      - Entity linking via WAT (DBpedia Spotlight equivalent)
      - Extraction of primary labels, aliases, and description terms
      - Optional related entity expansion
    """

    def __init__(
            self,
            wat_output_path: Optional[str] = None,
            wikiid_desc_path: Optional[str] = None,
            max_candidates_per_query: int = 30,
            min_entity_confidence: float = 0.1,
            description_max_chars: int = 200,
            include_related_entities: bool = False,
    ):
        """
        Initialize KB candidate extractor.

        Args:
            wat_output_path: Path to WAT JSONL output (one JSON per line)
                            Format: {"doc_id": "q123", "entities": [...]}
            wikiid_desc_path: Path to TSV mapping wikiid -> description
            max_candidates_per_query: Maximum number of KB candidates to return
            min_entity_confidence: Minimum confidence threshold for entities
            description_max_chars: Max characters to extract from descriptions
            include_related_entities: Whether to expand to related entities
        """
        self.max_candidates = max_candidates_per_query
        self.min_confidence = min_entity_confidence
        self.desc_max_chars = description_max_chars
        self.include_related = include_related_entities

        # Load entity linking results
        self.wat_entities: Dict[str, List[KBEntity]] = {}
        if wat_output_path:
            self._load_wat_output(wat_output_path, wikiid_desc_path)

        logger.info(
            f"Initialized KBCandidateExtractor: "
            f"max_candidates={max_candidates_per_query}, "
            f"min_confidence={min_entity_confidence}, "
            f"loaded_queries={len(self.wat_entities)}"
        )

    def _load_wat_output(
            self,
            wat_path: str,
            desc_path: Optional[str]
    ) -> None:
        """
        Load pre-computed WAT entity linking results.

        Args:
            wat_path: Path to WAT JSONL file
            desc_path: Optional path to description TSV
        """
        wat_path = Path(wat_path)
        if not wat_path.exists():
            logger.warning(f"WAT output not found at {wat_path}")
            return

        # Load descriptions if provided
        wikiid_to_desc: Dict[str, str] = {}
        if desc_path:
            desc_path = Path(desc_path)
            if desc_path.exists():
                logger.info(f"Loading entity descriptions from {desc_path}")
                wikiid_to_desc = load_wikiid_to_desc_tsv(
                    str(desc_path),
                    has_header=False
                )
                logger.info(f"Loaded {len(wikiid_to_desc)} entity descriptions")
            else:
                logger.warning(f"Description file not found at {desc_path}")

        # Load WAT entities
        logger.info(f"Loading WAT entities from {wat_path}")
        count = 0

        with open(wat_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    wat_doc = json.loads(line)
                    query_id = wat_doc.get("doc_id")

                    if not query_id:
                        continue

                    # Convert WAT format to KBEntity objects
                    entities = wat_json_to_kb_entities(
                        wat_doc=wat_doc,
                        wikiid_to_desc=wikiid_to_desc
                    )

                    if entities:
                        self.wat_entities[query_id] = entities
                        count += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse WAT JSON line: {e}")
                except Exception as e:
                    logger.warning(f"Error processing WAT entry: {e}")

        logger.info(f"Loaded entities for {count} queries")

    def extract_candidates(
            self,
            query_text: str,
            query_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Extract KB expansion candidates for a query.

        Args:
            query_text: The query string
            query_id: Query identifier (for looking up pre-computed entities)

        Returns:
            List of (term, confidence_score) tuples, sorted by score descending
        """
        # Get entities for this query
        entities = self._get_entities_for_query(query_text, query_id)

        if not entities:
            logger.debug(f"No KB entities found for query: {query_text[:50]}")
            return []

        # Filter by confidence
        entities = [e for e in entities if e.score >= self.min_confidence]

        if not entities:
            logger.debug(f"No entities passed confidence threshold for: {query_text[:50]}")
            return []

        logger.debug(f"Found {len(entities)} KB entities for query: {query_text[:50]}")

        # Extract candidate terms from entities
        candidates = self._extract_terms_from_entities(entities, query_text)

        # Deduplicate and rank
        candidates = self._deduplicate_and_rank(candidates)

        # Truncate to max candidates
        candidates = candidates[:self.max_candidates]

        # Convert to (term, score) format
        result = [(c.term, c.confidence) for c in candidates]

        logger.debug(f"Extracted {len(result)} KB candidates")
        return result

    def extract_candidates_with_metadata(
            self,
            query_text: str,
            query_id: Optional[str] = None,
    ) -> List[KBCandidate]:
        """
        Extract KB candidates with full metadata (for analysis/debugging).

        Args:
            query_text: The query string
            query_id: Query identifier

        Returns:
            List of KBCandidate objects with full metadata
        """
        entities = self._get_entities_for_query(query_text, query_id)

        if not entities:
            return []

        entities = [e for e in entities if e.score >= self.min_confidence]

        if not entities:
            return []

        candidates = self._extract_terms_from_entities(entities, query_text)
        candidates = self._deduplicate_and_rank(candidates)

        return candidates[:self.max_candidates]

    def _get_entities_for_query(
            self,
            query_text: str,
            query_id: Optional[str],
    ) -> List[KBEntity]:
        """
        Get entities for a query (from pre-computed results or on-the-fly).

        Args:
            query_text: Query text
            query_id: Query ID

        Returns:
            List of KBEntity objects
        """
        # Try to get pre-computed entities
        if query_id and query_id in self.wat_entities:
            return self.wat_entities[query_id]

        # If no pre-computed entities, we can't do on-the-fly linking
        # (WAT API would be needed for that)
        logger.debug(
            f"No pre-computed entities for query_id={query_id}, "
            f"returning empty list"
        )
        return []

    def _extract_terms_from_entities(
            self,
            entities: List[KBEntity],
            query_text: str,
    ) -> List[KBCandidate]:
        """
        Extract candidate terms from entity labels and descriptions.

        Following paper Section 3.2:
          - Entity primary label
          - Alternative labels/aliases (if available)
          - Key terms from entity description (first sentence, ≤200 chars)

        Args:
            entities: List of entities
            query_text: Original query text

        Returns:
            List of KBCandidate objects
        """
        candidates: List[KBCandidate] = []
        query_lower = query_text.lower()

        for entity in entities:
            # 1. Primary label
            if entity.label and entity.label.strip():
                label = entity.label.strip()
                # Skip if label is already in query (no point expanding with it)
                if label.lower() not in query_lower:
                    candidates.append(KBCandidate(
                        term=label,
                        source_entity=entity,
                        term_type="label",
                        confidence=entity.score,
                    ))

            # 2. Description terms (extract key terms from first sentence)
            if entity.description:
                desc_terms = self._extract_description_terms(
                    entity.description,
                    entity,
                    query_lower,
                )
                candidates.extend(desc_terms)

            # 3. Related entities (optional, usually not needed)
            if self.include_related:
                # This would require additional data (e.g., entity relations)
                # Not implemented in base version
                pass

        return candidates

    def _extract_description_terms(
            self,
            description: str,
            entity: KBEntity,
            query_lower: str,
    ) -> List[KBCandidate]:
        """
        Extract key terms from entity description.

        Strategy:
          - Take first sentence (up to '.', '!', '?')
          - Truncate to max_chars
          - Extract noun phrases or significant unigrams/bigrams

        For simplicity, we extract:
          - Individual words (≥4 chars, not stopwords)
          - Bigrams

        Args:
            description: Entity description text
            entity: Source entity
            query_lower: Lowercased query text

        Returns:
            List of KBCandidate objects
        """
        if not description or not description.strip():
            return []

        # Truncate to first sentence and max chars
        desc = description.strip()

        # Find first sentence
        for delimiter in ['.', '!', '?']:
            idx = desc.find(delimiter)
            if idx > 0:
                desc = desc[:idx]
                break

        # Truncate to max chars
        desc = desc[:self.desc_max_chars]

        # Tokenize (simple whitespace + punctuation removal)
        import re
        tokens = re.findall(r'\b[a-z]+\b', desc.lower())

        # Filter tokens
        stopwords = self._get_stopwords()
        tokens = [
            t for t in tokens
            if len(t) >= 4 and t not in stopwords and t not in query_lower
        ]

        candidates: List[KBCandidate] = []

        # Add unigrams
        for token in tokens:
            candidates.append(KBCandidate(
                term=token,
                source_entity=entity,
                term_type="description",
                confidence=entity.score * 0.7,  # Penalize description terms
            ))

        # Add bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            candidates.append(KBCandidate(
                term=bigram,
                source_entity=entity,
                term_type="description",
                confidence=entity.score * 0.8,  # Bigrams slightly better
            ))

        return candidates

    def _deduplicate_and_rank(
            self,
            candidates: List[KBCandidate],
    ) -> List[KBCandidate]:
        """
        Deduplicate candidates and rank by confidence.

        If the same term appears multiple times (from different entities),
        keep the one with highest confidence.

        Args:
            candidates: List of candidates

        Returns:
            Deduplicated and sorted list
        """
        # Group by term (case-insensitive)
        term_to_best: Dict[str, KBCandidate] = {}

        for cand in candidates:
            term_key = cand.term.lower()

            if term_key not in term_to_best:
                term_to_best[term_key] = cand
            else:
                # Keep higher confidence
                if cand.confidence > term_to_best[term_key].confidence:
                    term_to_best[term_key] = cand

        # Sort by confidence descending
        unique_candidates = list(term_to_best.values())
        unique_candidates.sort(key=lambda c: c.confidence, reverse=True)

        return unique_candidates

    @staticmethod
    def _get_stopwords() -> Set[str]:
        """
        Get basic English stopwords.

        Returns:
            Set of stopword strings
        """
        # Basic stopword list (you can expand this)
        return {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work',
            'first', 'well', 'way', 'even', 'new', 'want', 'because',
            'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was',
            'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having',
            'may', 'such', 'being', 'through', 'where', 'much', 'should',
            'very', 'does', 'both', 'each', 'more', 'might', 'must',
        }


# ---------------------------------------------------------------------------
# Convenience function for quick testing
# ---------------------------------------------------------------------------

def build_kb_candidates(
        query_text: str,
        wat_entities: List[KBEntity],
        max_candidates: int = 30,
        min_confidence: float = 0.1,
) -> List[Tuple[str, float]]:
    """
    Build KB candidates from a list of entities (convenience function).

    This is useful when you already have entities and just want candidates.

    Args:
        query_text: Query text
        wat_entities: List of KBEntity objects
        max_candidates: Max number of candidates
        min_confidence: Minimum confidence threshold

    Returns:
        List of (term, confidence) tuples
    """
    extractor = KBCandidateExtractor(
        max_candidates_per_query=max_candidates,
        min_entity_confidence=min_confidence,
    )

    # Manually inject entities
    extractor.wat_entities["temp_query"] = wat_entities

    return extractor.extract_candidates(
        query_text=query_text,
        query_id="temp_query",
    )


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def _main_cli():
    """
    Simple CLI for testing KB expansion.

    Example:
        python -m msmeqe.expansion.kb_expansion \\
            --wat-output data/wat_entities.jsonl \\
            --query "transformer neural networks"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="KB candidate extraction for query expansion"
    )
    parser.add_argument(
        "--wat-output",
        type=str,
        required=True,
        help="Path to WAT JSONL output",
    )
    parser.add_argument(
        "--wikiid-desc",
        type=str,
        default=None,
        help="Path to wikiid->description TSV",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text",
    )
    parser.add_argument(
        "--query-id",
        type=str,
        default=None,
        help="Query ID (for looking up pre-computed entities)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=30,
        help="Maximum number of candidates",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # Initialize extractor
    extractor = KBCandidateExtractor(
        wat_output_path=args.wat_output,
        wikiid_desc_path=args.wikiid_desc,
        max_candidates_per_query=args.max_candidates,
    )

    # Extract candidates
    candidates = extractor.extract_candidates(
        query_text=args.query,
        query_id=args.query_id,
    )

    # Print results
    print(f"\nQuery: {args.query}")
    print(f"Query ID: {args.query_id}")
    print(f"\nKB Expansion Candidates ({len(candidates)}):")
    print("-" * 60)

    for i, (term, score) in enumerate(candidates, 1):
        print(f"{i:2d}. {term:40s} {score:.4f}")


if __name__ == "__main__":
    _main_cli()