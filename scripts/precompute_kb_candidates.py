#!/usr/bin/env python3
# scripts/precompute_kb_candidates.py

"""
Precompute KB candidates for all queries.

This script runs WAT entity linking once for all queries and stores
the results in a JSONL file. This avoids repeated WAT calls during
experiments.

Output format (one JSON per line):
    {
        "qid": "123",
        "query_text": "neural networks",
        "candidates": [
            {
                "term": "artificial neural network",
                "entity_uri": "https://en.wikipedia.org/wiki/Artificial_neural_network",
                "confidence": 0.95,
                "rank": 1
            },
            ...
        ]
    }

Usage:
    python scripts/precompute_kb_candidates.py \\
        --topics data/queries.tsv \\
        --wat-output data/wat_entities.jsonl \\
        --output data/kb_candidates.jsonl \\
        --max-candidates 30
"""

import logging
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_queries_from_file(topics_file: str) -> Dict[str, str]:
    """
    Load queries from file.

    Supports TSV, JSON, JSONL formats.
    """
    logger.info(f"Loading queries from: {topics_file}")

    topics_path = Path(topics_file)
    queries = {}

    if topics_path.suffix == '.json':
        with open(topics_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                queries = data
            elif isinstance(data, list):
                queries = {str(i): q for i, q in enumerate(data)}

    elif topics_path.suffix == '.jsonl':
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

    logger.info(f"Loaded {len(queries)} queries")
    return queries


def precompute_kb_candidates(
        queries: Dict[str, str],
        wat_output_path: str,
        wikiid_desc_path: str = None,
        max_candidates: int = 30,
) -> Dict[str, List[Dict]]:
    """
    Precompute KB candidates for all queries.

    Args:
        queries: Query ID -> query text
        wat_output_path: Path to WAT entity linking output
        wikiid_desc_path: Optional path to Wikipedia descriptions
        max_candidates: Maximum candidates per query

    Returns:
        Dictionary {query_id: [candidate_dicts]}
    """
    from msmeqe.expansion.kb_expansion import KBCandidateExtractor

    logger.info("Initializing KB extractor...")
    kb_extractor = KBCandidateExtractor(
        wat_output_path=wat_output_path,
        wikiid_desc_path=wikiid_desc_path,
    )

    logger.info(f"Extracting KB candidates for {len(queries)} queries...")

    all_kb_candidates = {}
    failed_queries = []

    for qid, qtext in tqdm(queries.items(), desc="Extracting KB candidates"):
        try:
            # Extract KB candidates with metadata
            kb_candidates = kb_extractor.extract_candidates_with_metadata(
                query_text=qtext,
                query_id=qid,
            )

            # Convert to dictionaries
            candidate_dicts = []
            for rank, kb_cand in enumerate(kb_candidates[:max_candidates], start=1):
                candidate_dict = {
                    'term': kb_cand.term,
                    'entity_uri': kb_cand.source_entity.uri if hasattr(kb_cand, 'source_entity') else None,
                    'entity_label': kb_cand.source_entity.label if hasattr(kb_cand, 'source_entity') else kb_cand.term,
                    'confidence': float(kb_cand.confidence),
                    'rank': rank,
                }
                candidate_dicts.append(candidate_dict)

            all_kb_candidates[qid] = candidate_dicts

            if not candidate_dicts:
                logger.warning(f"No KB candidates for query {qid}: {qtext}")

        except Exception as e:
            logger.error(f"Failed to extract KB candidates for query {qid}: {e}")
            failed_queries.append(qid)
            all_kb_candidates[qid] = []
            continue

    logger.info(f"Extracted KB candidates for {len(all_kb_candidates)} queries")
    if failed_queries:
        logger.warning(f"Failed for {len(failed_queries)} queries: {failed_queries[:10]}...")

    return all_kb_candidates


def save_kb_candidates(
        kb_candidates: Dict[str, List[Dict]],
        queries: Dict[str, str],
        output_path: str,
):
    """
    Save KB candidates to JSONL file.

    Args:
        kb_candidates: Query ID -> [candidate_dicts]
        queries: Query ID -> query text
        output_path: Output JSONL path
    """
    logger.info(f"Saving KB candidates to: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for qid in sorted(kb_candidates.keys()):
            entry = {
                'qid': qid,
                'query_text': queries.get(qid, ''),
                'candidates': kb_candidates[qid],
                'num_candidates': len(kb_candidates[qid]),
            }
            f.write(json.dumps(entry) + '\n')

    logger.info(f"Saved KB candidates for {len(kb_candidates)} queries")

    # Print statistics
    total_candidates = sum(len(cands) for cands in kb_candidates.values())
    avg_candidates = total_candidates / len(kb_candidates) if kb_candidates else 0

    queries_with_candidates = sum(1 for cands in kb_candidates.values() if len(cands) > 0)

    logger.info("\nStatistics:")
    logger.info(f"  Total queries:              {len(kb_candidates)}")
    logger.info(f"  Queries with candidates:    {queries_with_candidates}")
    logger.info(f"  Queries without candidates: {len(kb_candidates) - queries_with_candidates}")
    logger.info(f"  Total candidates:           {total_candidates}")
    logger.info(f"  Average per query:          {avg_candidates:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute KB candidates for queries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--topics",
        type=str,
        required=True,
        help="Query file (TSV/JSON/JSONL)",
    )
    parser.add_argument(
        "--wat-output",
        type=str,
        required=True,
        help="Path to WAT entity linking output",
    )
    parser.add_argument(
        "--wikiid-desc",
        type=str,
        default=None,
        help="Optional path to Wikipedia descriptions TSV",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file for KB candidates",
    )

    # Options
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=30,
        help="Maximum candidates per query",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    )

    logger.info("=" * 60)
    logger.info("PRECOMPUTING KB CANDIDATES")
    logger.info("=" * 60)
    logger.info(f"Topics:      {args.topics}")
    logger.info(f"WAT output:  {args.wat_output}")
    logger.info(f"Output:      {args.output}")

    # Load queries
    queries = load_queries_from_file(args.topics)

    # Precompute KB candidates
    kb_candidates = precompute_kb_candidates(
        queries=queries,
        wat_output_path=args.wat_output,
        wikiid_desc_path=args.wikiid_desc,
        max_candidates=args.max_candidates,
    )

    # Save to file
    save_kb_candidates(kb_candidates, queries, args.output)

    logger.info("\n" + "=" * 60)
    logger.info("PRECOMPUTATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"KB candidates saved to: {args.output}")
    logger.info("")
    logger.info("Usage in evaluation:")
    logger.info(f"  python scripts/run_msmeqe_evaluation.py \\")
    logger.info(f"      --kb-candidates {args.output} \\")
    logger.info(f"      ...")


if __name__ == "__main__":
    main()