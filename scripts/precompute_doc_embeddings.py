# scripts/precompute_doc_embeddings.py

"""
Precompute document embeddings for efficient dense retrieval.

This script:
  1. Loads all documents from the collection
  2. Encodes them using Sentence-BERT
  3. Saves embeddings and doc IDs to disk

Run this ONCE before training the budget model.

Usage:
    python scripts/precompute_doc_embeddings.py \\
        --collection msmarco-passage \\
        --index-path data/msmarco_index \\
        --output-dir data/msmarco_index \\
        --batch-size 512
"""

import logging
import argparse
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import ir_datasets

from msmeqe.reranking.semantic_encoder import SemanticEncoder

logger = logging.getLogger(__name__)


def precompute_document_embeddings(
        collection_name: str,
        output_dir: str,
        encoder: SemanticEncoder,
        batch_size: int = 512,
        max_docs: int = None,
):
    """
    Precompute and save document embeddings.

    Args:
        collection_name: IR dataset collection name
        output_dir: Where to save embeddings
        encoder: SemanticEncoder instance
        batch_size: Encoding batch size
        max_docs: Limit number of documents (for testing)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading collection: {collection_name}")
    dataset = ir_datasets.load(collection_name)

    # Get total document count
    try:
        total_docs = dataset.docs_count()
        if max_docs:
            total_docs = min(total_docs, max_docs)
    except:
        total_docs = None

    logger.info(f"Encoding documents (batch_size={batch_size})...")

    doc_ids = []
    all_embeddings = []

    batch_texts = []
    batch_ids = []

    for i, doc in enumerate(tqdm(dataset.docs_iter(), total=total_docs, desc="Processing")):
        if max_docs and i >= max_docs:
            break

        # Get document text
        if hasattr(doc, 'text'):
            doc_text = doc.text
        elif hasattr(doc, 'body'):
            doc_text = doc.body
        else:
            continue

        batch_texts.append(doc_text)
        batch_ids.append(doc.doc_id)

        # Encode batch when full
        if len(batch_texts) >= batch_size:
            embeddings = encoder.encode(batch_texts)
            all_embeddings.append(embeddings)
            doc_ids.extend(batch_ids)

            batch_texts = []
            batch_ids = []

    # Encode remaining batch
    if batch_texts:
        embeddings = encoder.encode(batch_texts)
        all_embeddings.append(embeddings)
        doc_ids.extend(batch_ids)

    # Concatenate all embeddings
    logger.info("Concatenating embeddings...")
    doc_embeddings = np.vstack(all_embeddings)

    logger.info(f"Encoded {len(doc_ids)} documents, embedding shape: {doc_embeddings.shape}")

    # Normalize embeddings (for cosine similarity)
    logger.info("Normalizing embeddings...")
    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    doc_embeddings = doc_embeddings / (norms + 1e-12)

    # Save embeddings
    embeddings_path = output_dir / "doc_embeddings.npy"
    logger.info(f"Saving embeddings to {embeddings_path}")
    np.save(str(embeddings_path), doc_embeddings)

    # Save doc IDs
    ids_path = output_dir / "doc_ids.json"
    logger.info(f"Saving doc IDs to {ids_path}")
    with open(ids_path, 'w') as f:
        json.dump(doc_ids, f)

    # Log file sizes
    emb_size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    logger.info(f"Embeddings file size: {emb_size_mb:.1f} MB")

    logger.info("Precomputation complete!")


def main():
    parser = argparse.ArgumentParser(description="Precompute document embeddings")
    parser.add_argument("--collection", type=str, required=True,
                        help="IR dataset collection name")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--model-name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-BERT model name")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Encoding batch size")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum documents to encode (for testing)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    # Initialize encoder
    logger.info(f"Loading encoder: {args.model_name}")
    encoder = SemanticEncoder(model_name=args.model_name)

    # Precompute embeddings
    precompute_document_embeddings(
        collection_name=args.collection,
        output_dir=args.output_dir,
        encoder=encoder,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()