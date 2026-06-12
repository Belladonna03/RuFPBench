"""Index Epstein email data into ChromaDB with BM25 sparse + OpenAI dense embeddings.

Pipeline: download epstein_only.json from Google Drive → index.

Data prep (CSV parsing, empty-body removal, formatting, deduplication, chunking)
is done upstream.  The result is epstein_only.json — a dict mapping thread_id
(str) to a list of chunk dicts [{text, metadata}], hosted on Google Drive.
"""
import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any

import chromadb
from chromadb import *
from chromadb.utils.embedding_functions import *
from openai import OpenAI
from dotenv import load_dotenv

from ...core.indexing import embed_in_batches, add_to_chroma_with_retry, create_bm25_vectors

load_dotenv()

# Google Drive file ID for epstein_only.json
EPSTEIN_ONLY_FILE_ID = "1kxFzGtee_pLBPq302A_2UqKcEhYU25_0"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_data(output_dir: str) -> str:
    """Download epstein_only.json from Google Drive. Returns path to the file."""
    import gdown

    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, "epstein_only.json")

    if os.path.exists(dest):
        print(f"Already downloaded: {dest}")
        return dest

    url = f"https://drive.google.com/uc?id={EPSTEIN_ONLY_FILE_ID}"
    print("Downloading epstein_only.json from Google Drive...")
    gdown.download(url, dest, quiet=False)
    print(f"Downloaded to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Index into ChromaDB
# ---------------------------------------------------------------------------

def index_to_chroma(
    chunked: Dict[str, List[Dict]],
    collection_name: str,
    embedding_model: str = "text-embedding-3-small",
) -> None:
    """Index pre-chunked data into ChromaDB Cloud with BM25 sparse + dense vectors."""
    chroma_client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        database=os.getenv("CHROMA_DATABASE"),
    )

    # Validate collection doesn't exist
    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing:
        raise ValueError(f"Collection '{collection_name}' already exists. Use a unique collection name.")

    # Create schema with BM25 sparse + dense indexes
    schema = Schema()

    sparse_ef = Bm25EmbeddingFunction(query_config={"task": "document"})
    schema.create_index(
        config=SparseVectorIndexConfig(
            source_key=K.DOCUMENT,
            embedding_function=sparse_ef,
            bm25=True,
        ),
        key="bm25_vector",
    )

    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=embedding_model,
    )
    schema.create_index(config=VectorIndexConfig(
        space="cosine",
        embedding_function=embedding_function,
    ))

    chroma_collection = chroma_client.create_collection(
        name=collection_name,
        schema=schema,
    )
    print(f"Collection '{collection_name}' created")

    # Flatten chunks
    ids, documents, metadatas = [], [], []
    for tid_str, chunks in chunked.items():
        for i, chunk in enumerate(chunks):
            chunk_id = f"{tid_str}_{i}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({"source": tid_str, "dataset": "epstein"})

    print(f"Indexing {len(ids)} chunks...")

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = embed_in_batches(openai_client, documents, model=embedding_model)
    sparse_embeddings = create_bm25_vectors(documents)
    metadatas_with_bm25 = [{**m, "bm25_vector": s} for m, s in zip(metadatas, sparse_embeddings)]

    add_to_chroma_with_retry(chroma_collection, ids, documents, embeddings, metadatas_with_bm25)
    print("Indexing complete!")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_index(
    collection: str,
    output_dir: str,
    input_json: str = None,
    embedding_model: str = "text-embedding-3-small",
) -> dict:
    """Run the indexing pipeline: download → index.

    Args:
        collection: ChromaDB collection name.
        output_dir: Where to write downloaded data and stats.
        input_json: Path to epstein_only.json. If None, downloads from Google Drive.
        embedding_model: OpenAI embedding model name.

    Returns stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    errors: List[str] = []

    # Download or use provided input
    if input_json is None:
        input_json = download_data(output_dir)
    else:
        if not os.path.exists(input_json):
            raise FileNotFoundError(f"Input file not found: {input_json}")

    # Load pre-chunked threads
    with open(input_json) as f:
        chunked: Dict[str, List[Dict]] = json.load(f)
    print(f"Loaded {len(chunked)} threads, "
          f"{sum(len(v) for v in chunked.values())} chunks")

    # The downloaded file is already in the right format for utils.py (get_thread)
    chunks_path = input_json

    num_threads = len(chunked)
    num_chunks = sum(len(v) for v in chunked.values())

    # Index
    try:
        index_to_chroma(chunked, collection, embedding_model)
    except Exception as e:
        errors.append(str(e))
        print(f"Error during indexing: {e}")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        stats = {
            "num_threads": num_threads,
            "num_chunks": num_chunks,
            "collection": collection,
            "chunks_path": chunks_path,
            "errors": errors,
        }
        with open(os.path.join(output_dir, f"{timestamp}.json"), "w") as f:
            json.dump(stats, f, indent=4)
        raise

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    stats = {
        "num_threads": num_threads,
        "num_chunks": num_chunks,
        "collection": collection,
        "chunks_path": chunks_path,
        "errors": errors,
    }
    with open(os.path.join(output_dir, f"{timestamp}.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Stats written to {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Index Epstein email data into ChromaDB (download → index)."
    )
    parser.add_argument("--input", "-i", default=None,
                        help="Path to epstein_only.json (default: download from Google Drive)")
    parser.add_argument("--collection", "-c", required=True,
                        help="ChromaDB collection name")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for data and stats (default: data/epstein)")
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                        help="OpenAI embedding model (default: text-embedding-3-small)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join("data", "epstein")

    run_index(
        args.collection,
        output_dir=output_dir,
        input_json=args.input,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
