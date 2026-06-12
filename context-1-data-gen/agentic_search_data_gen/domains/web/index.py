"""Index web page data into ChromaDB with BM25 sparse + OpenAI dense embeddings."""
import os
import json
import glob
import argparse
import random
from datetime import datetime, timezone

from chromadb import *
import chromadb
from chromadb.utils.embedding_functions import *
from openai import OpenAI
from dotenv import load_dotenv

from ...core.indexing import recursive_chunk, embed_in_batches, add_to_chroma_with_retry, create_bm25_vectors

load_dotenv()


def run_index(input_dir: str, collection: str) -> dict:
    """Index web page chunks into ChromaDB. Returns stats dict."""
    output_dir = os.path.join(input_dir, "index_output")
    os.makedirs(output_dir, exist_ok=True)

    # Auto-discover JSON files, excluding index_output subdirectory
    all_json = glob.glob(os.path.join(input_dir, "*.json"))
    filepaths = [f for f in all_json if not f.startswith(os.path.join(input_dir, "index_output"))]
    print(f"Discovered {len(filepaths)} JSON files in {input_dir}")

    # Auto-extract distractor URLs to exclude from filtered_distractors
    distractor_urls_to_exclude = set()
    for filepath in filepaths:
        with open(filepath) as f:
            data = json.load(f)
        for task in data.get("tasks", []):
            for fd in task.get("filtered_distractors", []):
                if "id" in fd:
                    distractor_urls_to_exclude.add(fd["id"])
    print(f"Excluding {len(distractor_urls_to_exclude)} filtered distractor URLs")

    urls_and_contents = {}

    for filepath in filepaths:
        with open(filepath) as f:
            data = json.load(f)

            if "tasks" not in data:
                continue

            for task in data["tasks"]:
                for k, v in task["items_and_contents"].items():
                    if k in urls_and_contents:
                        if v.startswith("[Page truncated"):
                            clean_v = v.split("tokens]\n")[1]
                            existing_content = urls_and_contents[k]
                            combined_content = existing_content + "\n" + clean_v
                            urls_and_contents[k] = combined_content
                    else:
                        clean_v = v
                        if v.startswith("[Page truncated"):
                            clean_v = v.split("tokens]\n")[1]
                        urls_and_contents[k] = clean_v

                if "distractors_and_contents" not in task:
                    continue

                for k, v in task["distractors_and_contents"].items():
                    if k in distractor_urls_to_exclude:
                        continue

                    if k in urls_and_contents:
                        if v.startswith("[Page truncated"):
                            clean_v = v.split("tokens]\n")[1]
                            existing_content = urls_and_contents[k]
                            combined_content = existing_content + "\n" + clean_v
                            urls_and_contents[k] = combined_content
                    else:
                        clean_v = v
                        if v.startswith("[Page truncated"):
                            clean_v = v.split("tokens]\n")[1]
                        urls_and_contents[k] = clean_v

    print(len(urls_and_contents), "unique pages")

    urls_list = list(urls_and_contents.items())
    random.shuffle(urls_list)
    urls_and_contents = dict(urls_list)

    print("randomly shuffled")

    current_id = 0
    url_to_id = {}
    id_to_chunks = {}

    for url, content in urls_and_contents.items():
        url_to_id[url] = str(current_id)
        chunks = recursive_chunk(content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{current_id}_{i}"
            id_to_chunks[chunk_id] = chunk

        current_id += 1

    num_pages = len(url_to_id)
    num_chunks = len(id_to_chunks)
    errors = []

    try:
        chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            database=os.getenv("CHROMA_DATABASE")
        )

        # Ensure collection doesn't already exist
        existing = [c.name for c in chroma_client.list_collections()]
        if collection in existing:
            raise ValueError(f"Collection '{collection}' already exists. Use a unique collection name.")

        schema = Schema()

        sparse_ef = Bm25EmbeddingFunction(query_config={'task': 'document'})

        schema.create_index(
            config=SparseVectorIndexConfig(
                source_key=K.DOCUMENT,
                embedding_function=sparse_ef,
                bm25=True
            ),
            key="bm25_vector"
        )

        embedding_function = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        schema.create_index(config=VectorIndexConfig(
            space="cosine",
            embedding_function=embedding_function
        ))

        chroma_collection = chroma_client.create_collection(
            name=collection,
            schema=schema
        )
        print(f"Collection {collection} created")

        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        ids, documents, metadatas = [], [], []
        for chunk_id, chunk in id_to_chunks.items():
            ids.append(chunk_id)
            documents.append(chunk)
            source = chunk_id.split("_")[0]
            metadatas.append({"source": source})

        print(f"Indexing {len(ids)} chunks...")

        embeddings = embed_in_batches(openai_client, documents)
        sparse_embeddings = create_bm25_vectors(documents)
        metadatas_with_bm25 = [{**m, "bm25_vector": s} for m, s in zip(metadatas, sparse_embeddings)]

        add_to_chroma_with_retry(chroma_collection, ids, documents, embeddings, metadatas_with_bm25)

        print("Done!")
    except Exception as e:
        errors.append(str(e))
        print(f"Error during indexing: {e}")

        # Write stats before re-raising
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        stats = {
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "collection": collection,
            "errors": errors,
        }
        with open(os.path.join(output_dir, f"{timestamp}.json"), "w") as f:
            json.dump(stats, f, indent=4)

        raise

    with open(os.path.join(output_dir, "url_to_id.json"), "w") as f:
        json.dump(url_to_id, f, indent=4)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    stats = {
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "collection": collection,
        "errors": errors,
    }
    with open(os.path.join(output_dir, f"{timestamp}.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Output written to {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", required=True,
                        help="Directory containing task JSON files (e.g., data/web/output)")
    parser.add_argument("--collection", "-c", required=True,
                        help="ChromaDB collection name")
    args = parser.parse_args()

    run_index(args.input_dir, args.collection)


if __name__ == "__main__":
    main()
