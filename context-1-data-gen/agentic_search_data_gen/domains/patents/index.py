"""Index patent data into ChromaDB with BM25 sparse + OpenAI dense embeddings."""
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


def run_index(input_dir: str, collection: str, embedding_model: str = "text-embedding-3-small") -> dict:
    """Index patent chunks into ChromaDB. Returns stats dict."""
    output_dir = os.path.join(input_dir, "index_output")
    os.makedirs(output_dir, exist_ok=True)

    # Auto-discover JSON files, excluding errors.json and index_output subdirectory
    all_json = glob.glob(os.path.join(input_dir, "*.json"))
    filepaths = [
        f for f in all_json
        if not f.startswith(os.path.join(input_dir, "index_output"))
        and os.path.basename(f) != "errors.json"
    ]
    print(f"Discovered {len(filepaths)} JSON files in {input_dir}")

    app_no_to_distractors = {}
    app_no_to_contents = {}

    for filepath in filepaths:
        with open(filepath) as f:
            data = json.load(f)

            if "application_number" in data:
                app_no = data["application_number"]
                app_no_to_distractors[app_no] = {}

                # references
                if "892" in data and "references" in data['892']:
                    for k, v in data['892']['references'].items():
                        contents = ""
                        ref_app_no = v['application_number']
                        if "abstract" in v:
                            contents += v['abstract']
                        if "description" in v:
                            contents += v['description']
                        if "claims" in v:
                            if isinstance(v['claims'], list):
                                contents += "\n".join(v['claims'])
                            else:
                                contents += v['claims']

                        app_no_to_contents[ref_app_no] = contents

                # similar patents
                if "similar_patents" in data:
                    app_no_to_distractors[app_no]["similar_patents"] = []
                    for k, v in data['similar_patents'].items():
                        app_no_to_distractors[app_no]["similar_patents"].append(v['application_number'])
                        contents = ""
                        sim_app_no = v['application_number']
                        if "abstract" in v:
                            contents += v['abstract']
                        if "description" in v:
                            contents += v['description']
                        if "claims" in v:
                            if isinstance(v['claims'], list):
                                contents += "\n".join(v['claims'])
                            else:
                                contents += v['claims']

                        app_no_to_contents[sim_app_no] = contents

    print(len(app_no_to_contents), "unique patents")

    app_no_list = list(app_no_to_contents.items())
    random.shuffle(app_no_list)
    app_no_to_contents = dict(app_no_list)

    print("randomly shuffled")

    id_to_chunks = {}

    for app_no, content in app_no_to_contents.items():
        if not content or not content.strip():
            continue
        chunks = recursive_chunk(content)

        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                continue
            chunk_id = f"{app_no}_{i}"
            id_to_chunks[chunk_id] = chunk

    num_patents = len(app_no_to_contents)
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
            model_name=embedding_model
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
            "num_patents": num_patents,
            "num_chunks": num_chunks,
            "collection": collection,
            "errors": errors,
        }
        with open(os.path.join(output_dir, f"{timestamp}.json"), "w") as f:
            json.dump(stats, f, indent=4)

        raise

    with open(os.path.join(output_dir, "app_no_to_distractors.json"), "w") as f:
        json.dump(app_no_to_distractors, f, indent=4)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    stats = {
        "num_patents": num_patents,
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
                        help="Directory containing patent JSON files (e.g., data/patents/output)")
    parser.add_argument("--collection", "-c", required=True,
                        help="ChromaDB collection name")
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                        help="OpenAI embedding model (default: text-embedding-3-small)")
    args = parser.parse_args()

    run_index(args.input_dir, args.collection, embedding_model=args.embedding_model)


if __name__ == "__main__":
    main()
