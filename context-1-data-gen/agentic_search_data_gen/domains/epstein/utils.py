import json
import random
import chromadb
import os
from chromadb import *
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from chromadb.utils.embedding_functions import Bm25EmbeddingFunction

from ...core.utils import get_anthropic_client

load_dotenv()

# ---------------------------------------------------------------------------
# Lazy-init private state
# ---------------------------------------------------------------------------
_cloud_collection = None
_local_corpus = None
_openai_client = None
_bm25_ef = None

_collection_name = None
_corpus_path = None


def init_utils(collection_name: str = None, corpus_path: str = None) -> None:
    """Configure collection and corpus for this session.

    Called by __main__.py after indexing to point at the freshly-created
    collection and chunks file.  When not called, defaults are used for
    backwards compatibility.
    """
    global _collection_name, _corpus_path
    # Reset cached objects so they get re-created with new config
    global _cloud_collection, _local_corpus
    _cloud_collection = None
    _local_corpus = None
    _collection_name = collection_name
    _corpus_path = corpus_path


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_bm25_ef():
    global _bm25_ef
    if _bm25_ef is None:
        _bm25_ef = Bm25EmbeddingFunction(avg_len=4000, task="query")
    return _bm25_ef


def _get_collection():
    global _cloud_collection
    if _cloud_collection is None:
        chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            database=os.getenv("CHROMA_DATABASE"),
        )
        name = _collection_name or "epstein_only_1_8"
        _cloud_collection = chroma_client.get_collection(name=name)
    return _cloud_collection


def _get_corpus() -> dict:
    global _local_corpus
    if _local_corpus is None:
        path = _corpus_path or "../data/epstein/epstein_chunks.json"
        with open(path) as f:
            _local_corpus = json.load(f)
    return _local_corpus


def embed(texts: List[str]) -> List[List[float]]:
    client = _get_openai_client()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in resp.data]

def get_random_seed_threads():
    corpus = _get_corpus()
    thread_ids = random.sample(list(corpus.keys()), 5)
    thread_texts = [get_thread(thread_id) for thread_id in thread_ids]

    formatted_text = ""

    for thread_id, thread_text in zip(thread_ids, thread_texts):
        formatted_text += f"Thread ID: {thread_id}\n\n{thread_text}\n\n--------------------------------\n\n"

    return thread_texts

def get_data_for_person(person: str):
    collection = _get_collection()
    search = Search().where(
        K.DOCUMENT.regex(rf"(?i){person}")
    ).select(K.DOCUMENT)

    res = collection.search(search)

    return [(id, document) for id, document in zip(res['ids'][0], res['documents'][0])]

def get_random_across_person(person: str):
    results = get_data_for_person(person)

    random_results = random.sample(results, min(5, len(results)))

    formatted_text = ""
    for id, document in random_results:
        formatted_text += f"Thread ID: {id.split('_')[0]} | Chunk ID: {id}\n\n{document}\n\n--------------------------------\n\n"

    return formatted_text

def hybrid_search_across_all(query: str, k: int = 5) -> str:
    """Hybrid search (BM25 + semantic) across entire corpus using RRF."""
    bm25_ef = _get_bm25_ef()
    sparse_vector = bm25_ef([query])[0]
    dense_vector = embed([query])[0]

    collection = _get_collection()
    search = (
        Search()
        .rank(Rrf([
            Knn(key="bm25_vector", query=sparse_vector, return_rank=True, limit=k * 4, default=10.0),
            Knn(key="#embedding", query=dense_vector, return_rank=True, limit=k * 4, default=10.0),
        ]))
        .select(K.DOCUMENT)
        .limit(k)
    )

    res = collection.search(search)
    ids = res["ids"][0]
    documents = res["documents"][0]

    formatted = ""
    for id, doc in zip(ids, documents):
        thread_id = id.split("_")[0]
        formatted += f"Thread ID: {thread_id} | Chunk ID: {id}\n\n{doc}\n\n--------------------------------\n\n"

    return formatted if formatted else "No results found"

def grep_across_all(pattern: str, k: int = 5) -> str:
    """Regex search across entire corpus."""
    collection = _get_collection()
    search = (
        Search()
        .where(K.DOCUMENT.regex(rf"(?i){pattern}"))
        .select(K.DOCUMENT)
        .limit(k)
    )

    res = collection.search(search)
    ids = res["ids"][0]
    documents = res["documents"][0]

    formatted = ""
    for id, doc in zip(ids, documents):
        thread_id = id.split("_")[0]
        formatted += f"Thread ID: {thread_id} | Chunk ID: {id}\n\n{doc}\n\n--------------------------------\n\n"

    return formatted if formatted else "No results found"

def search_across_person(person: str, query: str, collection_name: str, k: int = 5) -> str:
    """Search within a person's emails using local collection."""
    results = get_data_for_person(person)

    if not results:
        return f"No emails found for {person}"

    total_chars = sum(len(doc) for _, doc in results)
    if total_chars < 40000:  # ~10k tokens
        formatted = ""
        for id, doc in results:
            thread_id = id.split("_")[0]
            formatted += f"Thread ID: {thread_id} | Chunk ID: {id}\n\n{doc}\n\n--------------------------------\n\n"
        return formatted

    local_client = chromadb.Client()
    local_collection = local_client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    try:
        ids = [r[0] for r in results]
        docs = [r[1] for r in results]
        embeddings = embed(docs)

        local_collection.add(ids=ids, documents=docs, embeddings=embeddings)

        query_embedding = embed([query])[0]
        search_results = local_collection.query(query_embeddings=[query_embedding], n_results=k)

        formatted = ""
        for id, doc in zip(search_results["ids"][0], search_results["documents"][0]):
            thread_id = id.split("_")[0]
            formatted += f"Thread ID: {thread_id} | Chunk ID: {id}\n\n{doc}\n\n--------------------------------\n\n"

        return formatted if formatted else "No results found"
    finally:
        local_client.delete_collection(name=collection_name)

def get_thread(thread_id: str):
    corpus = _get_corpus()
    thread = corpus[thread_id]
    formatted_thread = "".join([email['text'] for email in thread])

    return formatted_thread
