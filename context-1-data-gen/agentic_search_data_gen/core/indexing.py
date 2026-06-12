"""Shared chunking, embedding, and indexing utilities."""
import time
from typing import List, Dict, Any

import tiktoken
from openai import OpenAI

TOKENS_PER_CHUNK = 512
CHROMA_BYTE_LIMIT = 16384
BUFFER = 100

_encoding = tiktoken.get_encoding("cl100k_base")


def get_token_count(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_encoding.encode(text))


def recursive_chunk(content: str, tokens_per_chunk: int = TOKENS_PER_CHUNK, byte_limit: int = CHROMA_BYTE_LIMIT) -> List[str]:
    """Recursively split content into chunks that fit within token and byte limits."""
    chunk_bytes = len(content.encode('utf-8'))
    chunk_tokens = get_token_count(content)

    if chunk_bytes <= byte_limit and chunk_tokens <= tokens_per_chunk + BUFFER:
        return [content]

    tokens = _encoding.encode(content)
    mid = len(tokens) // 2

    first_half = _encoding.decode(tokens[:mid])
    second_half = _encoding.decode(tokens[mid:])

    chunks = []
    chunks.extend(recursive_chunk(first_half, tokens_per_chunk, byte_limit))
    chunks.extend(recursive_chunk(second_half, tokens_per_chunk, byte_limit))

    return chunks


def embed_in_batches(openai_client: OpenAI, texts: List[str], model: str = "text-embedding-3-small", max_tokens_per_batch: int = 200_000) -> List[List[float]]:
    """Embed texts in batches, respecting the OpenAI token limit."""
    all_embeddings = []
    current_batch = []
    current_tokens = 0
    batch_count = 0

    for text in texts:
        token_count = get_token_count(text)

        would_exceed_tokens = current_tokens + token_count > max_tokens_per_batch

        if current_batch and would_exceed_tokens:
            batch_count += 1
            print(f"  OpenAI batch {batch_count}: {len(current_batch)} items, {current_tokens} tokens")
            batch_embeddings = [
                response.embedding
                for response in openai_client.embeddings.create(model=model, input=current_batch).data
            ]
            all_embeddings.extend(batch_embeddings)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += token_count

    if current_batch:
        batch_count += 1
        print(f"  OpenAI batch {batch_count}: {len(current_batch)} items, {current_tokens} tokens")
        batch_embeddings = [
            response.embedding
            for response in openai_client.embeddings.create(model=model, input=current_batch).data
        ]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def add_to_chroma_with_retry(collection, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], batch_size: int = 200, max_retries: int = 5):
    """Add documents to a ChromaDB collection in batches with retry logic."""
    for i in range(0, len(ids), batch_size):
        ids_batch = ids[i:i+batch_size]
        docs_batch = documents[i:i+batch_size]
        embeds_batch = embeddings[i:i+batch_size]
        metas_batch = metadatas[i:i+batch_size]

        for attempt in range(max_retries):
            try:
                collection.add(
                    ids=ids_batch,
                    documents=docs_batch,
                    embeddings=embeds_batch,
                    metadatas=metas_batch
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"  Chroma add failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        print(f"Uploaded batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")


def create_bm25_vectors(documents: List[str]) -> List[Dict[str, Any]]:
    """Create BM25 sparse vectors for a list of documents.

    Requires fastembed to be installed (pip install fastembed).
    """
    from fastembed.sparse.bm25 import Bm25
    from chromadb.utils.sparse_embedding_utils import normalize_sparse_vector

    bm25_model = Bm25(model_name="Qdrant/bm25")
    sparse_embeddings = [
        normalize_sparse_vector(indices=v.indices.tolist(), values=v.values.tolist())
        for v in bm25_model.embed(documents)
    ]
    return sparse_embeddings
