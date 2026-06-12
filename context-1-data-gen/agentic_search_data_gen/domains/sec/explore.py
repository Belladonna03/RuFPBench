import os
import argparse
import json
import random
import time
from typing import Any, Callable, Dict, List, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import chromadb
from chromadb.utils.embedding_functions import Bm25EmbeddingFunction
from dotenv import load_dotenv

from .prompts import EXPLORATION_INSTRUCTION, EXPLORATION_PROMPT, TRUTH_TYPES, SEC_FORCE_OUTPUT_MESSAGE
from ...core.utils import get_anthropic_client, count_tokens, parse_tag, get_embedding_client
from ...core.rerank import BasetenReranker
from .utils import parse_supporting_items, format_chunks

load_dotenv()

# BM25 embedding function for hybrid search
bm25_ef = Bm25EmbeddingFunction(avg_len=4000, task="query")


# ============================================================================
# Tool Definitions
# ============================================================================

EXPLORE_TOOLS = [
    {
        "name": "search_in_filing",
        "description": "Search across a specific filing. Returns ~10 most relevant chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "accession_no": {
                    "type": "string",
                    "description": "The Accession No. of the filing to search in (i.e. 0001736297-25-000145)"
                },
                "query": {
                    "type": "string",
                    "description": "Query to search, a simple phrase is sufficient"
                }
            },
            "required": ["accession_no", "query"]
        }
    },
    {
        "name": "search_in_company",
        "description": "Search across ALL filings for this company. Returns ~10 most relevant chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to search, a simple phrase is sufficient"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "random_in_filing",
        "description": "Get 5 random chunks from a specific filing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "accession_no": {
                    "type": "string",
                    "description": "The Accession No. of the filing (i.e. 0001736297-25-000145)"
                }
            },
            "required": ["accession_no"]
        }
    },
    {
        "name": "random_in_company",
        "description": "Get 5 random chunks from across all filings for this company.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_full_filing",
        "description": "Get the full text of a specific filing. Only works for short filings. Will suggest using search_in_filing for long filings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "accession_no": {
                    "type": "string",
                    "description": "The Accession No. of the filing (i.e. 0001736297-25-000145)"
                }
            },
            "required": ["accession_no"]
        }
    },
]

# Cross-collection tools (only for bridge.py)
CROSS_COLLECTION_TOOLS = [
    {
        "name": "hybrid_search_across_all",
        "description": "Hybrid search (BM25 + semantic) across ALL filings in the entire SEC collection, not just this company. Use this to find information that may exist in filings from other companies. Returns ~10 most relevant chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to search across all SEC filings, a simple phrase is sufficient"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "grep_across_all",
        "description": "Regex pattern search across ALL filings in the entire SEC collection, not just this company. Use this to find exact matches or patterns that may exist in filings from other companies. Returns ~10 matching chunks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for (case-insensitive)"
                }
            },
            "required": ["pattern"]
        }
    },
]

# Full tools list (for bridge.py)
TOOLS = EXPLORE_TOOLS + CROSS_COLLECTION_TOOLS

# Form types that are information-rich (annual reports)
INFO_RICH_FORMS = ["10-K", "20-F"]


# ============================================================================
# Company Search Engine
# ============================================================================

class CompanySearchEngine:
    def __init__(
        self,
        collection_name: str = "sec_filings",
        ticker: str = None,
        chroma_cloud_client: chromadb.CloudClient = None,
        openai_client = None
    ):
        # Use shared clients if provided, otherwise create new ones
        if chroma_cloud_client is None:
            self.chroma_cloud_client = chromadb.CloudClient(
                api_key=os.getenv("CHROMA_API_KEY"),
                database=os.getenv("CHROMA_DATABASE")
            )
        else:
            self.chroma_cloud_client = chroma_cloud_client

        self.chroma_local_client = chromadb.Client()
        self.ticker = ticker
        self.cloud_collection = self.chroma_cloud_client.get_collection(collection_name)

        if openai_client is None:
            self.openai_client = get_embedding_client()
        else:
            self.openai_client = openai_client

        self.dense_embedding_model = "text-embedding-3-small"

        self.dense_embedding_collection = None
        self.all_chunks = self.get_all_chunks()
        self.create_local_collections(self.all_chunks)

    def get_all_chunks(self, max_retries: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        search = (
            chromadb.Search()
            .where(chromadb.Key("ticker") == self.ticker)
            .select(chromadb.Key.DOCUMENT, chromadb.Key.METADATA)
        )

        for attempt in range(max_retries):
            try:
                res = self.cloud_collection.search(search)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"ChromaDB connection error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait_time)

        all_chunks = {}

        for id, document, metadata in zip(res['ids'][0], res['documents'][0], res['metadatas'][0]):
            if metadata['source'] not in all_chunks:
                all_chunks[metadata['source']] = []
            all_chunks[metadata['source']].append({
                "id": id,
                "document": document,
                "accession_no": metadata['source']
            })

        return all_chunks

    def openai_embed(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        for attempt in range(max_retries):
            try:
                return [response.embedding for response in self.openai_client.embeddings.create(model=self.dense_embedding_model, input=texts).data]
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error embedding after {max_retries} retries: {e}")
                    raise
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                print(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

    def openai_embed_in_batches(self, texts: List[str], batch_size: int = 200) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.openai_embed(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def collection_add_in_batches(
        self,
        collection: Any,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]] | None = None
    ) -> None:
        import multiprocessing

        BATCH_SIZE = 100
        LEN = len(embeddings)
        N_THREADS = min(os.cpu_count() or multiprocessing.cpu_count(), 20)

        def add_batch(start: int, end: int) -> None:
            id_batch = ids[start:end]
            doc_batch = texts[start:end]
            try:
                if metadatas:
                    collection.add(ids=id_batch, documents=doc_batch, embeddings=embeddings[start:end], metadatas=metadatas[start:end])
                else:
                    collection.add(ids=id_batch, documents=doc_batch, embeddings=embeddings[start:end])
            except Exception as e:
                print(f"Error adding {start} to {end}")
                print(e)

        threadpool = ThreadPoolExecutor(max_workers=N_THREADS)

        for i in range(0, LEN, BATCH_SIZE):
            threadpool.submit(add_batch, i, min(i + BATCH_SIZE, LEN))

        threadpool.shutdown(wait=True)

    def create_local_collections(self, all_chunks: Dict[str, List[Dict[str, Any]]]) -> None:
        self.dense_embedding_collection = self.chroma_local_client.get_or_create_collection(name=f"ticker_{self.ticker}", metadata={"hnsw:space": "cosine"})
        if self.dense_embedding_collection.count() == 0:
            all_ids = []
            all_documents = []
            metadatas = []

            for accession_no, chunks in all_chunks.items():
                for chunk in chunks:
                    all_ids.append(chunk['id'])
                    all_documents.append(chunk['document'])
                    metadatas.append({
                        "accession_no": accession_no
                    })

            dense_embeddings = self.openai_embed_in_batches(all_documents)
            self.collection_add_in_batches(self.dense_embedding_collection, all_ids, all_documents, dense_embeddings, metadatas)

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        dense_query_embedding = self.openai_embed([query])[0]

        dense_results = self.dense_embedding_collection.query(
            query_embeddings=[dense_query_embedding],
            n_results=k
        )

        dense_results = [
            {
                "id": id,
                "document": document,
                "accession_no": metadata["accession_no"]
            }
            for id, document, metadata in zip(
                dense_results["ids"][0],
                dense_results["documents"][0],
                dense_results["metadatas"][0]
            )
        ]

        return dense_results


# ============================================================================
# Tool Executor
# ============================================================================

def random_from_chunks(chunks: List[Dict[str, Any]], num_chunks: int) -> List[Dict[str, Any]]:
    """Randomly sample chunks from a list."""
    if len(chunks) > num_chunks:
        return random.sample(chunks, num_chunks)
    return chunks


def random_chunks_for_init(
    available_forms: Dict[str, List[str]],
    all_chunks: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Get random chunks for agent initialization.

    Prioritizes info-rich forms (10-K, 20-F) and samples from other form types.
    Skips non-key forms that don't exist in all_chunks.
    """
    available_form_types = list(available_forms.keys())
    info_rich_form_types = [form_type for form_type in available_form_types if form_type in INFO_RICH_FORMS]
    other_form_types = [form_type for form_type in available_form_types if form_type not in INFO_RICH_FORMS]

    random_chunks = []

    if info_rich_form_types:
        info_rich_accession_no = available_forms[info_rich_form_types[0]][0]
        random_chunks.extend(random_from_chunks(all_chunks[info_rich_accession_no], 5))

    if len(other_form_types) > 3:
        random_other_form_types = random.sample(other_form_types, 3)
        for form_type in random_other_form_types:
            random_accession_no = random.choice(available_forms[form_type])
            if random_accession_no in all_chunks:
                random_chunks.extend(random_from_chunks(all_chunks[random_accession_no], 3))
    else:
        for form_type in other_form_types:
            ids = available_forms[form_type]
            random_form_id = random.choice(ids)
            if random_form_id in all_chunks:
                random_chunks.extend(random_from_chunks(all_chunks[random_form_id], 3))

    return random_chunks


class SECToolExecutor:
    """Handles tool execution for SEC filings exploration."""

    def __init__(
        self,
        collection_name: str,
        ticker: str,
        chroma_client: chromadb.CloudClient = None,
        openai_client = None,
        reranker: BasetenReranker = None
    ):
        self.collection_name = collection_name
        self.ticker = ticker
        self.token_counter = count_tokens

        # Use shared clients if provided, otherwise create new ones
        if chroma_client is None:
            self.chroma_client = chromadb.CloudClient(
                api_key=os.getenv("CHROMA_API_KEY"),
                database=os.getenv("CHROMA_DATABASE")
            )
        else:
            self.chroma_client = chroma_client

        self.collection = self.chroma_client.get_collection(collection_name)

        if openai_client is None:
            self.openai_client = get_embedding_client()
        else:
            self.openai_client = openai_client

        if reranker is None:
            self.reranker = BasetenReranker(
                token_counter=self.token_counter,
                max_tokens=5500,
                batch_size=100,
                max_concurrent_requests=32,
            )
        else:
            self.reranker = reranker

        self.company_search_engine = CompanySearchEngine(
            collection_name=collection_name,
            ticker=ticker,
            chroma_cloud_client=self.chroma_client,
            openai_client=self.openai_client
        )

        self.all_chunks = self._get_all_chunks()
        self.seen_chunk_ids: Set[str] = set()

    def _get_all_chunks(self, max_retries: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all chunks for the ticker from ChromaDB with retry logic."""
        search = (
            chromadb.Search()
            .where(chromadb.Key("ticker") == self.ticker)
            .select(chromadb.Key.DOCUMENT, chromadb.Key.METADATA)
        )

        for attempt in range(max_retries):
            try:
                res = self.collection.search(search)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"ChromaDB connection error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait_time)

        all_chunks = {}

        for id, document, metadata in zip(res['ids'][0], res['documents'][0], res['metadatas'][0]):
            if metadata['source'] not in all_chunks:
                all_chunks[metadata['source']] = []
            all_chunks[metadata['source']].append({
                "id": id,
                "document": document,
                "accession_no": metadata['source']
            })

        return all_chunks

    def search(self, chunks: List[Dict[str, Any]], query: str) -> str:
        """Search and rerank chunks based on query."""
        filtered_chunks = [chunk for chunk in chunks if chunk["id"] not in self.seen_chunk_ids]
        ids = [chunk["id"] for chunk in filtered_chunks]
        docs = [chunk["document"] for chunk in filtered_chunks]

        max_retries = 5
        for attempt in range(max_retries):
            try:
                results = self.reranker(query=query, documents=docs)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        res_chunks = []

        for res in results:
            original_chunk = filtered_chunks[res.original_index]
            res_chunks.append({
                "id": original_chunk["id"],
                "document": res.document,
                "accession_no": original_chunk["accession_no"]
            })
            self.seen_chunk_ids.add(original_chunk["id"])

        return format_chunks(res_chunks)

    def _embed(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Embed texts using OpenAI with retry and exponential backoff."""
        for attempt in range(max_retries):
            try:
                resp = self.openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
                return [e.embedding for e in resp.data]
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error embedding after {max_retries} retries: {e}")
                    raise
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                print(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

    def hybrid_search_across_all(self, query: str, k: int = 10) -> str:
        """Hybrid search (BM25 + semantic) across entire SEC collection using RRF."""
        sparse_vector = bm25_ef([query])[0]
        dense_vector = self._embed([query])[0]

        search = (
            chromadb.Search()
            .rank(chromadb.Rrf([
                chromadb.Knn(key="bm25_vector", query=sparse_vector, return_rank=True, limit=k * 4, default=10.0),
                chromadb.Knn(key="#embedding", query=dense_vector, return_rank=True, limit=k * 4, default=10.0),
            ]))
            .select(chromadb.Key.DOCUMENT, chromadb.Key.METADATA)
            .limit(k)
        )

        res = self.collection.search(search)
        ids = res["ids"][0]
        documents = res["documents"][0]
        metadatas = res["metadatas"][0]

        # Filter out already seen chunks
        result_chunks = []
        for id, doc, meta in zip(ids, documents, metadatas):
            if id not in self.seen_chunk_ids:
                result_chunks.append({
                    "id": id,
                    "document": doc,
                    "accession_no": meta.get("source", "unknown")
                })
                self.seen_chunk_ids.add(id)

        return format_chunks(result_chunks) if result_chunks else "No results found"

    def grep_across_all(self, pattern: str, k: int = 10) -> str:
        """Regex search across entire SEC collection."""
        search = (
            chromadb.Search()
            .where(chromadb.Key.DOCUMENT.regex(rf"(?i){pattern}"))
            .select(chromadb.Key.DOCUMENT, chromadb.Key.METADATA)
            .limit(k)
        )

        res = self.collection.search(search)
        ids = res["ids"][0]
        documents = res["documents"][0]
        metadatas = res["metadatas"][0]

        # Filter out already seen chunks
        result_chunks = []
        for id, doc, meta in zip(ids, documents, metadatas):
            if id not in self.seen_chunk_ids:
                result_chunks.append({
                    "id": id,
                    "document": doc,
                    "accession_no": meta.get("source", "unknown")
                })
                self.seen_chunk_ids.add(id)

        return format_chunks(result_chunks) if result_chunks else "No results found"

    def execute(self, tool_name: str, tool_args: Dict[str, Any], iteration: int) -> str:
        """Execute a tool and return the result."""
        result = f"Iteration {iteration+1}\n\n"

        if tool_name == "search_in_filing":
            accession_no = tool_args.get("accession_no")
            if accession_no not in self.all_chunks:
                result += f"Error: Filing {accession_no} not found"
            else:
                filing_chunks = self.all_chunks[accession_no]
                result += self.search(filing_chunks, tool_args["query"])

        elif tool_name == "search_in_company":
            company_search_results = self.company_search_engine.search(tool_args["query"])
            result += self.search(company_search_results, tool_args["query"])

        elif tool_name == "random_in_filing":
            accession_no = tool_args.get("accession_no")
            if accession_no not in self.all_chunks:
                result += f"Error: Filing {accession_no} not found"
            else:
                filing_chunks = self.all_chunks[accession_no]
                filtered_chunks = [c for c in filing_chunks if c["id"] not in self.seen_chunk_ids]
                random_chunks = random_from_chunks(filtered_chunks, 5)
                self.seen_chunk_ids.update([chunk["id"] for chunk in random_chunks])
                result += format_chunks(random_chunks)

        elif tool_name == "random_in_company":
            filtered_chunks = []
            for k, v in self.all_chunks.items():
                for chunk in v:
                    if chunk["id"] not in self.seen_chunk_ids:
                        filtered_chunks.append(chunk)

            random_chunks = random_from_chunks(filtered_chunks, 5)
            self.seen_chunk_ids.update([chunk["id"] for chunk in random_chunks])
            result += format_chunks(random_chunks)

        elif tool_name == "get_full_filing":
            accession_no = tool_args.get("accession_no")
            if accession_no not in self.all_chunks:
                result += f"Error: Filing {accession_no} not found"
            else:
                full_filing = format_chunks(self.all_chunks[accession_no])
                if self.token_counter(full_filing) > 10000:
                    result += "Filing is too long to get the full text. Use the search_in_filing tool to get chunks instead."
                else:
                    result += full_filing

        elif tool_name == "hybrid_search_across_all":
            result += self.hybrid_search_across_all(tool_args["query"])

        elif tool_name == "grep_across_all":
            result += self.grep_across_all(tool_args["pattern"])

        else:
            result += f"Unknown tool: {tool_name}"

        return result


# ============================================================================
# Agent Loop Functions
# ============================================================================

def run_agent_loop(
    client,
    input_messages: List[Dict[str, Any]],
    trajectory: List[Dict[str, Any]],
    tool_executor: SECToolExecutor,
    max_iterations: int,
    parse_fn: Callable[[str], Dict[str, Any]],
    system_prompt: str,
    tools: List[Dict[str, Any]] = None,
    model: str = "claude-sonnet-4-5"
) -> Dict[str, Any] | None:
    """Run the agent loop with SEC filing tools."""
    if tools is None:
        tools = EXPLORE_TOOLS
    request_body = {
        "model": model,
        "system": system_prompt,
        "max_tokens": 10000,
        "tools": tools,
        "tool_choice": {"type": "auto"},
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    }

    parsed = None

    for i in range(max_iterations):
        request_body["messages"] = input_messages
        response = client.messages.create(**request_body)

        tool_use_items = [item for item in response.content if getattr(item, 'type', None) == 'tool_use']
        thinking_items = [item for item in response.content if getattr(item, 'type', None) == 'thinking']
        text_items = [item for item in response.content if getattr(item, 'type', None) == 'text']

        if thinking_items:
            for thinking_item in thinking_items:
                trajectory.append({
                    "type": "thinking",
                    "tool_name": None,
                    "arguments": None,
                    "output": thinking_item.thinking
                })

        if not tool_use_items:
            for item in text_items:
                if item.type == "text":
                    content = item.text
                    parsed = parse_fn(content)
                    trajectory.append({
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": content,
                        **parsed
                    })
            break

        serialized_items = []
        for item in response.content:
            serialized_item = item.model_dump(mode="python")
            if 'status' in serialized_item:
                del serialized_item['status']
            serialized_items.append(serialized_item)

        input_messages.append({"role": "assistant", "content": serialized_items})

        for tool_call in tool_use_items:
            tool_args = tool_call.input
            tool_name = tool_call.name

            result = tool_executor.execute(tool_name, tool_args, i)

            tool_output_msg = {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": f"[Tool call #{i+1}] {result}"
            }
            input_messages.append({"role": "user", "content": [tool_output_msg]})

            trajectory.append({
                "type": "tool_call",
                "tool_name": tool_name,
                "arguments": tool_args,
                "output": f"[Tool call #{i+1}] {result}"
            })

    return parsed


def force_output(
    client,
    input_messages: List[Dict[str, Any]],
    trajectory: List[Dict[str, Any]],
    force_message: str,
    parse_fn: Callable[[str], Dict[str, Any]],
    system_prompt: str,
    model: str = "claude-sonnet-4-5"
) -> Dict[str, Any] | None:
    """Force the agent to output structured response."""
    input_messages.append({"role": "user", "content": force_message})

    force_request = {
        "model": model,
        "system": system_prompt,
        "max_tokens": 4000,
        "messages": input_messages
    }

    response = client.messages.create(**force_request)

    thinking_items = [item for item in response.content if getattr(item, 'type', None) == 'thinking']
    text_items = [item for item in response.content if getattr(item, 'type', None) == 'text']

    if thinking_items:
        for thinking_item in thinking_items:
            trajectory.append({
                "type": "thinking",
                "tool_name": None,
                "arguments": None,
                "output": thinking_item.thinking
            })

    parsed = None
    for item in text_items:
        if item.type == "text":
            content = item.text
            parsed = parse_fn(content)
            trajectory.append({
                "type": "forced_output",
                "tool_name": None,
                "arguments": None,
                "output": content,
                **parsed
            })

    return parsed


# ============================================================================
# Batch Processing
# ============================================================================

def run_batch_files(
    run_single_fn: Callable,
    files_to_process: List[str],
    total_files: int,
    valid_files: List[str],
    max_workers: int
) -> Dict[str, Any]:
    """Run batch processing on files."""
    results = []
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(
            f"[bold blue]Processing {len(files_to_process)}/{len(valid_files)} valid files",
            total=len(files_to_process)
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(run_single_fn, f): f
                for f in files_to_process
            }

            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    results.append({"file": filepath, "status": "success", "result": result})
                except Exception as e:
                    errors.append({"file": filepath, "error": str(e)})
                progress.advance(task)

    print()
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(errors)}")

    if errors:
        print("\nFailed files:")
        for item in errors:
            print(f"  {item['file']}: {item['error']}")

    return {
        "total": total_files,
        "valid": len(valid_files),
        "processed": len(files_to_process),
        "successful": len(results),
        "failed": len(errors),
        "errors": errors
    }


# ============================================================================
# SEC Explorer Agent
# ============================================================================


def parse_structured_output(content: str) -> Dict[str, Any]:
    """Parse structured output from agent response."""
    return {
        "clues": parse_tag(content, "clues"),
        "question": parse_tag(content, "question"),
        "truth": parse_tag(content, "truth"),
        "supporting_items": parse_supporting_items(content)
    }


class SecExplorerAgent:
    """Agent for creating challenging questions from SEC filings."""

    def __init__(
        self,
        collection_name: str = "sec_filings",
        max_iterations: int = 30,
        model: str = "claude-sonnet-4-5",
    ):
        self.model = model
        self.client = get_anthropic_client()
        self.collection_name = collection_name
        self.max_iterations = max_iterations

        # Create shared clients to avoid "too many open files" errors
        self.chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            database=os.getenv("CHROMA_DATABASE")
        )
        self.openai_client = get_embedding_client()
        self.reranker = BasetenReranker(
            token_counter=count_tokens,
            max_tokens=5500,
            batch_size=100,
            max_concurrent_requests=32,
        )

    def _get_overview(self, data: Dict[str, Any]) -> str:
        """Build overview text from available forms."""
        overview_txt = ""
        for k, v in data["available_forms"].items():
            overview_txt += f"Form Type: {k}\n"
            for accession_no in v:
                overview_txt += f"  File Accession No: {accession_no}\n"
            overview_txt += "\n\n"
        return overview_txt

    def _get_initial_msg(
        self,
        tool_executor: SECToolExecutor,
        company_name: str,
        data: Dict[str, Any],
        truth_type: str
    ) -> str:
        """Build the initial message for the agent."""
        overview = self._get_overview(data)
        available_forms = data["available_forms"]
        chunks = random_chunks_for_init(available_forms, tool_executor.all_chunks)
        tool_executor.seen_chunk_ids.update([chunk["id"] for chunk in chunks])
        formatted_chunks = format_chunks(chunks)

        return EXPLORATION_PROMPT.format(
            company_name=company_name,
            overview=overview,
            random_chunks_init=formatted_chunks,
            truth_type=truth_type
        )

    def run_single(self, filepath: str) -> Dict[str, Any]:
        """Process a single file to generate questions."""
        with open(filepath, "r") as f:
            data = json.load(f)

        company_name = data["company_name"]
        ticker = data["ticker"]
        truth_type = data["truth_type"]

        tool_executor = SECToolExecutor(
            collection_name=self.collection_name,
            ticker=ticker,
            chroma_client=self.chroma_client,
            openai_client=self.openai_client,
            reranker=self.reranker
        )

        available_forms = data.get("available_forms", {})
        for form_type in INFO_RICH_FORMS:
            if form_type in available_forms:
                for accession_no in available_forms[form_type]:
                    if accession_no not in tool_executor.all_chunks:
                        raise ValueError(f"Key form {form_type} ({accession_no}) not found in ChromaDB")

        trajectory = []

        initial_msg = self._get_initial_msg(
            tool_executor=tool_executor,
            company_name=company_name,
            data=data,
            truth_type=truth_type
        )

        input_messages = [{"role": "user", "content": initial_msg}]
        trajectory.append({
            "type": "input_text",
            "tool_name": None,
            "arguments": None,
            "output": initial_msg
        })

        parsed = run_agent_loop(
            client=self.client,
            input_messages=input_messages,
            trajectory=trajectory,
            tool_executor=tool_executor,
            max_iterations=self.max_iterations,
            parse_fn=parse_structured_output,
            system_prompt=EXPLORATION_INSTRUCTION,
            model=self.model
        )

        if parsed is None or parsed.get("question") is None:
            parsed = force_output(
                client=self.client,
                input_messages=input_messages,
                trajectory=trajectory,
                force_message=SEC_FORCE_OUTPUT_MESSAGE,
                parse_fn=parse_structured_output,
                system_prompt=EXPLORATION_INSTRUCTION,
                model=self.model
            )

        clues = parsed["clues"] if parsed else None
        question = parsed["question"] if parsed else None
        truth = parsed["truth"] if parsed else None
        supporting_items = parsed["supporting_items"] if parsed else []

        # Get chunk contents for supporting items
        items_and_contents = {}

        for item in supporting_items:
            item_id = item.get("id", "")
            if item_id:
                # Find the full chunk document from all_chunks
                for accession_chunks in tool_executor.all_chunks.values():
                    for c in accession_chunks:
                        if c["id"] == item_id:
                            items_and_contents[item_id] = c["document"]
                            break

        # Save in same schema as web explorer agents
        result = {
            "tasks": [
                {
                    "level": 0,
                    "clues": clues,
                    "question": question,
                    "truth": truth,
                    "truth_type": truth_type,
                    "supporting_items": supporting_items,
                    "items_and_contents": items_and_contents
                }
            ],
            "seen_chunk_ids": list(tool_executor.seen_chunk_ids)
        }

        # Update the data file
        data["tasks"] = result["tasks"]

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Processed {filepath}")

        return result

    def is_processed(self, filepath: str) -> bool:
        """Check if a file has been fully processed with all required fields."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Check that tasks array exists and has at least one entry
            if not data.get("tasks") or len(data["tasks"]) == 0:
                return False

            task = data["tasks"][0]

            # Check required fields are not None
            if task.get("clues") is None:
                return False
            if task.get("question") is None:
                return False
            if task.get("truth") is None:
                return False

            # Check supporting_items has exactly 3 entries
            supporting_items = task.get("supporting_items", [])
            if len(supporting_items) != 3:
                return False

            # Verify each item has required fields
            for item in supporting_items:
                if not item.get("id"):
                    return False

            # Check items_and_contents has exactly 3 entries
            items_and_contents = task.get("items_and_contents", {})
            if len(items_and_contents) != 3:
                return False

            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def has_key_form(self, filepath: str) -> bool:
        """Check if file has key forms (10-K or 20-F)."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return data.get("contains_key_form", False)
        except Exception:
            return False

    def run_batch(self, input_dir: str, max_workers: int = 5) -> Dict[str, Any]:
        """Run batch processing on all files in a directory."""
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"Error: Directory {input_dir} does not exist")
            return {"total": 0, "valid": 0, "processed": 0, "successful": 0, "failed": 0, "errors": []}

        json_files = list(input_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return {"total": 0, "valid": 0, "processed": 0, "successful": 0, "failed": 0, "errors": []}

        # Filter for files with key forms
        key_form_files = [f for f in json_files if self.has_key_form(f)]
        files_to_process = [f for f in key_form_files if not self.is_processed(f)]

        print(f"Found {len(json_files)} total JSON files")
        print(f"With key forms (10-K or 20-F): {len(key_form_files)}")
        print(f"Already processed: {len(key_form_files) - len(files_to_process)}")
        print(f"To process: {len(files_to_process)}")

        if not files_to_process:
            print("All files have already been processed!")
            return {
                "total": len(json_files),
                "valid": len(key_form_files),
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "errors": []
            }

        return run_batch_files(
            run_single_fn=self.run_single,
            files_to_process=[str(f) for f in files_to_process],
            total_files=len(json_files),
            valid_files=[str(f) for f in key_form_files],
            max_workers=max_workers
        )


def main():
    parser = argparse.ArgumentParser(description="Process SEC filing JSON files with the SEC Explorer Agent")
    parser.add_argument("--input-dir", "-i", required=True, help="Path to input directory containing JSON files")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Maximum number of parallel workers (default: 8)")
    parser.add_argument("--max-iterations", "-n", type=int, default=20, help="Maximum iterations per file (default: 20)")
    parser.add_argument("--collection", type=str, default="sec_test_1_14", help="ChromaDB collection name (default: sec_test_1_14)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5", help="Model for exploration (default: claude-sonnet-4-5)")

    args = parser.parse_args()

    print("Initializing SEC Explorer Agent...")
    print(f"Model: {args.model}")
    agent = SecExplorerAgent(
        collection_name=args.collection,
        max_iterations=args.max_iterations,
        model=args.model,
    )

    result = agent.run_batch(args.input_dir, max_workers=args.max_workers)

    print("-" * 40)
    print(f"Total files: {result['total']}")
    print(f"Valid files: {result['valid']}")
    print(f"Processed: {result['processed']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")


if __name__ == "__main__":
    main()
