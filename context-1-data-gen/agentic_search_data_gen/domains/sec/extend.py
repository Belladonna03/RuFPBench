"""Cross-Company Bridging Agent for SEC filings.

This module creates cross-company multi-hop questions by:
1. Phase 1: Finding a bridging connection from Company A to Company B
2. Phase 2: Finding supporting clues within Company B
"""

import os
import re
import argparse
import json
import random
from glob import glob
from typing import Any, Dict, List, Set

import chromadb
from chromadb.utils.embedding_functions import Bm25EmbeddingFunction

from .prompts import (
    TRUTH_TYPES,
    SEC_BRIDGING_INSTRUCTION, SEC_BRIDGING_PROMPT, SEC_BRIDGING_FORCE_OUTPUT,
    SEC_SUPPORTING_CLUES_INSTRUCTION, SEC_SUPPORTING_CLUES_PROMPT, SEC_SUPPORTING_CLUES_FORCE_OUTPUT,
    SEC_BRIDGING_ITEM_EXTRACTION_PROMPT, SEC_SINGLE_ITEM_EXTRACTION_PROMPT
)
from ...core.utils import get_anthropic_client, count_tokens, parse_tag, parse_quotes, count_matching_quotes, min_required_matches, text_contains_quote, get_embedding_client
from ...core.rerank import BasetenReranker
from .explore import (
    SECToolExecutor, CompanySearchEngine, EXPLORE_TOOLS,
    run_agent_loop, force_output, run_batch_files
)
from .utils import parse_supporting_items, format_chunks as _format_chunks, get_latest_task, parse_single_item
from concurrent.futures import ThreadPoolExecutor, as_completed

# BM25 embedding function for hybrid search
bm25_ef = Bm25EmbeddingFunction(avg_len=4000, task="query")


# ============================================================================
# Phase 1: Bridging Tools (Cross-Collection Only)
# ============================================================================

BRIDGING_TOOLS = [
    {
        "name": "hybrid_search_across_all",
        "description": "Hybrid search (BM25 + semantic) across ALL filings in the entire SEC collection, not just one company. Returns ~10 most relevant chunks with their ticker symbols. Use this to find cross-company connections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to search across all SEC filings"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "grep_across_all",
        "description": "Regex pattern search across ALL filings in the entire SEC collection. Returns ~10 matching chunks with their ticker symbols. Use this to find exact matches across companies.",
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


# ============================================================================
# Phase 1: Bridging Tool Executor
# ============================================================================

class BridgingToolExecutor:
    """Handles tool execution for cross-collection bridging search."""

    def __init__(
        self,
        collection_name: str,
        exclude_ticker: str,
        chroma_client: chromadb.CloudClient = None,
        openai_client = None,
    ):
        self.collection_name = collection_name
        self.exclude_ticker = exclude_ticker  # Company A's ticker to exclude
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

        self.seen_chunk_ids: Set[str] = set()
        self.all_chunks: Dict[str, List[Dict[str, Any]]] = {}

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI."""
        resp = self.openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [e.embedding for e in resp.data]

    def hybrid_search_across_all(self, query: str, k: int = 10) -> str:
        """Hybrid search across entire SEC collection, excluding Company A."""
        sparse_vector = bm25_ef([query])[0]
        dense_vector = self._embed([query])[0]

        search = (
            chromadb.Search()
            .where(chromadb.Key("ticker") != self.exclude_ticker)
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

        result_chunks = []
        for id, doc, meta in zip(ids, documents, metadatas):
            if id not in self.seen_chunk_ids:
                chunk = {
                    "id": id,
                    "document": doc,
                    "accession_no": meta.get("source", "unknown"),
                    "ticker": meta.get("ticker", "unknown")
                }
                result_chunks.append(chunk)
                self.seen_chunk_ids.add(id)

                # Store for later retrieval
                ticker = chunk["ticker"]
                if ticker not in self.all_chunks:
                    self.all_chunks[ticker] = []
                self.all_chunks[ticker].append(chunk)

        return _format_chunks(result_chunks, include_ticker=True) if result_chunks else "No results found"

    def grep_across_all(self, pattern: str, k: int = 10) -> str:
        """Regex search across entire SEC collection, excluding Company A."""
        search = (
            chromadb.Search()
            .where(
                (chromadb.Key.DOCUMENT.regex(rf"(?i){pattern}")) &
                (chromadb.Key("ticker") != self.exclude_ticker)
            )
            .select(chromadb.Key.DOCUMENT, chromadb.Key.METADATA)
            .limit(k)
        )

        res = self.collection.search(search)
        ids = res["ids"][0]
        documents = res["documents"][0]
        metadatas = res["metadatas"][0]

        result_chunks = []
        for id, doc, meta in zip(ids, documents, metadatas):
            if id not in self.seen_chunk_ids:
                chunk = {
                    "id": id,
                    "document": doc,
                    "accession_no": meta.get("source", "unknown"),
                    "ticker": meta.get("ticker", "unknown")
                }
                result_chunks.append(chunk)
                self.seen_chunk_ids.add(id)

                # Store for later retrieval
                ticker = chunk["ticker"]
                if ticker not in self.all_chunks:
                    self.all_chunks[ticker] = []
                self.all_chunks[ticker].append(chunk)

        return _format_chunks(result_chunks, include_ticker=True) if result_chunks else "No results found"

    def execute(self, tool_name: str, tool_args: Dict[str, Any], iteration: int) -> str:
        """Execute a tool and return the result."""
        result = f"Iteration {iteration+1}\n\n"

        if tool_name == "hybrid_search_across_all":
            result += self.hybrid_search_across_all(tool_args["query"])
        elif tool_name == "grep_across_all":
            result += self.grep_across_all(tool_args["pattern"])
        else:
            result += f"Unknown tool: {tool_name}"

        return result


# ============================================================================
# Phase 1: Parse Bridging Output
# ============================================================================

def parse_bridging_output(content: str) -> Dict[str, Any]:
    """Parse bridging chunk output from Phase 1 agent."""
    result = {
        "bridging_clue": None,
        "chunk_id": None,
        "reasoning": None
    }

    result["bridging_clue"] = parse_tag(content, "bridging_clue")

    outer_match = re.search(r'<bridging_item>(.*?)</bridging_item>', content, re.DOTALL)
    if outer_match:
        inner = outer_match.group(1)
        result["chunk_id"] = parse_tag(inner, "chunk_id")
        result["reasoning"] = parse_tag(inner, "reasoning")

    return result


# ============================================================================
# Phase 2: Parse Supporting Clues Output
# ============================================================================

def parse_supporting_clues_output(content: str) -> Dict[str, Any]:
    """Parse supporting clues output from Phase 2 agent."""
    return {
        "supporting_clues": parse_tag(content, "supporting_clues"),
        "question": parse_tag(content, "question"),
        "truth": parse_tag(content, "truth"),
        "supporting_items": parse_supporting_items(content)
    }


def parse_bridge_verification_response(content: str) -> Dict[str, Any]:
    outer_match = re.search(r'<bridging_item>(.*?)</bridging_item>', content, re.DOTALL)
    if not outer_match:
        return {}
    inner = outer_match.group(1)
    clue_quotes = parse_quotes(inner, 'clue_quotes')
    item_quotes = parse_quotes(inner, 'item_quotes')
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', inner, re.DOTALL)
    return {
        'clue_quotes': clue_quotes if clue_quotes else [],
        'item_quotes': item_quotes if item_quotes else [],
        'reasoning': reasoning_match.group(1).strip() if reasoning_match else '',
        'not_relevant': clue_quotes is None or item_quotes is None
    }


def verify_bridge(
    client,
    model: str,
    prev_truth: str,
    prev_clues: str,
    prev_question: str,
    original_ticker: str,
    target_ticker: str,
    bridging_clue: str,
    bridging_chunk_id: str,
    bridging_reasoning: str,
    bridging_chunk_content: str,
    prev_item_id: str,
    prev_item_content: str,
    max_retries: int = 3
) -> tuple[bool, Dict[str, Any]]:
    prompt = SEC_BRIDGING_ITEM_EXTRACTION_PROMPT.format(
        prev_truth=prev_truth,
        prev_company_name=original_ticker,
        new_company_name=target_ticker,
        prev_clues=prev_clues,
        prev_question=prev_question,
        bridging_clue=bridging_clue,
        bridging_chunk_id=bridging_chunk_id,
        bridging_reasoning=bridging_reasoning,
        bridging_chunk_content=bridging_chunk_content,
        prev_truth_supporting_item=f"CHUNK ID: {prev_item_id}\n{prev_item_content}"
    )

    for _ in range(max_retries):
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = ""
        for item in response.content:
            if item.type == "text":
                response_text = item.text
                break

        extracted = parse_bridge_verification_response(response_text)
        if not extracted or extracted.get('not_relevant'):
            continue

        item_quotes = extracted.get('item_quotes', [])
        if item_quotes:
            matches = count_matching_quotes(item_quotes, bridging_chunk_content)
            required = min_required_matches(len(item_quotes))
            if matches >= required:
                return True, extracted

    return False, extracted if extracted else {}


def verify_single_supporting_item(
    client,
    model: str,
    clues: str,
    question: str,
    truth: str,
    item_id: str,
    item_reasoning: str,
    item_content: str,
    max_retries: int = 3
) -> tuple[bool, Dict[str, Any] | None]:
    prompt = SEC_SINGLE_ITEM_EXTRACTION_PROMPT.format(
        clues=clues,
        question=question,
        truth=truth,
        chunk_id=item_id,
        item_reasoning=item_reasoning,
        item_content=item_content
    )

    extracted = None
    for _ in range(max_retries):
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = ""
        for item in response.content:
            if item.type == "text":
                response_text = item.text
                break

        extracted = parse_single_item(response_text)
        if not extracted or extracted.get('not_relevant'):
            continue

        clue_quotes = extracted.get('clue_quotes', [])
        item_quotes = extracted.get('item_quotes', [])

        clue_valid = True
        if clue_quotes:
            matches = count_matching_quotes(clue_quotes, clues)
            required = min_required_matches(len(clue_quotes))
            clue_valid = matches >= required

        item_valid = True
        if item_quotes:
            matches = count_matching_quotes(item_quotes, item_content)
            required = min_required_matches(len(item_quotes))
            item_valid = matches >= required

        truth_valid = True
        if extracted.get('contains_truth'):
            truth_quotes = extracted.get('truth_quotes', [])
            for q in truth_quotes:
                if q.lower() != 'none' and not text_contains_quote(item_content, q):
                    truth_valid = False
                    break

        if clue_valid and item_valid and truth_valid:
            return True, extracted

    return False, extracted


def verify_phase2_items(
    client,
    model: str,
    clues: str,
    question: str,
    truth: str,
    supporting_items: List[Dict],
    items_and_contents: Dict[str, str],
    max_retries: int = 3
) -> tuple[bool, List[Dict[str, Any]]]:
    results = [None, None]

    def verify_item(idx, item):
        item_id = item.get('id', '')
        item_content = items_and_contents.get(item_id, '')
        if not item_content:
            return idx, False, None
        passed, extracted = verify_single_supporting_item(
            client, model, clues, question, truth,
            item_id, item.get('reasoning', ''), item_content, max_retries
        )
        return idx, passed, extracted

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(verify_item, i, item) for i, item in enumerate(supporting_items)]
        for future in as_completed(futures):
            idx, passed, extracted = future.result()
            results[idx] = {'passed': passed, 'extracted': extracted}

    all_passed = all(r and r['passed'] for r in results)
    extracted_items = [r['extracted'] if r else None for r in results]

    truth_contained = any(
        e and e.get('contains_truth') for e in extracted_items
    )

    return all_passed and truth_contained, extracted_items


# ============================================================================
# Main Bridging Agent
# ============================================================================

class SECBridgingAgent:
    """Agent for creating cross-company multi-hop questions.

    This agent runs two phases:
    1. Phase 1: Find a bridging chunk connecting Company A to Company B
    2. Phase 2: Find 2 supporting clues within Company B
    """

    def __init__(
        self,
        collection_name: str = "sec_filings",
        max_iterations_phase1: int = 10,
        max_iterations_phase2: int = 15,
        agent_model: str = "claude-sonnet-4-5",
        verification_model: str = "claude-opus-4-5",
    ):
        self.client = get_anthropic_client()
        self.collection_name = collection_name
        self.max_iterations_phase1 = max_iterations_phase1
        self.max_iterations_phase2 = max_iterations_phase2
        self.agent_model = agent_model
        self.verification_model = verification_model

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

    def _run_phase1(
        self,
        original_ticker: str,
        prev_clues: str,
        prev_question: str,
        prev_truth: str,
        truth_supporting_items_str: str,
        letter: str
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], BridgingToolExecutor]:
        """Phase 1: Find bridging chunk to a new company."""
        tool_executor = BridgingToolExecutor(
            collection_name=self.collection_name,
            exclude_ticker=original_ticker,
            chroma_client=self.chroma_client,
            openai_client=self.openai_client
        )

        trajectory = []

        formatted_prompt = SEC_BRIDGING_PROMPT.format(
            original_ticker=original_ticker,
            prev_clues=prev_clues,
            prev_question=prev_question,
            prev_truth=prev_truth,
            truth_supporting_items=truth_supporting_items_str,
            letter=letter
        )

        input_messages = [{"role": "user", "content": formatted_prompt}]
        trajectory.append({
            "type": "input_text",
            "tool_name": None,
            "arguments": None,
            "output": formatted_prompt
        })

        parsed = run_agent_loop(
            client=self.client,
            input_messages=input_messages,
            trajectory=trajectory,
            tool_executor=tool_executor,
            max_iterations=self.max_iterations_phase1,
            parse_fn=parse_bridging_output,
            system_prompt=SEC_BRIDGING_INSTRUCTION,
            tools=BRIDGING_TOOLS,
            model=self.agent_model
        )

        if parsed is None or parsed.get("chunk_id") is None:
            parsed = force_output(
                client=self.client,
                input_messages=input_messages,
                trajectory=trajectory,
                force_message=SEC_BRIDGING_FORCE_OUTPUT,
                parse_fn=parse_bridging_output,
                system_prompt=SEC_BRIDGING_INSTRUCTION
            )

        return parsed, trajectory, tool_executor

    def _run_phase2(
        self,
        target_ticker: str,
        bridging_chunk_content: str,
        bridge_reasoning: str,
        original_ticker: str,
        prev_clues: str,
        prev_truth: str,
        truth_type: str,
        letter: str
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], SECToolExecutor]:
        """Phase 2: Find supporting clues within Company B."""
        tool_executor = SECToolExecutor(
            collection_name=self.collection_name,
            ticker=target_ticker,
            chroma_client=self.chroma_client,
            openai_client=self.openai_client,
            reranker=self.reranker
        )

        trajectory = []

        formatted_prompt = SEC_SUPPORTING_CLUES_PROMPT.format(
            original_ticker=original_ticker,
            target_ticker=target_ticker,
            bridging_chunk_content=bridging_chunk_content,
            bridge_reasoning=bridge_reasoning,
            prev_clues=prev_clues,
            prev_truth=prev_truth,
            truth_type=truth_type,
            letter=letter
        )

        input_messages = [{"role": "user", "content": formatted_prompt}]
        trajectory.append({
            "type": "input_text",
            "tool_name": None,
            "arguments": None,
            "output": formatted_prompt
        })

        parsed = run_agent_loop(
            client=self.client,
            input_messages=input_messages,
            trajectory=trajectory,
            tool_executor=tool_executor,
            max_iterations=self.max_iterations_phase2,
            parse_fn=parse_supporting_clues_output,
            system_prompt=SEC_SUPPORTING_CLUES_INSTRUCTION,
            tools=EXPLORE_TOOLS,
            model=self.agent_model
        )

        if parsed is None or parsed.get("supporting_clues") is None:
            parsed = force_output(
                client=self.client,
                input_messages=input_messages,
                trajectory=trajectory,
                force_message=SEC_SUPPORTING_CLUES_FORCE_OUTPUT,
                parse_fn=parse_supporting_clues_output,
                system_prompt=SEC_SUPPORTING_CLUES_INSTRUCTION
            )

        return parsed, trajectory, tool_executor

    def is_valid(self, filepath: str) -> bool:
        """Check if file is valid for bridging.

        Requirements:
        - Latest task has passed_verification=True
        - Latest task has additional_chunks_filtered=True
        - Latest task has NOT been attempted for extension (extend_attempted != True)
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            tasks = data.get("tasks", [])
            if not tasks:
                return False

            latest_task = get_latest_task(tasks)
            if latest_task is None:
                return False

            if latest_task.get("passed_verification") is not True:
                return False

            if latest_task.get("additional_chunks_filtered") is not True:
                return False

            # Skip if extension was already attempted (verification failed)
            if latest_task.get("extend_attempted") is True:
                return False

            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def is_processed(self, filepath: str) -> bool:
        """Check if file has been fully processed."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            tasks = data.get("tasks", [])
            if not tasks:
                return False

            latest_task = get_latest_task(tasks)
            if latest_task is None:
                return False

            if not latest_task.get("passed_verification"):
                return False

            next_level = latest_task.get("level", 0) + 1
            next_task = None
            for t in tasks:
                if t.get("level") == next_level:
                    next_task = t
                    break

            if next_task is None:
                return False

            # Check required fields
            if next_task.get("clues") is None:
                return False
            if next_task.get("question") is None:
                return False
            if next_task.get("truth") is None:
                return False

            # Check bridging_item
            bridging_item = next_task.get("bridging_item")
            if bridging_item is None or not bridging_item.get("id"):
                return False

            if not next_task.get("new_company"):
                return False

            supporting_items = next_task.get("supporting_items", [])
            if len(supporting_items) != 2:
                return False

            items_and_contents = next_task.get("items_and_contents", {})
            if len(items_and_contents) != 3:
                return False

            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def run_single(self, input_filepath: str) -> Dict[str, Any]:
        """Process a single file to create cross-company bridge."""
        with open(input_filepath, "r") as f:
            data = json.load(f)

        print(f"Running single file: {input_filepath}")

        tasks = data.get("tasks", [])
        prev_task = get_latest_task(tasks)
        if prev_task is None:
            raise ValueError(f"No tasks found in {input_filepath}")

        original_ticker = data.get("ticker")
        if not original_ticker:
            raise ValueError(f"No ticker found in {input_filepath}")

        prev_level = prev_task.get("level", 0)
        prev_clues = prev_task["clues"]
        prev_question = prev_task["question"]
        prev_truth = prev_task["truth"]
        prev_items_and_contents = prev_task["items_and_contents"]

        prev_truth_item_id = None
        prev_truth_item_content = None
        for item in prev_task["supporting_items"]:
            if item.get("contains_truth"):
                prev_truth_item_id = item.get("id", "")
                prev_truth_item_content = prev_items_and_contents.get(prev_truth_item_id, "")
                break

        if not prev_truth_item_id:
            raise ValueError(f"No supporting item contains truth in {input_filepath}")

        truth_supporting_items_str = f"CHUNK ID: {prev_truth_item_id}\n{prev_truth_item_content}\n\n"

        truth_type = random.choice(TRUTH_TYPES)
        letter = chr(ord('B') + prev_level)

        print(f"  Phase 1: Finding bridge from {original_ticker}...")
        phase1_result, phase1_trajectory, bridging_executor = self._run_phase1(
            original_ticker=original_ticker,
            prev_clues=prev_clues,
            prev_question=prev_question,
            prev_truth=prev_truth,
            truth_supporting_items_str=truth_supporting_items_str,
            letter=letter
        )

        if not phase1_result or not phase1_result.get("chunk_id"):
            raise ValueError("Phase 1 failed: Could not find bridging chunk")

        bridging_chunk_id = phase1_result["chunk_id"]
        bridging_chunk_content = ""
        target_ticker = None
        for ticker, ticker_chunks in bridging_executor.all_chunks.items():
            for chunk in ticker_chunks:
                if chunk["id"] == bridging_chunk_id:
                    bridging_chunk_content = chunk["document"]
                    target_ticker = ticker
                    break
            if target_ticker:
                break

        if not target_ticker:
            raise ValueError(f"Phase 1 failed: Could not find ticker for bridging chunk {bridging_chunk_id}")

        # Validate that target_ticker is different from original_ticker
        if target_ticker == original_ticker:
            raise ValueError(f"Phase 1 failed: Bridging chunk {bridging_chunk_id} is from the same company ({target_ticker})")

        print(f"  Phase 1 complete: Found bridge to {target_ticker}")

        print(f"  Verifying bridge...")
        bridge_passed, bridge_extracted = verify_bridge(
            client=self.client,
            model=self.verification_model,
            prev_truth=prev_truth,
            prev_clues=prev_clues,
            prev_question=prev_question,
            original_ticker=original_ticker,
            target_ticker=target_ticker,
            bridging_clue=phase1_result.get("bridging_clue", ""),
            bridging_chunk_id=bridging_chunk_id,
            bridging_reasoning=phase1_result.get("reasoning", ""),
            bridging_chunk_content=bridging_chunk_content,
            prev_item_id=prev_truth_item_id,
            prev_item_content=prev_truth_item_content
        )

        if not bridge_passed:
            # Mark as attempted so we don't retry
            prev_task["extend_attempted"] = True
            with open(input_filepath, "w") as f:
                json.dump(data, f, indent=4)
            raise ValueError("Bridge verification failed")

        print(f"  Bridge verified")

        print(f"  Phase 2: Finding supporting clues in {target_ticker}...")
        phase2_result, phase2_trajectory, supporting_executor = self._run_phase2(
            target_ticker=target_ticker,
            bridging_chunk_content=bridging_chunk_content,
            bridge_reasoning=phase1_result.get("reasoning", ""),
            original_ticker=original_ticker,
            prev_clues=prev_clues,
            prev_truth=prev_truth,
            truth_type=truth_type,
            letter=letter
        )

        if not phase2_result or not phase2_result.get("supporting_clues"):
            raise ValueError("Phase 2 failed: Could not find supporting clues")

        print(f"  Phase 2 complete")

        # Get bridging_clue from phase 1 and supporting_clues from phase 2
        bridging_clue = phase1_result.get("bridging_clue", "")
        supporting_clues = phase2_result["supporting_clues"]

        # Concatenate bridging_clue + supporting_clues to form the final clues
        # The bridging_clue uses abstract references (e.g., "that same country")
        # The supporting_clues describe Company B without revealing prev_truth
        combined_clues = f"{bridging_clue} {supporting_clues}".strip()

        question = phase2_result["question"]
        truth = phase2_result["truth"]
        supporting_items = phase2_result["supporting_items"]

        new_items_and_contents = {}
        failed_ids = []

        if bridging_chunk_id and bridging_chunk_content:
            new_items_and_contents[bridging_chunk_id] = bridging_chunk_content

        for item in supporting_items:
            item_id = item.get("id", "")
            if item_id:
                found = False
                for accession_chunks in supporting_executor.all_chunks.values():
                    for chunk in accession_chunks:
                        if chunk["id"] == item_id:
                            new_items_and_contents[item_id] = chunk["document"]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    failed_ids.append({"id": item_id, "error": "Chunk not found"})

        print(f"  Verifying phase 2 items...")
        phase2_passed, phase2_extracted = verify_phase2_items(
            client=self.client,
            model=self.verification_model,
            clues=combined_clues,
            question=question,
            truth=truth,
            supporting_items=supporting_items,
            items_and_contents=new_items_and_contents
        )

        if not phase2_passed:
            # Mark as attempted so we don't retry
            prev_task["extend_attempted"] = True
            with open(input_filepath, "w") as f:
                json.dump(data, f, indent=4)
            raise ValueError("Phase 2 verification failed")

        print(f"  Phase 2 verified")

        for i, item in enumerate(supporting_items):
            if phase2_extracted[i]:
                item.update(phase2_extracted[i])

        new_task = {
            "level": prev_level + 1,
            "letter": letter,
            "clues": combined_clues,
            "bridging_clue": bridging_clue,
            "supporting_clues": supporting_clues,
            "question": question,
            "truth": truth,
            "truth_type": truth_type,
            "new_company": target_ticker,
            "bridging_item": {
                "id": bridging_chunk_id,
                "ticker": target_ticker,
                "relevant_prev_id": prev_truth_item_id,
                "reasoning": bridge_extracted.get("reasoning", "") or phase1_result.get("reasoning", ""),
                "clue_quotes": bridge_extracted.get("clue_quotes", []),
                "item_quotes": bridge_extracted.get("item_quotes", [])
            },
            "supporting_items": supporting_items,
            "items_and_contents": new_items_and_contents,
            "failed_ids": failed_ids,
            "passed_verification": True
        }

        data["tasks"].append(new_task)

        with open(input_filepath, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Wrote {input_filepath}")

        return data

    def run_batch(self, input_dir: str, max_workers: int = 8) -> Dict[str, Any]:
        """Run batch processing on all valid files."""
        all_files = glob(os.path.join(input_dir, "*.json"))
        valid_files = [f for f in all_files if self.is_valid(f)]
        files_to_process = [f for f in valid_files if not self.is_processed(f)]

        print(f"Found {len(all_files)} total JSON files")
        print(f"Valid for bridging (passed verification): {len(valid_files)}")
        print(f"Already bridged: {len(valid_files) - len(files_to_process)}")
        print(f"To process: {len(files_to_process)}")

        if not files_to_process:
            print("All valid files have already been bridged!")
            return {
                "total": len(all_files),
                "valid": len(valid_files),
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "errors": []
            }

        return run_batch_files(self.run_single, files_to_process, len(all_files), valid_files, max_workers)


def main():
    parser = argparse.ArgumentParser(description="Run the SECBridgingAgent to create cross-company multi-hop questions.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing JSON files")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Maximum number of parallel workers (default: 8)")
    parser.add_argument("--max-iterations-phase1", type=int, default=10, help="Maximum iterations for Phase 1 (default: 10)")
    parser.add_argument("--max-iterations-phase2", type=int, default=15, help="Maximum iterations for Phase 2 (default: 15)")
    parser.add_argument("--collection", type=str, default="sec_test_1_14", help="ChromaDB collection name (default: sec_test_1_14)")
    parser.add_argument("--agent-model", type=str, default="claude-sonnet-4-5", help="Model for agent loops (default: claude-sonnet-4-5)")
    parser.add_argument("--verification-model", type=str, default="claude-opus-4-5", help="Model for verification (default: claude-opus-4-5)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        exit(1)

    print(f"Input directory: {args.input}")
    print(f"Collection: {args.collection}")
    print(f"Agent model: {args.agent_model}")
    print(f"Verification model: {args.verification_model}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max iterations Phase 1: {args.max_iterations_phase1}")
    print(f"Max iterations Phase 2: {args.max_iterations_phase2}")
    print("-" * 40)

    agent = SECBridgingAgent(
        collection_name=args.collection,
        max_iterations_phase1=args.max_iterations_phase1,
        max_iterations_phase2=args.max_iterations_phase2,
        agent_model=args.agent_model,
        verification_model=args.verification_model,
    )
    result = agent.run_batch(args.input, max_workers=args.max_workers)

    print("-" * 40)
    print(f"Total files: {result['total']}")
    print(f"Valid files: {result['valid']}")
    print(f"Processed: {result['processed']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    if result['errors']:
        print("\nErrors:")
        for err in result['errors']:
            print(f"  {err['file']}: {err['error']}")


if __name__ == "__main__":
    main()
