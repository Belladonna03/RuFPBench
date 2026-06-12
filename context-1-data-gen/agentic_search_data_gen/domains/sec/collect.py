"""Collector Agent for finding additional supporting chunks.

This module finds additional chunks that contain the same factual information
as the original supporting chunks from explore.py output.

For each task with 3 supporting_chunks, it runs 3 agents in parallel to find
additional chunks for each clue.
"""

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Set, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
from dotenv import load_dotenv
import chromadb

from .prompts import CHUNK_COLLECTOR_SYSTEM_PROMPT, CHUNK_COLLECTOR_PROMPT, CHUNK_COLLECTOR_PROMPT_TRUTH_VER
from ...core.utils import count_tokens, get_anthropic_client, get_embedding_client
from ...core.rerank import BasetenReranker
from .explore import CompanySearchEngine
from .utils import get_latest_task, format_chunks

load_dotenv()


# Tool definitions for collector agent
COLLECTOR_TOOLS = [
    {
        "name": "search",
        "description": "Search across all filings for this company. Returns up to 10 most relevant chunks.",
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
        "name": "grep",
        "description": "Grep for regex pattern matches across all filings for this company. Returns up to 10 matching chunks.",
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
    }
]


def parse_additional_chunks(output_text: str) -> List[Dict[str, str]]:
    """Parse the <additional_chunks> output from the collector agent.

    Parses chunk_id from model output and normalizes to 'id' internally.
    """
    chunks = []

    # Check for "None" response
    match = re.search(r'<additional_chunks>\s*None\s*</additional_chunks>', output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return []

    # Parse chunk entries
    outer_match = re.search(r'<additional_chunks>(.*?)</additional_chunks>', output_text, re.DOTALL)
    if outer_match:
        chunks_content = outer_match.group(1)
        chunk_matches = re.findall(r'<chunk>(.*?)</chunk>', chunks_content, re.DOTALL)
        for chunk_match in chunk_matches:
            # Parse chunk_id from model output (normalize to 'id' internally)
            chunk_id_match = re.search(r'<chunk_id>(.*?)</chunk_id>', chunk_match, re.DOTALL)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', chunk_match, re.DOTALL)

            if chunk_id_match:
                chunk_id = chunk_id_match.group(1).strip()
                # Skip empty IDs
                if chunk_id:
                    chunks.append({
                        'id': chunk_id,  # Normalize chunk_id -> id
                        'reasoning': reasoning_match.group(1).strip() if reasoning_match else ''
                    })

    return chunks


class ChunkCollectorAgent:
    """Agent that finds additional chunks containing the same factual information."""

    def __init__(
        self,
        collection_name: str = "sec_filings",
        max_iterations: int = 10,
        model: str = "claude-sonnet-4-5",
    ):
        self.collection_name = collection_name
        self.max_iterations = max_iterations
        self.model = model

        self.anthropic_client = get_anthropic_client()

        # Create shared clients to avoid connection exhaustion
        self.chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            database=os.getenv("CHROMA_DATABASE")
        )
        self.collection = self.chroma_client.get_collection(collection_name)
        self.openai_client = get_embedding_client()

        self.reranker = BasetenReranker(
            token_counter=count_tokens,
            max_tokens=5500,
            batch_size=100,
            max_concurrent_requests=32,
        )

    def _get_all_chunks(self, ticker: str, max_retries: int = 5) -> List[Dict[str, Any]]:
        """Get all chunks for a given ticker with retry logic."""
        search = (
            chromadb.Search()
            .where(chromadb.Key("ticker") == ticker)
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

        chunks = []
        for id, document, metadata in zip(res['ids'][0], res['documents'][0], res['metadatas'][0]):
            chunks.append({
                "id": id,
                "document": document,
                "accession_no": metadata.get('source', '')
            })
        return chunks

    def _search(
        self,
        all_chunks: List[Dict[str, Any]],
        seen_chunk_ids: Set[str],
        query: str,
        company_search_engine: CompanySearchEngine
    ) -> tuple[str, List[str]]:
        """Search for chunks matching the query, filtering out seen chunks."""
        # First get candidates from company search
        company_search_results = company_search_engine.search(query)

        # Filter out seen chunks
        filtered_chunks = [chunk for chunk in company_search_results if chunk["id"] not in seen_chunk_ids]

        if not filtered_chunks:
            return "No new chunks found matching the query.", []

        docs = [chunk["document"] for chunk in filtered_chunks]

        # Rerank results
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
        new_chunk_ids = []
        for res in results:
            original_chunk = filtered_chunks[res.original_index]
            res_chunks.append({
                "id": original_chunk["id"],
                "document": res.document
            })
            new_chunk_ids.append(original_chunk["id"])

        return format_chunks(res_chunks), new_chunk_ids

    def _grep(
        self,
        all_chunks: List[Dict[str, Any]],
        seen_chunk_ids: Set[str],
        pattern: str
    ) -> tuple[str, List[str]]:
        """Grep for chunks matching the pattern, filtering out seen chunks."""
        # Filter out seen chunks
        filtered_chunks = [chunk for chunk in all_chunks if chunk["id"] not in seen_chunk_ids]

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return f"Invalid regex pattern: {pattern}", []

        matching_chunks = []
        new_chunk_ids = []
        for chunk in filtered_chunks:
            if regex.search(chunk["document"]):
                matching_chunks.append(chunk)
                new_chunk_ids.append(chunk["id"])
                if len(matching_chunks) >= 10:
                    break

        if not matching_chunks:
            return "No chunks found matching the pattern.", []

        return format_chunks(matching_chunks), new_chunk_ids

    def run_single_item(
        self,
        ticker: str,
        question: str,
        truth: str,
        clues: str,
        supporting_item: Dict[str, Any],
        chunk_content: str,
        all_chunks: List[Dict[str, Any]],
        company_search_engine: "CompanySearchEngine"
    ) -> List[Dict[str, str]]:
        """Run collector agent for a single supporting item.

        Args:
            ticker: Company ticker symbol
            question: The question being answered
            truth: The truth/answer
            clues: All clues context
            supporting_item: Dict with 'id', 'clue_quotes', 'item_quotes', 'contains_truth', 'truth_quotes'
            chunk_content: Full content of the chunk from items_and_contents
            all_chunks: Pre-fetched chunks for this ticker (shared across items)
            company_search_engine: Pre-initialized search engine (shared across items)

        Returns:
            List of additional chunks with 'id' and 'reasoning'
        """
        original_id = supporting_item['id']
        clue_quotes = supporting_item.get('clue_quotes', [])
        item_quotes = supporting_item.get('item_quotes', [])
        contains_truth = supporting_item.get('contains_truth', False)
        truth_quotes = supporting_item.get('truth_quotes', [])

        # Initialize seen chunk IDs with the original chunk
        seen_chunk_ids: Set[str] = {original_id}

        # Format quotes for the prompt
        clue_quotes_str = '\n'.join(f'<q>{q}</q>' for q in clue_quotes) if clue_quotes else 'N/A'
        item_quotes_str = '\n'.join(f'<q>{q}</q>' for q in item_quotes) if item_quotes else 'N/A'

        # Choose prompt based on whether this item contains truth
        if contains_truth:
            truth_quotes_str = '\n'.join(f'<q>{q}</q>' for q in truth_quotes) if truth_quotes else 'N/A'
            initial_msg = CHUNK_COLLECTOR_PROMPT_TRUTH_VER.format(
                question=question,
                truth=truth,
                clues=clues,
                clue_quotes=clue_quotes_str,
                chunk=chunk_content,
                item_quotes=item_quotes_str,
                truth_quotes=truth_quotes_str
            )
        else:
            initial_msg = CHUNK_COLLECTOR_PROMPT.format(
                question=question,
                truth=truth,
                clues=clues,
                clue_quotes=clue_quotes_str,
                chunk=chunk_content,
                item_quotes=item_quotes_str
            )

        input_messages = [{"role": "user", "content": initial_msg}]

        request_body = {
            "model": self.model,
            "system": CHUNK_COLLECTOR_SYSTEM_PROMPT,
            "max_tokens": 10000,
            "tools": COLLECTOR_TOOLS,
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        }

        final_output = ""

        for i in range(self.max_iterations):
            request_body["messages"] = input_messages
            response = self.anthropic_client.messages.create(**request_body)

            tool_use_items = [item for item in response.content if getattr(item, 'type', None) == 'tool_use']
            text_items = [item for item in response.content if getattr(item, 'type', None) == 'text']

            # If no tool calls, we have final output
            if not tool_use_items:
                for item in text_items:
                    if item.type == "text":
                        final_output += item.text
                break

            # Serialize assistant response
            serialized_items = []
            for item in response.content:
                serialized_item = item.model_dump(mode="python")
                if 'status' in serialized_item:
                    del serialized_item['status']
                serialized_items.append(serialized_item)

            input_messages.append({
                "role": "assistant",
                "content": serialized_items
            })

            # Process each tool call
            for tool_call in tool_use_items:
                tool_args = tool_call.input
                tool_name = tool_call.name

                result = f"Iteration {i+1}\n\n"
                new_chunk_ids = []

                if tool_name == "search":
                    formatted_results, new_chunk_ids = self._search(
                        all_chunks, seen_chunk_ids, tool_args["query"], company_search_engine
                    )
                    result += formatted_results
                elif tool_name == "grep":
                    formatted_results, new_chunk_ids = self._grep(
                        all_chunks, seen_chunk_ids, tool_args["pattern"]
                    )
                    result += formatted_results
                else:
                    result += f"Unknown tool: {tool_name}"

                # Update seen chunk IDs with newly returned chunks
                seen_chunk_ids.update(new_chunk_ids)

                tool_output_msg = {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result
                }
                input_messages.append({"role": "user", "content": [tool_output_msg]})
        else:
            # Hit max iterations, force output
            final_request = {
                "model": self.model,
                "system": CHUNK_COLLECTOR_SYSTEM_PROMPT,
                "max_tokens": 4000,
                "messages": input_messages + [{
                    "role": "user",
                    "content": "You have reached the maximum number of iterations. Based on all the chunks you have reviewed, output your findings now using the <additional_chunks> format. If you found no additional chunks, respond with <additional_chunks>None</additional_chunks>."
                }]
            }

            final_response = self.anthropic_client.messages.create(**final_request)
            for item in final_response.content:
                if getattr(item, 'type', None) == 'text':
                    final_output = item.text.strip()

        # Parse the output
        additional_chunks = parse_additional_chunks(final_output)

        # Filter out any chunks that were in seen_chunk_ids (shouldn't happen but safety check)
        # The original chunk should not appear in additional_chunks
        additional_chunks = [c for c in additional_chunks if c['id'] != original_id]

        return additional_chunks

    def run_single_file(self, filepath: str) -> Dict[str, Any]:
        """Process a single JSON file, running collector for each item in parallel.

        Processes the latest task by level that has passed_verification=True.
        For level > 0 tasks, also processes the bridging_item.

        Args:
            filepath: Path to the JSON file

        Returns:
            Dict with processing results
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Check for required data
        if "tasks" not in data or len(data["tasks"]) == 0:
            raise ValueError("No tasks found in file")

        # Get the latest task by level
        task = get_latest_task(data["tasks"])
        if task is None:
            raise ValueError("No tasks found in file")

        # Only process tasks that passed verification
        if not task.get("passed_verification", False):
            return {
                "filepath": filepath,
                "status": "skipped",
                "reason": "Task did not pass verification"
            }

        if "supporting_items" not in task or len(task["supporting_items"]) == 0:
            raise ValueError("No supporting_items found in task")

        # For level > 0 tasks, use new_company ticker; otherwise use original ticker
        level = task.get("level", 0)
        if level > 0 and task.get("new_company"):
            ticker = task["new_company"]
        else:
            ticker = data["ticker"]

        question = task.get("question", "")
        truth = task.get("truth", "")
        clues = task.get("clues", "")
        supporting_items = task["supporting_items"]
        bridging_item = task.get("bridging_item")
        items_and_contents = task.get("items_and_contents", {})

        # Pre-initialize shared resources ONCE for all items in this file
        # This avoids re-embedding all chunks for each item
        all_chunks = self._get_all_chunks(ticker)
        company_search_engine = CompanySearchEngine(
            collection_name=self.collection_name,
            ticker=ticker,
            chroma_cloud_client=self.chroma_client,
            openai_client=self.openai_client
        )

        # Determine number of workers based on items to process
        num_items = len(supporting_items) + (1 if bridging_item else 0)

        # Run collector for each item in parallel
        with ThreadPoolExecutor(max_workers=min(num_items, 4)) as executor:
            futures = {}

            # Submit supporting items
            for i, item in enumerate(supporting_items):
                item_id = item['id']
                chunk_content = items_and_contents.get(item_id, "")
                future = executor.submit(
                    self.run_single_item,
                    ticker,
                    question,
                    truth,
                    clues,
                    item,
                    chunk_content,
                    all_chunks,
                    company_search_engine
                )
                futures[future] = ('supporting', i)

            # Submit bridging item if present (level > 0 tasks)
            if bridging_item and bridging_item.get('id'):
                bridging_id = bridging_item['id']
                bridging_content = items_and_contents.get(bridging_id, "")
                # Create a temporary item dict with the same structure as supporting items
                bridging_as_item = {
                    'id': bridging_id,
                    'clue_quotes': bridging_item.get('clue_quotes', []),
                    'item_quotes': bridging_item.get('item_quotes', []),
                    'contains_truth': False,  # Bridging items don't contain the truth
                    'truth_quotes': []
                }
                future = executor.submit(
                    self.run_single_item,
                    ticker,
                    question,
                    truth,
                    clues,
                    bridging_as_item,
                    bridging_content,
                    all_chunks,
                    company_search_engine
                )
                futures[future] = ('bridging', 0)

            # Collect results
            supporting_results = [None] * len(supporting_items)
            bridging_result = None

            for future in as_completed(futures):
                item_type, idx = futures[future]
                if item_type == 'supporting':
                    supporting_results[idx] = future.result()
                else:
                    bridging_result = future.result()

        # Add additional_chunks to each supporting item
        for i, additional_chunks in enumerate(supporting_results):
            supporting_items[i]['additional_chunks'] = additional_chunks

        # Add additional_chunks to bridging item if processed
        if bridging_item and bridging_result is not None:
            bridging_item['additional_chunks'] = bridging_result

        # Collect all unique additional chunk IDs that need content
        all_additional_chunk_ids = set()
        for additional_chunks in supporting_results:
            for chunk in additional_chunks:
                chunk_id = chunk['id']
                if chunk_id not in items_and_contents:
                    all_additional_chunk_ids.add(chunk_id)

        # Also collect from bridging item
        if bridging_result:
            for chunk in bridging_result:
                chunk_id = chunk['id']
                if chunk_id not in items_and_contents:
                    all_additional_chunk_ids.add(chunk_id)

        # Fetch content for all additional chunks and update items_and_contents
        if all_additional_chunk_ids:
            # Query ChromaDB for these chunk IDs with retry logic
            chunk_ids_list = list(all_additional_chunk_ids)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    chunk_results = self.collection.get(ids=chunk_ids_list)
                    # Update items_and_contents with the fetched content
                    for chunk_id, document in zip(chunk_results['ids'], chunk_results['documents']):
                        items_and_contents[chunk_id] = document
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Log error but continue - some chunks might not be found
                        print(f"Warning: Could not fetch some additional chunk contents after {max_retries} retries: {e}")
                    else:
                        wait_time = 2 ** attempt
                        print(f"ChromaDB get error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(wait_time)

        # Write back to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Processed {filepath}")

        num_processed = len(supporting_items) + (1 if bridging_result is not None else 0)
        return {
            "filepath": filepath,
            "status": "success",
            "num_items_processed": num_processed,
            "has_bridging": bridging_item is not None
        }

    def is_processed(self, filepath: str) -> bool:
        """Check if a file has already been processed with additional_chunks on latest task's items."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            if "tasks" not in data or len(data["tasks"]) == 0:
                return False

            # Get the latest task by level
            task = get_latest_task(data["tasks"])
            if task is None:
                return False

            # If task didn't pass verification, consider it processed (we skip it)
            if not task.get("passed_verification", False):
                return True

            supporting_items = task.get("supporting_items", [])
            if len(supporting_items) == 0:
                return False

            # Check if all supporting items have additional_chunks field
            if not all("additional_chunks" in item for item in supporting_items):
                return False

            # For level > 0 tasks, also check bridging_item
            level = task.get("level", 0)
            if level > 0:
                bridging_item = task.get("bridging_item")
                if bridging_item and "additional_chunks" not in bridging_item:
                    return False

            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def has_valid_task(self, filepath: str, level_filter: Optional[int] = None) -> bool:
        """Check if file has a valid latest task with items that passed verification."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            if "tasks" not in data or len(data["tasks"]) == 0:
                return False

            # Get the latest task by level
            task = get_latest_task(data["tasks"])
            if task is None:
                return False

            # Must have passed verification
            if not task.get("passed_verification", False):
                return False

            # Must have question, truth, and supporting_items
            if not task.get("question"):
                return False
            if not task.get("truth"):
                return False

            supporting_items = task.get("supporting_items", [])
            if len(supporting_items) < 1:
                return False

            # Each supporting item must have required fields
            for item in supporting_items:
                if not item.get("id"):
                    return False
                if "clue_quotes" not in item:
                    return False
                if "item_quotes" not in item:
                    return False

            # For level > 0 tasks, also validate bridging_item
            level = task.get("level", 0)
            if level > 0:
                bridging_item = task.get("bridging_item")
                if not bridging_item:
                    return False
                if not bridging_item.get("id"):
                    return False
                if "clue_quotes" not in bridging_item:
                    return False
                if "item_quotes" not in bridging_item:
                    return False

            if level_filter is not None:
                if level != level_filter:
                    return False

            return True

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def _process_file(self, filepath: Path):
        """Process a single file with error handling."""
        try:
            result = self.run_single_file(str(filepath))
            return {"filepath": filepath, "success": True, "result": result}
        except Exception as e:
            return {"filepath": filepath, "success": False, "error": str(e)}

    def run_batch(self, input_dir: str, level_filter: Optional[int] = None, max_workers: int = 4) -> Dict[str, Any]:
        """Run collector on all valid files in a directory.

        Args:
            input_dir: Directory containing JSON files
            max_workers: Number of parallel file processors

        Returns:
            Summary dict with counts and errors
        """
        console = Console()
        input_path = Path(input_dir)

        if not input_path.exists():
            console.print(f"[red]Error: Directory {input_dir} does not exist[/red]")
            return {"total": 0, "valid": 0, "processed": 0, "successful": 0, "failed": 0, "errors": []}

        json_files = list(input_path.glob("*.json"))
        if not json_files:
            console.print(f"[red]No JSON files found in {input_dir}[/red]")
            return {"total": 0, "valid": 0, "processed": 0, "successful": 0, "failed": 0, "errors": []}

        # Filter for files with valid tasks
        valid_files = [f for f in json_files if self.has_valid_task(f, level_filter)]
        unprocessed_files = [f for f in valid_files if not self.is_processed(f)]

        console.print("[bold cyan]Chunk Collection Phase[/bold cyan]")
        console.print(f"Found {len(json_files)} total JSON files")
        console.print(f"With valid tasks: {len(valid_files)}")
        console.print(f"Already processed: {len(valid_files) - len(unprocessed_files)}")
        console.print(f"To process: {len(unprocessed_files)}")

        if not unprocessed_files:
            console.print("[green]All files already processed![/green]")
            return {
                "total": len(json_files),
                "valid": len(valid_files),
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "errors": []
            }

        console.print()

        successful = []
        failed = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=10,
        ) as progress:
            overall_task = progress.add_task(
                f"[bold blue]Collecting chunks for {len(unprocessed_files)} files",
                total=len(unprocessed_files)
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_file, filepath): filepath for filepath in unprocessed_files}

                for future in as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        successful.append(result["filepath"].name)
                    else:
                        failed.append({"file": result["filepath"].name, "error": result["error"]})
                    progress.advance(overall_task)

        console.print()
        console.print(f"[bold green]Successful:[/bold green] {len(successful)}")
        console.print(f"[bold red]Failed:[/bold red] {len(failed)}")

        if failed:
            console.print("\n[bold red]Failed files:[/bold red]")
            for item in failed:
                console.print(f"  {item['file']}: {item['error']}")

        return {
            "total": len(json_files),
            "valid": len(valid_files),
            "processed": len(unprocessed_files),
            "successful": len(successful),
            "failed": len(failed),
            "errors": failed
        }


def main():
    parser = argparse.ArgumentParser(
        description="Collect additional supporting chunks for SEC filing tasks"
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Path to input directory containing JSON files"
    )
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=4,
        help="Maximum number of parallel file workers (default: 4)"
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=15,
        help="Maximum iterations per chunk collector (default: 15)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="sec_test_1_14",
        help="ChromaDB collection name (default: sec_test_1_14)"
    )
    parser.add_argument(
        "--single-file", "-f",
        type=str,
        help="Process a single file instead of a directory"
    )
    parser.add_argument(
        "--level-filter", "-l",
        type=int,
        help="Level filter (default: None)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model for collection (default: claude-sonnet-4-5)"
    )

    args = parser.parse_args()

    console = Console()

    console.print("[bold]Initializing Chunk Collector Agent...[/bold]")
    console.print(f"Model: {args.model}")
    agent = ChunkCollectorAgent(
        collection_name=args.collection,
        max_iterations=args.max_iterations,
        model=args.model,
    )

    if args.single_file:
        console.print(f"Processing single file: {args.single_file}")
        result = agent.run_single_file(args.single_file)
        console.print("[bold green]Done![/bold green]")
        if result.get("status") == "skipped":
            console.print(f"[yellow]Skipped: {result.get('reason')}[/yellow]")
        else:
            console.print(f"Processed {result.get('num_items_processed', 0)} supporting items")
    else:
        result = agent.run_batch(args.input_dir, level_filter=args.level_filter, max_workers=args.max_workers)

        console.print()
        console.print("-" * 40)
        console.print(f"Total files: {result['total']}")
        console.print(f"Valid files: {result['valid']}")
        console.print(f"Processed: {result['processed']}")
        console.print(f"Successful: {result['successful']}")
        console.print(f"Failed: {result['failed']}")


if __name__ == "__main__":
    main()
