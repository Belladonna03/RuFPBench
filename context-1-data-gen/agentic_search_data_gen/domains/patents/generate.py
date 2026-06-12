from typing import Dict, Any, List
import os
from dotenv import load_dotenv
import json
import chromadb
from pathlib import Path
import tiktoken
import re
import ast
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
import xml.etree.ElementTree as ET
from .prompts import (
    EVAL_GEN_SYSTEM_PROMPT,
    EVAL_GEN_PROMPT
)
from ...core.utils import get_anthropic_client
import httpx

load_dotenv()


class EvalGenNew:
    def __init__(self, anthropic_client, model="claude-opus-4-5"):
        self.anthropic_client = anthropic_client
        self.model = model

    def has_extraction_result(self, filepath: str) -> bool:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return "extraction_result" in data and len(data["extraction_result"]["extracted"]) > 0
        except:
            return False

    def parse_task(self, xml_string):
        xml_string = xml_string.strip()
        if xml_string.startswith('```xml'):
            xml_string = xml_string[6:]
        if xml_string.startswith('```'):
            xml_string = xml_string[3:]
        if xml_string.endswith('```'):
            xml_string = xml_string[:-3]
        xml_string = xml_string.strip()

        root = ET.fromstring(xml_string)

        # If root element is 'task', return its text directly
        if root.tag == 'task':
            return root.text

        # Otherwise look for a child 'task' element
        task = root.findtext('task')
        return task

    def get_abstract_by_author(self, references, author_name):
        matches = []
        docid = None
        for k, v in references.items():
            for inventor in v["inventors"]:
                if author_name.lower() in inventor.lower():
                    if "Abstract" in v["abstract"]:
                        matches.append((k, v["abstract"].split("Abstract")[1]))
                    else:
                        matches.append((k, v["abstract"]))
                    docid = k

        if len(matches) == 0:
            return None, None
        elif len(matches) > 1:
            return None, None
        else:
            return matches[0][1], docid


    def run_single(self, extracted_item_info) -> Dict[str, Any]:
        extracted_item = extracted_item_info["extracted_item"]
        references = extracted_item_info["references"]

        all_authors = []

        for mapping in extracted_item["claim_element_mappings"]:
            for citation in mapping["citations"]:
                author_name = citation["name"].lower()
                if author_name not in all_authors:
                    all_authors.append(author_name)

        author_to_abstract = {}
        docids = []
        failed_authors = []

        for author in all_authors:
            abstract, docid = self.get_abstract_by_author(references, author)
            if abstract is None:
                failed_authors.append(author)
            else:
                author_to_abstract[author] = abstract
                docids.append(docid) # should be unique

        # If any author failed to resolve, skip this task
        if failed_authors:
            return {
                "skip": True,
                "error_info": {
                    "claim_number": extracted_item.get("claim_number"),
                    "failed_authors": failed_authors
                }
            }

        rejected_patent_claim = extracted_item["claim_text"]
        rejection_reasoning = extracted_item["reasoning"]

        rejection_details = ""
        for mapping in extracted_item["claim_element_mappings"]:
            rejection_details += f"Citations: {mapping['citations']}\n"
            rejection_details += f"Claim element: {mapping['claim_element']}\n"
            rejection_details += f"Prior art element: {mapping['prior_art_element']}\n"
            rejection_details += f"Mapping strength: {mapping['mapping_strength']}\n\n"
            rejection_details += "---\n"

        prior_art_abstracts = ""
        for author, abstract in author_to_abstract.items():
            prior_art_abstracts += f"{author.capitalize()}: {abstract}\n\n"

        formatted_prompt = EVAL_GEN_PROMPT.format(rejected_patent_claim=rejected_patent_claim, rejection_details=rejection_details, rejection_reasoning=rejection_reasoning, prior_art_abstracts=prior_art_abstracts)

        response = self.anthropic_client.messages.create(
            model=self.model,
            system=EVAL_GEN_SYSTEM_PROMPT,
            max_tokens=5000,
            messages=[{"role": "user", "content": formatted_prompt}],
            thinking={
                "type": "enabled",
                "budget_tokens": 2500
            }
        )
        text_items = [item for item in response.content if getattr(item, 'type', None) == 'text']
        text_only = text_items[0].text
        extracted = self.parse_task(text_only)
        return {
            "skip": False,
            "task": extracted,
            "positive_docids": list(set(docids))
        }


    def is_processed(self, extracted_item) -> bool:
        if "eval" in extracted_item:
            return True
        else:
            return False

    def _process_item(self, item_with_metadata):
        try:
            result = self.run_single(item_with_metadata["extracted_item_info"])
            return {
                "filepath": item_with_metadata["filepath"],
                "item_index": item_with_metadata["item_index"],
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "filepath": item_with_metadata["filepath"],
                "item_index": item_with_metadata["item_index"],
                "success": False,
                "error": str(e)
            }

    def run_batch(self, input_dir: str, max_workers: int = 8):
        console = Console()
        input_path = Path(input_dir)

        if not input_path.exists():
            console.print(f"[red]Error: Directory {input_dir} does not exist[/red]")
            return

        json_files = list(input_path.glob("*.json"))
        if not json_files:
            console.print(f"[red]No JSON files found in {input_dir}[/red]")
            return

        extracted_files = [f for f in json_files if self.has_extraction_result(f)]

        # Track items with their file and index
        items_with_metadata = []
        file_item_counts = {}  # Track how many unprocessed items per file
        all_items = 0

        for extracted_file in extracted_files:
            with open(extracted_file, "r") as f:
                data = json.load(f)
            extracted_items = data["extraction_result"]["extracted"]

            unprocessed_count = 0
            for idx, extracted_item in enumerate(extracted_items):
                all_items += 1
                if not self.is_processed(extracted_item):
                    info = {
                        "extracted_item": extracted_item,
                        "references": data["892"]["references"]
                    }
                    items_with_metadata.append({
                        "filepath": extracted_file,
                        "item_index": idx,
                        "extracted_item_info": info
                    })
                    unprocessed_count += 1

            if unprocessed_count > 0:
                file_item_counts[str(extracted_file)] = unprocessed_count

        console.print(f"Found {len(json_files)} total JSON files")
        console.print(f"Files with extraction result: {len(extracted_files)}")
        console.print(f"Already processed items: {all_items - len(items_with_metadata)}")
        console.print(f"To process items: {len(items_with_metadata)}")
        console.print(f"Files to update: {len(file_item_counts)}")

        if not items_with_metadata:
            console.print("[green]All items already processed![/green]")
            return

        console.print()

        # Track results by file
        from collections import defaultdict
        file_results = defaultdict(list)
        file_completed_counts = defaultdict(int)

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
                f"[bold blue]Generating eval tasks for {len(items_with_metadata)} items",
                total=len(items_with_metadata)
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_item, item): item for item in items_with_metadata}

                for future in as_completed(futures):
                    result = future.result()
                    filepath_str = str(result["filepath"])

                    # Collect result
                    file_results[filepath_str].append(result)
                    file_completed_counts[filepath_str] += 1

                    progress.advance(overall_task)

                    # Check if all items for this file are done
                    if file_completed_counts[filepath_str] == file_item_counts[filepath_str]:
                        self._write_file_results(result["filepath"], file_results[filepath_str], console)

        console.print()
        console.print(f"[bold green]Processing complete![/bold green]")

    def _write_file_results(self, filepath: Path, results: List[Dict], console: Console):
        """Write results for a single file once all its tasks are complete"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Initialize errored list if not present
            if "errored" not in data["extraction_result"]:
                data["extraction_result"]["errored"] = []

            successful_count = 0
            error_count = 0

            for result in results:
                idx = result["item_index"]
                extracted_item = data["extraction_result"]["extracted"][idx]

                if result["success"]:
                    result_data = result["result"]
                    if result_data.get("skip"):
                        # Add to errored list
                        data["extraction_result"]["errored"].append(result_data["error_info"])
                        error_count += 1
                    else:
                        # Add eval data to the item
                        extracted_item["eval"] = {
                            "task": result_data["task"],
                            "positive_docids": result_data["positive_docids"]
                        }
                        successful_count += 1
                else:
                    # Processing failed with exception
                    data["extraction_result"]["errored"].append({
                        "claim_number": extracted_item.get("claim_number"),
                        "error": result["error"]
                    })
                    error_count += 1

            # Write back to file
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)

            console.print(f"[green]Wrote {filepath.name}: {successful_count} successful, {error_count} errors[/green]")

        except Exception as e:
            console.print(f"[red]Failed to write {filepath.name}: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Generate eval tasks from USPTO JSON files")
    parser.add_argument("--input-dir", "-i", required=True, help="Path to input directory containing JSON files")
    parser.add_argument("--max-workers", "-w", type=int, default=6, help="Maximum number of parallel workers")
    parser.add_argument("--model", default="claude-opus-4-5", help="Anthropic model for generation (default: claude-opus-4-5)")
    args = parser.parse_args()

    console = Console()

    anthropic_client = get_anthropic_client()

    console.print("[bold]Starting Eval Generation [/bold]\n")
    eval_gen_agent = EvalGenNew(anthropic_client, model=args.model)
    eval_gen_agent.run_batch(args.input_dir, max_workers=args.max_workers)

    console.print("\n[bold green]Eval Generation complete![/bold green]")


if __name__ == "__main__":
    main()
