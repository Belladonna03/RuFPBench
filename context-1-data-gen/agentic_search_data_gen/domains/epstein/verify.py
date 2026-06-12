import os
import re
import json
import argparse
from glob import glob
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from dotenv import load_dotenv

from .prompts import EPSTEIN_EXTRACTION_PROMPT_SINGLE, EPSTEIN_COHERENCE_CHECK_PROMPT
from .utils import get_anthropic_client
from ...core.utils import parse_quotes
from ...core.verify import BaseVerifier

load_dotenv()


class EpsteinVerifier(BaseVerifier):
    """Epstein-specific verifier for thread-based verification."""

    def __init__(self, client=None, model: str = None, max_retries: int = 3):
        super().__init__(client=client, model=model, max_retries=max_retries)
        self.id_field = 'id'

    def run_single_item_extraction(
        self,
        clues: str,
        question: str,
        truth: str,
        thread_id: str,
        reasoning: str,
        content: str
    ) -> Dict[str, Any] | None:
        """Run extraction for a single supporting item."""
        prompt = EPSTEIN_EXTRACTION_PROMPT_SINGLE.format(
            clues=clues,
            question=question,
            truth=truth,
            thread_id=thread_id,
            reasoning=reasoning,
            content=content
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            thinking={"type": "enabled", "budget_tokens": 2000},
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = ""
        for item in response.content:
            if item.type == "text":
                response_text = item.text
                break

        return self.parse_single_item(response_text, id_tag='id')

    def process_single_item_with_retries(
        self,
        item: Dict,
        clues: str,
        question: str,
        truth: str,
        items_and_contents: Dict[str, str],
    ) -> Tuple[Dict[str, Any] | None, int, bool]:
        """Process a single supporting item with retries."""
        thread_id = item.get("id", "")
        reasoning = item.get("reasoning", "")
        content = items_and_contents.get(thread_id, "")

        if not content or content.startswith("Error:"):
            return None, 0, False

        extracted_item = None
        for attempt in range(self.max_retries):
            extracted_item = self.run_single_item_extraction(
                clues, question, truth, thread_id, reasoning, content
            )

            if not extracted_item:
                continue

            if extracted_item.get('not_relevant', False):
                return extracted_item, attempt, False

            verification = self.verify_supporting_item(extracted_item, clues, items_and_contents)
            item_valid = (
                verification['clue_quotes_valid'] and
                verification['item_quotes_valid'] and
                (not extracted_item.get('contains_truth', False) or verification['truth_quotes_valid'])
            )

            if item_valid:
                return extracted_item, attempt, True

        return extracted_item, self.max_retries, False

    def process_task(self, task: Dict) -> Tuple[List[Dict[str, Any]], int, bool]:
        """Process a task."""
        clues = task.get("clues", "")
        question = task.get("question", "")
        truth = task.get("truth", "")
        supporting_items = task.get("supporting_items", [])
        items_and_contents = task.get("items_and_contents", {})

        if not supporting_items or not items_and_contents:
            return [], 0, False

        def process_item(item):
            return self.process_single_item_with_retries(
                item, clues, question, truth, items_and_contents
            )

        return self.process_items_parallel(process_item, supporting_items)

    def run_coherence_check(
        self,
        clues: str,
        question: str,
        truth: str,
        supporting_items: List[Dict],
        items_and_contents: Dict[str, str],
    ) -> Tuple[bool, str]:
        """Run coherence check to verify supporting items connect logically."""
        # Format supporting items for the prompt
        formatted_items = []
        for i, item in enumerate(supporting_items, 1):
            item_id = item.get("id", "")
            content = items_and_contents.get(item_id, "")
            reasoning = item.get("reasoning", "")
            formatted_items.append(f"""<item_{i}>
    <id>{item_id}</id>
    <content>
{content}
    </content>
    <reasoning>{reasoning}</reasoning>
</item_{i}>""")

        supporting_items_formatted = "\n".join(formatted_items)

        prompt = EPSTEIN_COHERENCE_CHECK_PROMPT.format(
            clues=clues,
            question=question,
            truth=truth,
            supporting_items_formatted=supporting_items_formatted
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            thinking={"type": "enabled", "budget_tokens": 2000},
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = ""
        for item in response.content:
            if item.type == "text":
                response_text = item.text
                break

        # Parse the response
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL)
        coherent_match = re.search(r'<coherent>(.*?)</coherent>', response_text, re.DOTALL)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        is_coherent = False
        if coherent_match:
            coherent_str = coherent_match.group(1).strip().lower()
            is_coherent = coherent_str == 'true'

        return is_coherent, reasoning

    def process_file(self, filepath: str) -> Dict[str, Any]:
        """Process a file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        tasks = data.get("tasks", [])
        if not tasks:
            return {"status": "skipped", "reason": "no tasks", "passed_verification": None}

        updated = False
        passed_verification = None

        for task in tasks:
            if self.is_task_processed(task):
                continue

            try:
                extracted_items, retry_count, quotes_verified = self.process_task(task)

                if not extracted_items:
                    print(f"Warning: No items extracted for task in {filepath}")
                    task["passed_pre_verification"] = False
                    task["passed_verification"] = False
                    task["extraction_retry_count"] = self.max_retries
                    passed_verification = False
                    updated = True
                    continue

                task_updated = self.update_supporting_items(
                    task.get("supporting_items", []),
                    extracted_items,
                    task.get("items_and_contents", {})
                )

                task["extraction_retry_count"] = retry_count
                truth_contained = self.check_truth_contained(task.get("supporting_items", []))
                passed_pre_verification = quotes_verified and truth_contained
                task["passed_pre_verification"] = passed_pre_verification

                # Second verification step: coherence check
                if passed_pre_verification:
                    clues = task.get("clues", "")
                    question = task.get("question", "")
                    truth = task.get("truth", "")
                    supporting_items = task.get("supporting_items", [])
                    items_and_contents = task.get("items_and_contents", {})

                    is_coherent, coherence_reasoning = self.run_coherence_check(
                        clues, question, truth, supporting_items, items_and_contents
                    )
                    task["coherence_reasoning"] = coherence_reasoning
                    task["passed_verification"] = is_coherent
                else:
                    task["passed_verification"] = False
                    task["coherence_reasoning"] = ""

                passed_verification = task["passed_verification"]

                if task_updated:
                    updated = True

            except Exception as e:
                print(f"Error processing task in {filepath}: {e}")
                task["passed_pre_verification"] = False
                task["passed_verification"] = False
                task["extraction_retry_count"] = -1
                passed_verification = False
                updated = True
                continue

        if updated:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)

        return {"status": "success", "updated": updated, "passed_verification": passed_verification}

    def get_valid_files(self, input_dir: str) -> List[str]:
        """Get valid files from directory."""
        all_files = glob(os.path.join(input_dir, "*.json"))
        valid_files = []

        for filepath in all_files:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                tasks = data.get("tasks", [])
                if not tasks:
                    continue

                task = tasks[0]
                if not task.get("clues") or not task.get("question") or not task.get("truth"):
                    continue

                if not task.get("supporting_items") or not task.get("items_and_contents"):
                    continue

                if len(task.get("supporting_items", [])) != 3:
                    continue

                if len(task.get("items_and_contents", {})) != len(task.get("supporting_items", [])):
                    continue

                valid_files.append(filepath)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return valid_files

    def run_batch(self, input_dir: str, max_workers: int = 8) -> Dict[str, Any]:
        """Run batch processing."""
        valid_files = self.get_valid_files(input_dir)
        files_to_process = [f for f in valid_files if not self.is_file_fully_processed(f)]

        results = []
        errors = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"Processing {len(files_to_process)}/{len(valid_files)} files",
                total=len(files_to_process)
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_file, f): f
                    for f in files_to_process
                }

                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        result = future.result()
                        results.append({"file": filepath, **result})
                    except Exception as e:
                        errors.append({"file": filepath, "error": str(e)})
                    progress.advance(task)

        passed_count = sum(1 for r in results if r.get("passed_verification") is True)
        failed_verification_count = sum(1 for r in results if r.get("passed_verification") is False)

        return {
            "total_valid": len(valid_files),
            "processed": len(files_to_process),
            "skipped": len(valid_files) - len(files_to_process),
            "successful": len(results),
            "failed": len(errors),
            "passed_verification": passed_count,
            "failed_verification": failed_verification_count,
            "errors": errors
        }


def main():
    parser = argparse.ArgumentParser(description="Verify and extract quotes from supporting items.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing JSON files")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Maximum number of parallel workers (default: 8)")
    parser.add_argument("--max-retries", "-r", type=int, default=3, help="Maximum extraction retries on verification failure (default: 3)")
    parser.add_argument("--model", "-m", type=str, default="claude-opus-4-5", help="Model to use (default: claude-opus-4-5)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        exit(1)

    print(f"Input directory: {args.input}")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max retries: {args.max_retries}")
    print("-" * 40)

    client = get_anthropic_client()
    verifier = EpsteinVerifier(client=client, model=args.model, max_retries=args.max_retries)
    result = verifier.run_batch(args.input, max_workers=args.max_workers)

    print("-" * 40)
    print(f"Total valid files: {result['total_valid']}")
    print(f"Skipped (already processed): {result['skipped']}")
    print(f"Processed: {result['processed']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    print(f"Passed verification: {result['passed_verification']}")
    print(f"Failed verification: {result['failed_verification']}")
    if result['errors']:
        print("\nErrors:")
        for err in result['errors']:
            print(f"  {err['file']}: {err['error']}")


if __name__ == "__main__":
    main()
