import os
import re
import json
import argparse
from glob import glob
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from dotenv import load_dotenv

from .prompts import (
    WEB_EXTRACTION_PROMPT_SINGLE,
    WEB_BRIDGING_EXTRACTION_PROMPT_SINGLE,
    WEB_DISTRACTION_EXTRACTION_PROMPT_SINGLE,
    WEB_DISTRACTION_EXTRACTION_PROMPT_MULTIPLE,
)
from ...core.utils import (
    count_matching_quotes,
    min_required_matches,
    parse_quotes,
    get_anthropic_client
)
from ...core.verify import BaseVerifier

load_dotenv()


class WebVerifier(BaseVerifier):
    """Web-specific verifier with support for bridging items and extension tasks."""

    def __init__(self, client=None, model: str = None, max_retries: int = 3):
        super().__init__(client=client, model=model, max_retries=max_retries)
        self.id_field = 'id'

    def normalize_task_items(self, task: Dict) -> None:
        """Normalize all items in a task in place (convert 'url' -> 'id')."""
        # Normalize supporting_items
        for item in task.get("supporting_items", []):
            if 'url' in item and 'id' not in item:
                item['id'] = item.pop('url')

        # Normalize bridging_item
        bridging = task.get("bridging_item", {})
        if bridging and 'url' in bridging and 'id' not in bridging:
            bridging['id'] = bridging.pop('url')

    def build_failure_reasons(
        self,
        quotes_verified: bool,
        truth_contained: bool,
        verification_results: List[Dict[str, bool]] = None,
        bridging_verification: Dict[str, bool] = None,
        has_not_relevant_items: bool = False,
        has_bridging_not_relevant: bool = False,
        custom_reason: str = None
    ) -> List[str]:
        """Build a list of reasons explaining why verification failed."""
        reasons = []

        if custom_reason:
            reasons.append(custom_reason)
            return reasons

        if has_not_relevant_items:
            reasons.append("Model indicated supporting item content is not relevant to the clues")

        if has_bridging_not_relevant:
            reasons.append("Model indicated bridging item content is not relevant")

        if not quotes_verified and verification_results:
            for i, result in enumerate(verification_results):
                item_reasons = []
                if not result.get('clue_quotes_valid', True):
                    item_reasons.append("clue quotes not found in clues text")
                if not result.get('item_quotes_valid', True):
                    item_reasons.append("item quotes not found in page content")
                if not result.get('truth_quotes_valid', True):
                    item_reasons.append("truth quotes not found in page content")
                if item_reasons:
                    reasons.append(f"Supporting item {i+1}: {', '.join(item_reasons)}")

        if not quotes_verified and bridging_verification:
            bridging_reasons = []
            if not bridging_verification.get('item_clue_quotes_valid', True):
                bridging_reasons.append("bridging item clue quotes not found in clues")
            if not bridging_verification.get('item_quotes_valid', True):
                bridging_reasons.append("bridging item quotes not found in page content")
            if not bridging_verification.get('prev_item_clue_quotes_valid', True):
                bridging_reasons.append("previous item clue quotes not found in previous clues")
            if not bridging_verification.get('prev_item_quotes_valid', True):
                bridging_reasons.append("previous item quotes not found in previous page content")
            if bridging_reasons:
                reasons.append(f"Bridging item: {', '.join(bridging_reasons)}")

        if not truth_contained:
            reasons.append("No supporting item contains verifiable truth quotes")

        return reasons

    def has_not_relevant(self, extracted_items: List[Dict[str, Any]]) -> bool:
        """Check if any extracted item has not_relevant=True."""
        return any(item.get('not_relevant', False) for item in extracted_items)

    def has_bridging_not_relevant(self, bridging_item: Dict[str, Any]) -> bool:
        """Check if bridging item or prev_item has not_relevant=True."""
        item = bridging_item.get('item', {})
        prev_item = bridging_item.get('prev_item', {})
        return item.get('not_relevant', False) or prev_item.get('not_relevant', False)

    def parse_supporting_items(self, content: str) -> List[Dict[str, Any]]:
        """Parse supporting items from LLM response."""
        items = []
        outer_match = re.search(r'<supporting_items>(.*?)</supporting_items>', content, re.DOTALL)
        if not outer_match:
            return items

        items_content = outer_match.group(1)
        item_matches = re.findall(r'<item>(.*?)</item>', items_content, re.DOTALL)

        for item_match in item_matches:
            url_match = re.search(r'<url>(.*?)</url>', item_match, re.DOTALL)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_match, re.DOTALL)
            contains_truth_match = re.search(r'<contains_truth>(.*?)</contains_truth>', item_match, re.DOTALL)

            clue_quotes = parse_quotes(item_match, 'clue_quotes')
            item_quotes = parse_quotes(item_match, 'item_quotes')
            truth_quotes = parse_quotes(item_match, 'truth_quotes')

            contains_truth = False
            if contains_truth_match:
                contains_truth_str = contains_truth_match.group(1).strip().lower()
                contains_truth = contains_truth_str == 'true'

            not_relevant = clue_quotes is None or item_quotes is None

            items.append({
                'id': url_match.group(1).strip() if url_match else '',
                'clue_quotes': clue_quotes if clue_quotes is not None else [],
                'item_quotes': item_quotes if item_quotes is not None else [],
                'reasoning': reasoning_match.group(1).strip() if reasoning_match else '',
                'contains_truth': contains_truth,
                'truth_quotes': truth_quotes if truth_quotes is not None else [],
                'not_relevant': not_relevant
            })

        return items

    def format_supporting_items_for_prompt(
        self,
        supporting_items: List[Dict],
        items_and_contents: Dict[str, str]
    ) -> str:
        """Format supporting items for prompt."""
        formatted = ""
        for item in supporting_items:
            url = item.get(self.id_field, "")
            reasoning = item.get("reasoning", "")
            content = items_and_contents.get(url, "")

            formatted += f"""    <item>
        <url>{url}</url>
        <reasoning>{reasoning}</reasoning>
        <content>
{content}
        </content>
    </item>
"""
        return formatted.strip()

    def format_bridging_item_for_prompt(
        self,
        bridging_item: Dict,
        items_and_contents: Dict[str, str]
    ) -> str:
        """Format bridging_item with its content for the prompt."""
        url = bridging_item.get(self.id_field, "")
        relevant_prev_url = bridging_item.get("relevant_prev_url", "")
        reasoning = bridging_item.get("reasoning", "")
        content = items_and_contents.get(url, "")

        return f"""<url>{url}</url>
<relevant_prev_url>{relevant_prev_url}</relevant_prev_url>
<reasoning>{reasoning}</reasoning>
<content>
{content}
</content>"""

    def format_previous_supporting_item_for_prompt(
        self,
        prev_task: Dict,
        relevant_prev_url: str
    ) -> str:
        """Format the previous task's supporting item that contains the bridging connection."""
        items_and_contents = prev_task.get("items_and_contents", {})
        content = items_and_contents.get(relevant_prev_url, "")
        prev_clues = prev_task.get("clues", "")

        return f"""<url>{relevant_prev_url}</url>
<previous_clues>
{prev_clues}
</previous_clues>
<content>
{content}
</content>"""

    def parse_bridging_item_from_response(self, content: str) -> Dict[str, Any]:
        """Parse bridging_item from extraction response."""
        outer_match = re.search(r'<bridging_item>(.*?)</bridging_item>', content, re.DOTALL)
        if not outer_match:
            return {}

        bridging_content = outer_match.group(1)

        item_match = re.search(r'<item>(.*?)</item>', bridging_content, re.DOTALL)
        item_data = {}
        if item_match:
            item_content = item_match.group(1)
            url_match = re.search(r'<url>(.*?)</url>', item_content, re.DOTALL)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_content, re.DOTALL)

            clue_quotes = parse_quotes(item_content, 'clue_quotes')
            item_quotes = parse_quotes(item_content, 'item_quotes')

            not_relevant = clue_quotes is None or item_quotes is None

            item_data = {
                'id': url_match.group(1).strip() if url_match else '',
                'clue_quotes': clue_quotes if clue_quotes is not None else [],
                'item_quotes': item_quotes if item_quotes is not None else [],
                'reasoning': reasoning_match.group(1).strip() if reasoning_match else '',
                'not_relevant': not_relevant
            }

        prev_item_match = re.search(r'<prev_item>(.*?)</prev_item>', bridging_content, re.DOTALL)
        prev_item_data = {}
        if prev_item_match:
            prev_item_content = prev_item_match.group(1)
            prev_url_match = re.search(r'<relevant_prev_url>(.*?)</relevant_prev_url>', prev_item_content, re.DOTALL)
            prev_reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', prev_item_content, re.DOTALL)

            clue_quotes = parse_quotes(prev_item_content, 'clue_quotes')
            prev_item_quotes = parse_quotes(prev_item_content, 'prev_item_quotes')

            not_relevant = clue_quotes is None or prev_item_quotes is None

            prev_item_data = {
                'relevant_prev_url': prev_url_match.group(1).strip() if prev_url_match else '',
                'clue_quotes': clue_quotes if clue_quotes is not None else [],
                'prev_item_quotes': prev_item_quotes if prev_item_quotes is not None else [],
                'reasoning': prev_reasoning_match.group(1).strip() if prev_reasoning_match else '',
                'not_relevant': not_relevant
            }

        return {
            'item': item_data,
            'prev_item': prev_item_data
        }

    def verify_bridging_item(
        self,
        bridging_item: Dict[str, Any],
        clues: str,
        items_and_contents: Dict[str, str],
        prev_clues: str = "",
        prev_items_and_contents: Dict[str, str] = None
    ) -> Dict[str, bool]:
        """Verify quotes for a bridging item (no truth check)."""
        if prev_items_and_contents is None:
            prev_items_and_contents = {}

        result = {
            'item_clue_quotes_valid': True,
            'item_quotes_valid': True,
            'prev_item_clue_quotes_valid': True,
            'prev_item_quotes_valid': True
        }

        item = bridging_item.get('item', {})
        prev_item = bridging_item.get('prev_item', {})

        item_clue_quotes = item.get('clue_quotes', [])
        if item_clue_quotes:
            matches = count_matching_quotes(item_clue_quotes, clues)
            required = min_required_matches(len(item_clue_quotes))
            result['item_clue_quotes_valid'] = matches >= required

        item_quotes = item.get('item_quotes', [])
        url = item.get('id', '') or item.get('url', '')
        content = items_and_contents.get(url, '')

        if item_quotes and content and not content.startswith("Error"):
            matches = count_matching_quotes(item_quotes, content)
            required = min_required_matches(len(item_quotes))
            result['item_quotes_valid'] = matches >= required

        prev_clue_quotes = prev_item.get('clue_quotes', [])
        if prev_clue_quotes:
            # Check prev_item's clue_quotes against both current task's clues and previous task's clues
            combined_clues = clues + "\n" + prev_clues if prev_clues else clues
            matches = count_matching_quotes(prev_clue_quotes, combined_clues)
            required = min_required_matches(len(prev_clue_quotes))
            result['prev_item_clue_quotes_valid'] = matches >= required

        prev_item_quotes = prev_item.get('prev_item_quotes', [])
        prev_url = prev_item.get('relevant_prev_url', '')
        prev_content = prev_items_and_contents.get(prev_url, '')

        if prev_item_quotes and prev_content and not prev_content.startswith("Error"):
            matches = count_matching_quotes(prev_item_quotes, prev_content)
            required = min_required_matches(len(prev_item_quotes))
            result['prev_item_quotes_valid'] = matches >= required

        return result

    def get_tasks_by_level(self, tasks: List[Dict]) -> List[Dict]:
        """Sort tasks by their level value (ascending)."""
        return sorted(tasks, key=lambda t: t.get("level", 0))

    def get_task_to_verify(self, tasks: List[Dict]) -> Tuple[Dict, Dict]:
        """Find the task to verify: last task (by level) without passed_verification."""
        sorted_tasks = self.get_tasks_by_level(tasks)

        for i in range(len(sorted_tasks) - 1, -1, -1):
            task = sorted_tasks[i]
            if "passed_verification" not in task:
                prev_task = sorted_tasks[i - 1] if i > 0 else None
                return task, prev_task

        return None, None

    def run_single_item_extraction(
        self,
        clues: str,
        question: str,
        truth: str,
        url: str,
        reasoning: str,
        content: str
    ) -> Dict[str, Any] | None:
        """Run extraction for a single supporting item."""
        prompt = WEB_EXTRACTION_PROMPT_SINGLE.format(
            clues=clues,
            question=question,
            truth=truth,
            url=url,
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

        return self.parse_single_item(response_text, id_tag='url')

    def run_bridging_extraction(
        self,
        clues: str,
        question: str,
        truth: str,
        previous_truth: str,
        bridging_url: str,
        bridging_reasoning: str,
        bridging_content: str,
        prev_url: str,
        previous_clues: str,
        prev_content: str
    ) -> Dict[str, Any]:
        """Run extraction for bridging item."""
        prompt = WEB_BRIDGING_EXTRACTION_PROMPT_SINGLE.format(
            clues=clues,
            question=question,
            truth=truth,
            previous_truth=previous_truth,
            bridging_url=bridging_url,
            bridging_reasoning=bridging_reasoning,
            bridging_content=bridging_content,
            prev_url=prev_url,
            previous_clues=previous_clues,
            prev_content=prev_content
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

        return self.parse_bridging_item_from_response(response_text)

    def verify_extension_items(
        self,
        bridging_item: Dict[str, Any],
        supporting_items: List[Dict[str, Any]],
        clues: str,
        items_and_contents: Dict[str, str],
        prev_clues: str = "",
        prev_items_and_contents: Dict[str, str] = None
    ) -> Tuple[bool, Dict[str, bool], List[Dict[str, bool]]]:
        """Verify bridging_item and supporting_items for an extension task."""
        if prev_items_and_contents is None:
            prev_items_and_contents = {}

        all_valid = True

        bridging_verification = self.verify_bridging_item(
            bridging_item, clues, items_and_contents, prev_clues, prev_items_and_contents
        )

        if not bridging_verification['item_clue_quotes_valid']:
            all_valid = False
        if not bridging_verification['item_quotes_valid']:
            all_valid = False
        if not bridging_verification['prev_item_clue_quotes_valid']:
            all_valid = False
        if not bridging_verification['prev_item_quotes_valid']:
            all_valid = False

        supporting_verifications = []
        for item in supporting_items:
            verification = self.verify_supporting_item(item, clues, items_and_contents)
            supporting_verifications.append(verification)

            if not verification['clue_quotes_valid']:
                all_valid = False
            if not verification['item_quotes_valid']:
                all_valid = False
            if item.get('contains_truth', False) and not verification['truth_quotes_valid']:
                all_valid = False

        return all_valid, bridging_verification, supporting_verifications

    def process_bridging_with_retries(
        self,
        clues: str,
        question: str,
        truth: str,
        previous_truth: str,
        bridging_item: Dict,
        items_and_contents: Dict[str, str],
        previous_clues: str,
        prev_items_and_contents: Dict[str, str],
    ) -> Tuple[Dict[str, Any], int, bool]:
        """Process bridging item with retries."""
        bridging_url = bridging_item.get(self.id_field, "")
        bridging_reasoning = bridging_item.get("reasoning", "")
        bridging_content = items_and_contents.get(bridging_url, "")

        prev_url = bridging_item.get("relevant_prev_url", "")
        prev_content = prev_items_and_contents.get(prev_url, "")

        if not bridging_content or bridging_content.startswith("Error"):
            return {}, 0, False

        extracted_bridging = {}
        for attempt in range(self.max_retries):
            extracted_bridging = self.run_bridging_extraction(
                clues, question, truth, previous_truth,
                bridging_url, bridging_reasoning, bridging_content,
                prev_url, previous_clues, prev_content
            )

            if not extracted_bridging:
                continue

            if self.has_bridging_not_relevant(extracted_bridging):
                return extracted_bridging, attempt, False

            verification = self.verify_bridging_item(
                extracted_bridging, clues, items_and_contents,
                previous_clues, prev_items_and_contents
            )

            bridging_valid = (
                verification['item_clue_quotes_valid'] and
                verification['item_quotes_valid'] and
                verification['prev_item_clue_quotes_valid'] and
                verification['prev_item_quotes_valid']
            )

            if bridging_valid:
                return extracted_bridging, attempt, True

        return extracted_bridging, self.max_retries, False

    def process_extension_task(
        self,
        task: Dict,
        prev_task: Dict,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int, bool]:
        """Process an extension task (task with level > 0)."""
        clues = task.get("clues", "")
        question = task.get("question", "")
        truth = task.get("truth", "")
        bridging_item = task.get("bridging_item", {})
        supporting_items = task.get("supporting_items", [])
        items_and_contents = task.get("items_and_contents", {})

        previous_truth = prev_task.get("truth", "")
        previous_clues = prev_task.get("clues", "")
        prev_items_and_contents = prev_task.get("items_and_contents", {})

        relevant_prev_url = bridging_item.get("relevant_prev_url", "")
        if relevant_prev_url not in prev_items_and_contents:
            print(f"Warning: relevant_prev_url '{relevant_prev_url}' not found in previous task's items_and_contents")
            return {}, [], 0, False

        if not supporting_items or not items_and_contents:
            return {}, [], 0, False

        def process_bridging():
            return self.process_bridging_with_retries(
                clues, question, truth, previous_truth,
                bridging_item, items_and_contents, previous_clues,
                prev_items_and_contents
            )

        def process_item(item):
            return self.process_single_item_with_retries(
                item, clues, question, truth, items_and_contents
            )

        with ThreadPoolExecutor(max_workers=len(supporting_items) + 1) as executor:
            bridging_future = executor.submit(process_bridging)
            item_futures = [executor.submit(process_item, item) for item in supporting_items]

            extracted_bridging, bridging_retries, bridging_valid = bridging_future.result()
            item_results = [f.result() for f in item_futures]

        extracted_items = []
        total_retries = bridging_retries
        all_valid = bridging_valid

        for extracted_item, retries, valid in item_results:
            total_retries = max(total_retries, retries)

            if extracted_item:
                extracted_items.append(extracted_item)
            else:
                all_valid = False

            if not valid:
                all_valid = False

        return extracted_bridging, extracted_items, total_retries, all_valid

    def update_bridging_item(
        self,
        original_item: Dict,
        extracted_item: Dict[str, Any]
    ) -> bool:
        """Update original bridging_item with extracted quotes."""
        if not extracted_item:
            return False

        item = extracted_item.get("item", {})
        prev_item = extracted_item.get("prev_item", {})

        original_item["clue_quotes"] = item.get("clue_quotes", [])
        original_item["item_quotes"] = item.get("item_quotes", [])
        original_item["reasoning"] = item.get("reasoning", "")
        if "not_relevant" in item:
            original_item["not_relevant"] = item["not_relevant"]

        original_item["prev_item"] = {
            "relevant_prev_url": prev_item.get("relevant_prev_url", ""),
            "clue_quotes": prev_item.get("clue_quotes", []),
            "prev_item_quotes": prev_item.get("prev_item_quotes", []),
            "reasoning": prev_item.get("reasoning", "")
        }
        if "not_relevant" in prev_item:
            original_item["prev_item"]["not_relevant"] = prev_item["not_relevant"]

        return True

    def process_single_item_with_retries(
        self,
        item: Dict,
        clues: str,
        question: str,
        truth: str,
        items_and_contents: Dict[str, str],
    ) -> Tuple[Dict[str, Any] | None, int, bool]:
        """Process a single supporting item with retries."""
        url = item.get(self.id_field, "")
        reasoning = item.get("reasoning", "")
        content = items_and_contents.get(url, "")

        if not content or content.startswith("Error"):
            return None, 0, False

        extracted_item = None
        for attempt in range(self.max_retries):
            extracted_item = self.run_single_item_extraction(
                clues, question, truth, url, reasoning, content
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

    def process_file(self, filepath: str) -> Dict[str, Any]:
        """Process a file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        tasks = data.get("tasks", [])
        if not tasks:
            return {"status": "skipped", "reason": "no tasks", "passed_verification": None}

        # Normalize all task items in place (convert 'url' -> 'id')
        for task in tasks:
            self.normalize_task_items(task)

        updated = False
        passed_verification = None

        if len(tasks) == 1:
            task = tasks[0]
            if not self.is_task_processed(task):
                try:
                    extracted_items, retry_count, quotes_verified = self.process_task(task)

                    if not extracted_items:
                        if retry_count == 0:
                            # No LLM calls were made (process failure) - don't write to JSON so it can be reprocessed
                            print(f"Warning: Process failure for task in {filepath} - no LLM calls made (likely missing content). File will not be updated.")
                            passed_verification = None
                            # Don't set updated = True, so the file won't be written
                        else:
                            # LLM was called but extraction failed
                            print(f"Warning: No items extracted for task in {filepath}")
                            task["passed_verification"] = False
                            task["failure_reasons"] = ["No quotes could be extracted from the supporting items"]
                            task["extraction_retry_count"] = retry_count
                            passed_verification = False
                            updated = True
                    else:
                        task_updated = self.update_supporting_items(
                            task.get("supporting_items", []),
                            extracted_items,
                            task.get("items_and_contents", {}),
                        )

                        task["extraction_retry_count"] = retry_count
                        truth_contained = self.check_truth_contained(task.get("supporting_items", []))
                        task["passed_verification"] = quotes_verified and truth_contained
                        passed_verification = task["passed_verification"]

                        if not passed_verification:
                            _, verification_results = self.verify_all_items(
                                extracted_items,
                                task.get("clues", ""),
                                task.get("items_and_contents", {})
                            )
                            task["failure_reasons"] = self.build_failure_reasons(
                                quotes_verified=quotes_verified,
                                truth_contained=truth_contained,
                                verification_results=verification_results,
                                has_not_relevant_items=self.has_not_relevant(extracted_items)
                            )

                        if task_updated:
                            updated = True

                except Exception as e:
                    print(f"Error processing task in {filepath}: {e}")
                    task["passed_verification"] = False
                    task["failure_reasons"] = [f"Exception during processing: {str(e)}"]
                    task["extraction_retry_count"] = -1
                    passed_verification = False
                    updated = True

        else:
            task_to_verify, prev_task = self.get_task_to_verify(tasks)

            if task_to_verify is not None:
                try:
                    bridging_item = task_to_verify.get("bridging_item", {})
                    is_extension = bool(bridging_item) and prev_task is not None

                    if is_extension:
                        relevant_prev_url = bridging_item.get("relevant_prev_url", "")
                        prev_items_and_contents = prev_task.get("items_and_contents", {})

                        if relevant_prev_url not in prev_items_and_contents:
                            print(f"Warning: relevant_prev_url '{relevant_prev_url}' not in previous task in {filepath}")
                            task_to_verify["passed_verification"] = False
                            task_to_verify["failure_reasons"] = [f"Bridging URL '{relevant_prev_url}' not found in previous task's items"]
                            task_to_verify["extraction_retry_count"] = 0
                            passed_verification = False
                            updated = True
                        else:
                            extracted_bridging, extracted_items, retry_count, quotes_verified = self.process_extension_task(
                                task_to_verify, prev_task
                            )

                            if not extracted_items:
                                if retry_count == 0:
                                    # No LLM calls were made (process failure) - don't write to JSON so it can be reprocessed
                                    print(f"Warning: Process failure for extension task in {filepath} - no LLM calls made (likely missing content). File will not be updated.")
                                    passed_verification = None
                                    # Don't set updated = True, so the file won't be written
                                else:
                                    # LLM was called but extraction failed
                                    print(f"Warning: No items extracted for extension task in {filepath}")
                                    task_to_verify["passed_verification"] = False
                                    task_to_verify["failure_reasons"] = ["No quotes could be extracted from the supporting items"]
                                    task_to_verify["extraction_retry_count"] = retry_count
                                    passed_verification = False
                                    updated = True
                            else:
                                self.update_bridging_item(
                                    task_to_verify.get("bridging_item", {}),
                                    extracted_bridging
                                )

                                self.update_supporting_items(
                                    task_to_verify.get("supporting_items", []),
                                    extracted_items,
                                    task_to_verify.get("items_and_contents", {}),
                                )

                                task_to_verify["extraction_retry_count"] = retry_count
                                truth_contained = self.check_truth_contained(task_to_verify.get("supporting_items", []))
                                task_to_verify["passed_verification"] = quotes_verified and truth_contained
                                passed_verification = task_to_verify["passed_verification"]

                                if not passed_verification:
                                    _, bridging_verif, supporting_verifs = self.verify_extension_items(
                                        extracted_bridging,
                                        extracted_items,
                                        task_to_verify.get("clues", ""),
                                        task_to_verify.get("items_and_contents", {}),
                                        prev_task.get("clues", ""),
                                        prev_items_and_contents
                                    )
                                    task_to_verify["failure_reasons"] = self.build_failure_reasons(
                                        quotes_verified=quotes_verified,
                                        truth_contained=truth_contained,
                                        verification_results=supporting_verifs,
                                        bridging_verification=bridging_verif,
                                        has_not_relevant_items=self.has_not_relevant(extracted_items),
                                        has_bridging_not_relevant=self.has_bridging_not_relevant(extracted_bridging)
                                    )

                                updated = True

                    else:
                        extracted_items, retry_count, quotes_verified = self.process_task(task_to_verify)

                        if not extracted_items:
                            if retry_count == 0:
                                # No LLM calls were made (process failure) - don't write to JSON so it can be reprocessed
                                print(f"Warning: Process failure for task in {filepath} - no LLM calls made (likely missing content). File will not be updated.")
                                passed_verification = None
                                # Don't set updated = True, so the file won't be written
                            else:
                                # LLM was called but extraction failed
                                print(f"Warning: No items extracted for task in {filepath}")
                                task_to_verify["passed_verification"] = False
                                task_to_verify["failure_reasons"] = ["No quotes could be extracted from the supporting items"]
                                task_to_verify["extraction_retry_count"] = retry_count
                                passed_verification = False
                                updated = True
                        else:
                            self.update_supporting_items(
                                task_to_verify.get("supporting_items", []),
                                extracted_items,
                                task_to_verify.get("items_and_contents", {}),
                            )

                            task_to_verify["extraction_retry_count"] = retry_count
                            truth_contained = self.check_truth_contained(task_to_verify.get("supporting_items", []))
                            task_to_verify["passed_verification"] = quotes_verified and truth_contained
                            passed_verification = task_to_verify["passed_verification"]

                            if not passed_verification:
                                _, verification_results = self.verify_all_items(
                                    extracted_items,
                                    task_to_verify.get("clues", ""),
                                    task_to_verify.get("items_and_contents", {})
                                )
                                task_to_verify["failure_reasons"] = self.build_failure_reasons(
                                    quotes_verified=quotes_verified,
                                    truth_contained=truth_contained,
                                    verification_results=verification_results,
                                    has_not_relevant_items=self.has_not_relevant(extracted_items)
                                )

                            updated = True

                except Exception as e:
                    print(f"Error processing task in {filepath}: {e}")
                    task_to_verify["passed_verification"] = False
                    task_to_verify["failure_reasons"] = [f"Exception during processing: {str(e)}"]
                    task_to_verify["extraction_retry_count"] = -1
                    passed_verification = False
                    updated = True

        if updated:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)

        return {"status": "success", "updated": updated, "passed_verification": passed_verification}

    # ==================== Distractor Verification Methods ====================

    def parse_distractor_verification_single(self, content: str) -> Dict[str, Any]:
        """Parse response from single truth distractor verification."""
        item_match = re.search(r'<item>(.*?)</item>', content, re.DOTALL)
        if not item_match:
            return {}

        item_content = item_match.group(1)
        url_match = re.search(r'<url>(.*?)</url>', item_content, re.DOTALL)
        contains_truth_match = re.search(r'<contains_truth>(.*?)</contains_truth>', item_content, re.DOTALL)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_content, re.DOTALL)
        truth_quotes = parse_quotes(item_content, 'truth_quotes')

        contains_truth = False
        if contains_truth_match:
            contains_truth_str = contains_truth_match.group(1).strip().lower()
            contains_truth = contains_truth_str == 'true'

        return {
            'url': url_match.group(1).strip() if url_match else '',
            'contains_truth': contains_truth,
            'truth_quotes': truth_quotes if truth_quotes else [],
            'reasoning': reasoning_match.group(1).strip() if reasoning_match else ''
        }

    def parse_distractor_verification_multiple(self, content: str) -> Dict[str, Any]:
        """Parse response from multiple truths distractor verification."""
        item_match = re.search(r'<item>(.*?)</item>', content, re.DOTALL)
        if not item_match:
            return {}

        item_content = item_match.group(1)
        url_match = re.search(r'<url>(.*?)</url>', item_content, re.DOTALL)
        contains_truth_match = re.search(r'<contains_truth>(.*?)</contains_truth>', item_content, re.DOTALL)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_content, re.DOTALL)

        contains_truth = False
        if contains_truth_match:
            contains_truth_str = contains_truth_match.group(1).strip().lower()
            contains_truth = contains_truth_str == 'true'

        # Parse matched truths
        matched_truths = []
        matched_truths_match = re.search(r'<matched_truths>(.*?)</matched_truths>', item_content, re.DOTALL)
        if matched_truths_match:
            matched_content = matched_truths_match.group(1)
            if 'None' not in matched_content:
                match_blocks = re.findall(r'<match>(.*?)</match>', matched_content, re.DOTALL)
                for block in match_blocks:
                    truth_match = re.search(r'<truth>(.*?)</truth>', block, re.DOTALL)
                    quote_match = re.search(r'<quote>(.*?)</quote>', block, re.DOTALL)
                    if truth_match and quote_match:
                        matched_truths.append({
                            'truth': truth_match.group(1).strip(),
                            'quote': quote_match.group(1).strip()
                        })

        return {
            'url': url_match.group(1).strip() if url_match else '',
            'contains_truth': contains_truth,
            'matched_truths': matched_truths,
            'reasoning': reasoning_match.group(1).strip() if reasoning_match else ''
        }

    def run_distractor_verification_single(
        self,
        clues: str,
        question: str,
        truth: str,
        url: str,
        reasoning: str,
        content: str
    ) -> Dict[str, Any]:
        """Run verification for a single distractor with single truth."""
        prompt = WEB_DISTRACTION_EXTRACTION_PROMPT_SINGLE.format(
            clues=clues,
            question=question,
            truth=truth,
            url=url,
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

        return self.parse_distractor_verification_single(response_text)

    def run_distractor_verification_multiple(
        self,
        clues: str,
        question: str,
        truths: List[str],
        url: str,
        reasoning: str,
        content: str
    ) -> Dict[str, Any]:
        """Run verification for a single distractor with multiple truths."""
        truths_formatted = "\n".join([f"- {t}" for t in truths])

        prompt = WEB_DISTRACTION_EXTRACTION_PROMPT_MULTIPLE.format(
            clues=clues,
            question=question,
            truths=truths_formatted,
            url=url,
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

        return self.parse_distractor_verification_multiple(response_text)

    def verify_single_distractor(
        self,
        distractor: Dict,
        clues: str,
        question: str,
        truths: List[str],
        distractors_and_contents: Dict[str, str]
    ) -> Tuple[Dict[str, Any], bool]:
        """Verify a single distractor against one or more truths.

        Returns: (verification_result, is_valid_distractor)
        A valid distractor should NOT contain any truth.
        """
        url = distractor.get("id", "") or distractor.get("url", "")
        reasoning = distractor.get("reasoning", "")
        content = distractors_and_contents.get(url, "")

        if not content or content.startswith("Error"):
            return {"url": url, "error": "Could not fetch content"}, False

        if len(truths) == 1:
            result = self.run_distractor_verification_single(
                clues, question, truths[0], url, reasoning, content
            )
        else:
            result = self.run_distractor_verification_multiple(
                clues, question, truths, url, reasoning, content
            )

        # A valid distractor should NOT contain the truth
        is_valid = not result.get('contains_truth', True)
        return result, is_valid

    def verify_single_distractor_with_retries(
        self,
        distractor: Dict,
        clues: str,
        question: str,
        truths: List[str],
        distractors_and_contents: Dict[str, str],
        max_retries: int = 3
    ) -> Tuple[Dict[str, Any], bool, int]:
        """Verify a single distractor with retries.

        Returns: (verification_result, is_valid_distractor, retry_count)
        A valid distractor should NOT contain any truth.
        Retries up to max_retries times if distractor contains truth.
        """
        url = distractor.get("id", "") or distractor.get("url", "")
        reasoning = distractor.get("reasoning", "")
        content = distractors_and_contents.get(url, "")

        if not content or content.startswith("Error"):
            return {"url": url, "error": "Could not fetch content"}, False, 0

        result = None
        for attempt in range(max_retries):
            if len(truths) == 1:
                result = self.run_distractor_verification_single(
                    clues, question, truths[0], url, reasoning, content
                )
            else:
                result = self.run_distractor_verification_multiple(
                    clues, question, truths, url, reasoning, content
                )

            # A valid distractor should NOT contain the truth
            is_valid = not result.get('contains_truth', True)
            if is_valid:
                return result, True, attempt

        # After all retries, return the last result
        return result, False, max_retries

    def process_distractors_for_task(
        self,
        task: Dict,
        all_truths: List[str],
        distractors_and_contents: Dict[str, str],
        max_retries: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process all distractors for a task with retries and filtering.

        Returns: (valid_distractors, filtered_distractors, verification_results)
        - valid_distractors: distractors that passed verification
        - filtered_distractors: distractors that failed after max_retries
        - verification_results: all verification results
        """
        clues = task.get("clues", "")
        question = task.get("question", "")
        distractors = task.get("distractors", [])

        if not distractors:
            return [], [], []

        def verify_distractor_with_retry(distractor):
            return (
                distractor,
                self.verify_single_distractor_with_retries(
                    distractor, clues, question, all_truths, distractors_and_contents, max_retries
                )
            )

        valid_distractors = []
        filtered_distractors = []
        verification_results = []

        with ThreadPoolExecutor(max_workers=len(distractors)) as executor:
            futures = [executor.submit(verify_distractor_with_retry, d) for d in distractors]
            for future in as_completed(futures):
                distractor, (result, is_valid, retry_count) = future.result()
                result["retry_count"] = retry_count
                verification_results.append(result)

                if is_valid:
                    valid_distractors.append(distractor)
                else:
                    filtered_distractors.append({
                        "distractor": distractor,
                        "verification_result": result,
                        "retry_count": retry_count
                    })

        return valid_distractors, filtered_distractors, verification_results

    def get_distractor_contents(self, task: Dict) -> Dict[str, str]:
        """Get distractor contents from task's distractors_and_contents field."""
        return task.get("distractors_and_contents", {})

    def get_all_truths_for_file(self, tasks: List[Dict]) -> List[str]:
        """Get all truths from all tasks in a file."""
        truths = []
        for task in tasks:
            truth = task.get("truth", "")
            if truth and truth not in truths:
                truths.append(truth)
        return truths

    def get_task_for_distractor_verification(self, tasks: List[Dict]) -> Tuple[Dict, int]:
        """Find the task that needs distractor verification.

        Returns the last task (by level) that has distractors but hasn't been verified.
        """
        sorted_tasks = self.get_tasks_by_level(tasks)

        for i in range(len(sorted_tasks) - 1, -1, -1):
            task = sorted_tasks[i]
            # Has distractors but not verified yet
            if task.get("distractors") and "distractors_passed_verification" not in task:
                return task, i

        return None, -1

    def process_distractor_verification(self, filepath: str) -> Dict[str, Any]:
        """Process distractor verification for a file.

        Verifies each distractor up to 3 times. Distractors that fail after 3 tries
        are filtered out. Sets distractors_filtered=True if any were filtered.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        tasks = data.get("tasks", [])
        if not tasks:
            return {"status": "skipped", "reason": "no tasks", "distractors_passed_verification": None}

        task_to_verify, task_idx = self.get_task_for_distractor_verification(tasks)
        if task_to_verify is None:
            return {"status": "skipped", "reason": "no distractors to verify", "distractors_passed_verification": None}

        distractors = task_to_verify.get("distractors", [])
        if not distractors:
            return {"status": "skipped", "reason": "no distractors", "distractors_passed_verification": None}

        # Get all truths from all tasks up to and including this task
        # For level 0 tasks, just use the current truth
        # For level > 0 tasks, use all truths from previous levels + current
        task_level = task_to_verify.get("level", 0)
        sorted_tasks = self.get_tasks_by_level(tasks)
        all_truths = []
        for t in sorted_tasks:
            if t.get("level", 0) <= task_level:
                truth = t.get("truth", "")
                if truth and truth not in all_truths:
                    all_truths.append(truth)

        # Get distractor contents from stored distractors_and_contents
        distractors_and_contents = self.get_distractor_contents(task_to_verify)

        if not distractors_and_contents:
            return {"status": "error", "reason": "no distractors_and_contents found", "distractors_passed_verification": None}

        # Verify distractors with retries and filtering
        valid_distractors, filtered_distractors, verification_results = self.process_distractors_for_task(
            task_to_verify, all_truths, distractors_and_contents, max_retries=3
        )

        # Update task with valid distractors only (contains_truth: false)
        # Normalize url -> id before saving
        valid_distractor_results = []
        for r in verification_results:
            if not r.get('contains_truth', True):
                normalized = r.copy()
                if 'url' in normalized:
                    normalized['id'] = normalized.pop('url')
                valid_distractor_results.append(normalized)
        task_to_verify["valid_distractors"] = valid_distractor_results

        # Filter out invalid distractors and update the distractors list
        if filtered_distractors:
            task_to_verify["distractors_filtered"] = True
            task_to_verify["filtered_distractors"] = [
                {
                    "id": fd["distractor"].get("id", "") or fd["distractor"].get("url", ""),
                    "reasoning": fd["distractor"].get("reasoning", ""),
                    "verification_result": fd["verification_result"],
                    "retry_count": fd["retry_count"]
                }
                for fd in filtered_distractors
            ]
        else:
            task_to_verify["distractors_filtered"] = False

        # Always pass verification (we filter out bad ones instead of failing)
        task_to_verify["distractors_passed_verification"] = True

        del task_to_verify["distractors"]

        # Save updated file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        return {
            "status": "success",
            "updated": True,
            "distractors_passed_verification": True,
            "total_distractors": len(distractors),
            "valid_count": len(valid_distractors),
            "filtered_count": len(filtered_distractors)
        }

    def get_valid_files_for_distractor_verification(self, input_dir: str) -> List[str]:
        """Get files that have distractors needing verification."""
        all_files = glob(os.path.join(input_dir, "*.json"))
        valid_files = []

        for filepath in all_files:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                tasks = data.get("tasks", [])
                for task in tasks:
                    # Has distractors and distractors_and_contents but hasn't been verified
                    if (task.get("distractors") and
                        task.get("distractors_and_contents") and
                        "distractors_passed_verification" not in task):
                        valid_files.append(filepath)
                        break

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return valid_files

    def run_distractor_verification_batch(self, input_dir: str, max_workers: int = 8) -> Dict[str, Any]:
        """Run distractor verification on all files in a directory."""
        files_to_process = self.get_valid_files_for_distractor_verification(input_dir)

        results = []
        errors = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"Verifying distractors in {len(files_to_process)} files",
                total=len(files_to_process)
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_distractor_verification, f): f
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

        passed_count = sum(1 for r in results if r.get("distractors_passed_verification") is True)
        failed_count = sum(1 for r in results if r.get("distractors_passed_verification") is False)

        # Aggregate distractor counts across all files
        total_distractors = sum(r.get("total_distractors", 0) for r in results)
        total_kept = sum(r.get("valid_count", 0) for r in results)
        total_filtered = sum(r.get("filtered_count", 0) for r in results)

        return {
            "total_processed": len(files_to_process),
            "successful": len(results),
            "failed": len(errors),
            "passed_verification": passed_count,
            "failed_verification": failed_count,
            "total_distractors": total_distractors,
            "total_kept": total_kept,
            "total_filtered": total_filtered,
            "errors": errors
        }

    # ==================== End Distractor Verification Methods ====================

    def get_valid_files(self, input_dir: str) -> List[str]:
        all_files = glob(os.path.join(input_dir, "*.json"))
        valid_files = []

        for filepath in all_files:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                tasks = data.get("tasks", [])
                if not tasks:
                    continue

                all_valid = True
                for i, task in enumerate(tasks):
                    if not task.get("clues") or not task.get("question") or not task.get("truth"):
                        all_valid = False
                        break

                    if not task.get("supporting_items") or not task.get("items_and_contents"):
                        all_valid = False
                        break

                    level = task.get("level", 0)
                    num_supporting = len(task.get("supporting_items", []))

                    if level == 0:
                        if num_supporting != 3:
                            all_valid = False
                            break
                    else:
                        if num_supporting != 2:
                            all_valid = False
                            break
                        if not task.get("bridging_item"):
                            all_valid = False
                            break

                if all_valid:
                    valid_files.append(filepath)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return valid_files

    def run_batch(self, input_dir: str, max_workers: int = 8) -> Dict[str, Any]:
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
    parser.add_argument("--distractors", "-d", action="store_true", help="Run distractor verification instead of supporting item verification")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        exit(1)

    print(f"Input directory: {args.input}")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")

    client = get_anthropic_client()
    verifier = WebVerifier(client=client, model=args.model, max_retries=args.max_retries)

    if args.distractors:
        print("Mode: Distractor verification")
        print("-" * 40)

        result = verifier.run_distractor_verification_batch(args.input, max_workers=args.max_workers)

        print("-" * 40)
        print(f"Files processed: {result['total_processed']}")
        print(f"Total distractors: {result['total_distractors']}")
        print(f"Distractors kept: {result['total_kept']}")
        print(f"Distractors filtered out: {result['total_filtered']}")
        if result['errors']:
            print("\nErrors:")
            for err in result['errors']:
                print(f"  {err['file']}: {err['error']}")
    else:
        print("Mode: Supporting item verification")
        print(f"Max retries: {args.max_retries}")
        print("-" * 40)

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
