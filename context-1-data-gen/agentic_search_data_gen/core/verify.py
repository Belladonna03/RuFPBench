"""Shared verification logic for supporting items."""
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from .utils import (
    text_contains_quote,
    count_matching_quotes,
    min_required_matches,
    parse_quotes,
)


class BaseVerifier(ABC):
    """Base class for verification logic."""

    def __init__(self, client=None, model: str = None, max_retries: int = 3):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.id_field = 'id'

    def verify_supporting_item(
        self,
        item: Dict[str, Any],
        clues: str,
        items_and_contents: Dict[str, str],
    ) -> Dict[str, bool]:
        """Verify quotes for a supporting item."""
        result = {'clue_quotes_valid': True, 'item_quotes_valid': True, 'truth_quotes_valid': True}

        clue_quotes = item.get('clue_quotes', [])
        if clue_quotes:
            matches = count_matching_quotes(clue_quotes, clues)
            required = min_required_matches(len(clue_quotes))
            result['clue_quotes_valid'] = matches >= required

        item_quotes = item.get('item_quotes', [])
        item_id = item.get(self.id_field, '')
        content = items_and_contents.get(item_id, '')

        if item_quotes and content and not content.startswith("Error"):
            matches = count_matching_quotes(item_quotes, content)
            required = min_required_matches(len(item_quotes))
            result['item_quotes_valid'] = matches >= required

        if item.get('contains_truth', False):
            truth_quotes = item.get('truth_quotes', [])
            if truth_quotes and content and not content.startswith("Error"):
                actual_truth_quotes = [q for q in truth_quotes if q.lower() != 'none']
                if actual_truth_quotes:
                    for quote in actual_truth_quotes:
                        if not text_contains_quote(content, quote):
                            result['truth_quotes_valid'] = False
                            break

        return result

    def verify_all_items(
        self,
        supporting_items: List[Dict[str, Any]],
        clues: str,
        items_and_contents: Dict[str, str],
    ) -> Tuple[bool, List[Dict[str, bool]]]:
        """Verify all supporting items."""
        all_valid = True
        results = []

        for item in supporting_items:
            verification = self.verify_supporting_item(item, clues, items_and_contents)
            results.append(verification)

            if not verification['clue_quotes_valid']:
                all_valid = False
            if not verification['item_quotes_valid']:
                all_valid = False
            if item.get('contains_truth', False) and not verification['truth_quotes_valid']:
                all_valid = False

        return all_valid, results

    def check_truth_contained(self, supporting_items: List[Dict[str, Any]]) -> bool:
        """Check if any supporting item contains the truth."""
        return any(item.get('contains_truth', False) for item in supporting_items)

    def parse_single_item(self, content: str, id_tag: str = None) -> Dict[str, Any] | None:
        """Parse a single <item> block from LLM response."""
        if id_tag is None:
            id_tag = self.id_field

        item_match = re.search(r'<item>(.*?)</item>', content, re.DOTALL)
        if not item_match:
            return None

        item_content = item_match.group(1)
        id_match = re.search(rf'<{id_tag}>(.*?)</{id_tag}>', item_content, re.DOTALL)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_content, re.DOTALL)
        contains_truth_match = re.search(r'<contains_truth>(.*?)</contains_truth>', item_content, re.DOTALL)

        clue_quotes = parse_quotes(item_content, 'clue_quotes')
        item_quotes = parse_quotes(item_content, 'item_quotes')
        truth_quotes = parse_quotes(item_content, 'truth_quotes')

        contains_truth = False
        if contains_truth_match:
            contains_truth_str = contains_truth_match.group(1).strip().lower()
            contains_truth = contains_truth_str == 'true'

        not_relevant = clue_quotes is None or item_quotes is None

        return {
            'id': id_match.group(1).strip() if id_match else '',
            'clue_quotes': clue_quotes if clue_quotes is not None else [],
            'item_quotes': item_quotes if item_quotes is not None else [],
            'reasoning': reasoning_match.group(1).strip() if reasoning_match else '',
            'contains_truth': contains_truth,
            'truth_quotes': truth_quotes if truth_quotes is not None else [],
            'not_relevant': not_relevant
        }

    def update_supporting_items(
        self,
        original_items: List[Dict],
        extracted_items: List[Dict[str, Any]],
        items_and_contents: Dict[str, str],
    ) -> bool:
        """Update original items with extracted quotes."""
        updated = False
        id_to_extracted = {item["id"]: item for item in extracted_items}

        for item in original_items:
            item_id = item.get(self.id_field, '')
            if item_id in id_to_extracted:
                extracted = id_to_extracted[item_id]
                item["clue_quotes"] = extracted["clue_quotes"]
                item["item_quotes"] = extracted["item_quotes"]
                item["reasoning"] = extracted["reasoning"]
                item["truth_quotes"] = extracted["truth_quotes"] if extracted["truth_quotes"] else []

                truth_quotes = extracted["truth_quotes"] or []
                content = items_and_contents.get(item_id, "")
                valid_truth_quotes = [q for q in truth_quotes if q.lower() != 'none']
                if valid_truth_quotes and content and not content.startswith("Error"):
                    all_match = all(text_contains_quote(content, q) for q in valid_truth_quotes)
                    item["contains_truth"] = all_match
                else:
                    item["contains_truth"] = False

                if "not_relevant" in extracted:
                    item["not_relevant"] = extracted["not_relevant"]
                updated = True

        return updated

    def is_task_processed(self, task: Dict) -> bool:
        return "passed_verification" in task

    def is_file_fully_processed(self, filepath: str) -> bool:
        """Check if all tasks in a file have been processed."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            tasks = data.get("tasks", [])
            if not tasks:
                return True

            for task in tasks:
                if not self.is_task_processed(task):
                    return False

            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def process_items_parallel(
        self,
        process_single_fn,
        items: List[Dict],
        max_workers: int = None
    ) -> Tuple[List[Dict[str, Any]], int, bool]:
        """Process items in parallel and return (extracted_items, max_retries, all_valid)."""
        if max_workers is None:
            max_workers = len(items)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_fn, items))

        extracted_items = []
        total_retries = 0
        all_valid = True

        for extracted_item, retries, valid in results:
            total_retries = max(total_retries, retries)
            if extracted_item:
                extracted_items.append(extracted_item)
            else:
                all_valid = False
            if not valid:
                all_valid = False

        return extracted_items, total_retries, all_valid

    @abstractmethod
    def run_single_item_extraction(self, *args, **kwargs) -> Dict[str, Any] | None:
        """Run extraction for a single item. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def process_single_item_with_retries(
        self,
        item: Dict,
        clues: str,
        question: str,
        truth: str,
        items_and_contents: Dict[str, str],
    ) -> Tuple[Dict[str, Any] | None, int, bool]:
        """Process a single item with retries. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def process_task(self, task: Dict) -> Tuple[List[Dict[str, Any]], int, bool]:
        """Process a task. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def process_file(self, filepath: str) -> Dict[str, Any]:
        """Process a file. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_valid_files(self, input_dir: str) -> List[str]:
        """Get valid files from directory. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def run_batch(self, input_dir: str, max_workers: int = 8) -> Dict[str, Any]:
        """Run batch processing. Must be implemented by subclasses."""
        pass


# Standalone functions for backward compatibility
def verify_supporting_item(
    item: Dict[str, Any],
    clues: str,
    items_and_contents: Dict[str, str],
    id_field: str = 'id'
) -> Dict[str, bool]:
    """Verify quotes for a supporting item (standalone function for compatibility)."""
    verifier = _CompatibilityVerifier(id_field=id_field)
    return verifier.verify_supporting_item(item, clues, items_and_contents)


def verify_all_items(
    supporting_items: List[Dict[str, Any]],
    clues: str,
    items_and_contents: Dict[str, str],
    id_field: str = 'id'
) -> Tuple[bool, List[Dict[str, bool]]]:
    """Verify all supporting items (standalone function for compatibility)."""
    verifier = _CompatibilityVerifier(id_field=id_field)
    return verifier.verify_all_items(supporting_items, clues, items_and_contents)


def check_truth_contained(supporting_items: List[Dict[str, Any]]) -> bool:
    """Check if any supporting item contains the truth (standalone function for compatibility)."""
    verifier = _CompatibilityVerifier()
    return verifier.check_truth_contained(supporting_items)


def parse_single_item(content: str, id_tag: str = 'id') -> Dict[str, Any] | None:
    """Parse a single <item> block from LLM response (standalone function for compatibility)."""
    verifier = _CompatibilityVerifier(id_field=id_tag)
    return verifier.parse_single_item(content, id_tag=id_tag)


def update_supporting_items(
    original_items: List[Dict],
    extracted_items: List[Dict[str, Any]],
    items_and_contents: Dict[str, str],
    id_field: str = 'id'
) -> bool:
    """Update original items with extracted quotes (standalone function for compatibility)."""
    verifier = _CompatibilityVerifier(id_field=id_field)
    return verifier.update_supporting_items(original_items, extracted_items, items_and_contents)


def is_task_processed(task: Dict) -> bool:
    """Check if a task has been processed (standalone function for compatibility)."""
    verifier = _CompatibilityVerifier()
    return verifier.is_task_processed(task)


def is_file_fully_processed(filepath: str) -> bool:
    """Check if all tasks in a file have been processed (standalone function for compatibility)."""
    verifier = _CompatibilityVerifier()
    return verifier.is_file_fully_processed(filepath)


def process_items_parallel(
    process_single_fn,
    items: List[Dict],
    max_workers: int = None
) -> Tuple[List[Dict[str, Any]], int, bool]:
    """Process items in parallel (standalone function for compatibility)."""
    verifier = _CompatibilityVerifier()
    return verifier.process_items_parallel(process_single_fn, items, max_workers)


class _CompatibilityVerifier(BaseVerifier):
    """Internal verifier for backward compatibility with standalone functions."""

    def __init__(self, id_field: str = 'id', **kwargs):
        super().__init__(**kwargs)
        self.id_field = id_field

    def run_single_item_extraction(self, *args, **kwargs):
        raise NotImplementedError("Compatibility verifier does not support extraction")

    def process_single_item_with_retries(self, *args, **kwargs):
        raise NotImplementedError("Compatibility verifier does not support processing")

    def process_task(self, task: Dict):
        raise NotImplementedError("Compatibility verifier does not support task processing")

    def process_file(self, filepath: str):
        raise NotImplementedError("Compatibility verifier does not support file processing")

    def get_valid_files(self, input_dir: str):
        raise NotImplementedError("Compatibility verifier does not support file discovery")

    def run_batch(self, input_dir: str, max_workers: int = 8):
        raise NotImplementedError("Compatibility verifier does not support batch processing")
