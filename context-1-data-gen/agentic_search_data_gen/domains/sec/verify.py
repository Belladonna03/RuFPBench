"""Verification and extraction of quotes from supporting items in SEC filings.

This module handles two types of verification:
1. Main verification: Extract and verify quotes from supporting items (default mode)
2. Collect verification: Verify additional chunks collected by collect.py (--mode collect)
"""

import os
import re
import json
import argparse
from glob import glob
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from dotenv import load_dotenv

from .prompts import (
    SEC_SINGLE_ITEM_EXTRACTION_PROMPT,
    SEC_EXTENSION_VERIFICATION_PROMPT,
    SEC_COLLECT_VERIFICATION_PROMPT,
    SEC_COLLECT_TRUTH_SECTION_TEMPLATE,
    SEC_COLLECT_TRUTH_INSTRUCTIONS_WITH,
    SEC_COLLECT_TRUTH_INSTRUCTIONS_WITHOUT,
    SEC_COLLECT_TRUTH_OUTPUT_WITH,
    SEC_COLLECT_TRUTH_OUTPUT_WITHOUT,
)
from ...core.utils import (
    get_anthropic_client,
    text_contains_quote,
    count_matching_quotes,
    min_required_matches,
    parse_quotes,
)
from .utils import get_latest_task, parse_supporting_items, parse_single_item

load_dotenv()


# ============================================================================
# Collect Verification Functions
# ============================================================================


def parse_collect_verification_response(content: str) -> Dict[str, Any]:
    """Parse verification response for additional chunks."""
    result = {
        'item_quotes': [],
        'truth_quotes': [],
        'reasoning': '',
        'not_relevant': False
    }

    outer_match = re.search(r'<verification>(.*?)</verification>', content, re.DOTALL)
    if not outer_match:
        return result

    inner = outer_match.group(1)

    # Parse item_quotes
    item_quotes = parse_quotes(inner, 'item_quotes')
    result['item_quotes'] = item_quotes if item_quotes is not None else []
    if item_quotes is None:
        result['not_relevant'] = True

    # Parse truth_quotes (optional)
    truth_quotes = parse_quotes(inner, 'truth_quotes')
    result['truth_quotes'] = truth_quotes if truth_quotes is not None else []

    # Parse reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', inner, re.DOTALL)
    result['reasoning'] = reasoning_match.group(1).strip() if reasoning_match else ''

    return result


def verify_collect_chunk(
    client,
    model: str,
    question: str,
    truth: str,
    clues: str,
    original_item_quotes: List[str],
    original_truth_quotes: List[str],
    contains_truth: bool,
    chunk_id: str,
    chunk_content: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Verify a single additional chunk from collect phase."""
    # Build the prompt
    item_quotes_str = '\n'.join(f'<q>{q}</q>' for q in original_item_quotes) if original_item_quotes else 'N/A'

    if contains_truth:
        truth_quotes_str = '\n'.join(f'<q>{q}</q>' for q in original_truth_quotes) if original_truth_quotes else 'N/A'
        truth_section = SEC_COLLECT_TRUTH_SECTION_TEMPLATE.format(truth_quotes=truth_quotes_str)
        truth_instructions = SEC_COLLECT_TRUTH_INSTRUCTIONS_WITH
        truth_output_format = SEC_COLLECT_TRUTH_OUTPUT_WITH
    else:
        truth_section = ""
        truth_instructions = SEC_COLLECT_TRUTH_INSTRUCTIONS_WITHOUT
        truth_output_format = SEC_COLLECT_TRUTH_OUTPUT_WITHOUT

    prompt = SEC_COLLECT_VERIFICATION_PROMPT.format(
        question=question,
        truth=truth,
        clues=clues,
        item_quotes=item_quotes_str,
        truth_section=truth_section,
        chunk_id=chunk_id,
        chunk_content=chunk_content,
        truth_instructions=truth_instructions,
        truth_output_format=truth_output_format
    )

    response_text = ""
    for _ in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[{"role": "user", "content": prompt}]
            )

            for item in response.content:
                if item.type == "text":
                    response_text = item.text
                    break
            break
        except Exception as e:
            print(f"Error verifying chunk {chunk_id}: {e}")
            continue

    return parse_collect_verification_response(response_text)


def verify_additional_chunk(
    client,
    model: str,
    question: str,
    truth: str,
    clues: str,
    original_item_quotes: List[str],
    original_truth_quotes: List[str],
    contains_truth: bool,
    chunk_id: str,
    chunk_content: str,
    max_retries: int = 3
) -> Tuple[bool, Dict[str, Any]]:
    """Verify a single additional chunk.

    Returns:
        Tuple of (passed_verification, extracted_data)
    """
    # Build the prompt
    item_quotes_str = '\n'.join(f'<q>{q}</q>' for q in original_item_quotes) if original_item_quotes else 'N/A'

    if contains_truth:
        truth_quotes_str = '\n'.join(f'<q>{q}</q>' for q in original_truth_quotes) if original_truth_quotes else 'N/A'
        truth_section = SEC_COLLECT_TRUTH_SECTION_TEMPLATE.format(truth_quotes=truth_quotes_str)
        truth_instructions = SEC_COLLECT_TRUTH_INSTRUCTIONS_WITH
        truth_output_format = SEC_COLLECT_TRUTH_OUTPUT_WITH
    else:
        truth_section = ""
        truth_instructions = SEC_COLLECT_TRUTH_INSTRUCTIONS_WITHOUT
        truth_output_format = SEC_COLLECT_TRUTH_OUTPUT_WITHOUT

    prompt = SEC_COLLECT_VERIFICATION_PROMPT.format(
        question=question,
        truth=truth,
        clues=clues,
        item_quotes=item_quotes_str,
        truth_section=truth_section,
        chunk_id=chunk_id,
        chunk_content=chunk_content,
        truth_instructions=truth_instructions,
        truth_output_format=truth_output_format
    )

    extracted = None
    for attempt in range(max_retries):
        try:
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

            extracted = parse_collect_verification_response(response_text)

            # If model says not relevant, don't retry
            if extracted.get('not_relevant'):
                return False, extracted

            # Verify item_quotes against chunk content
            item_quotes = extracted.get('item_quotes', [])
            if item_quotes:
                matches = count_matching_quotes(item_quotes, chunk_content)
                required = min_required_matches(len(item_quotes))
                if matches >= required:
                    # If contains_truth, also verify truth_quotes
                    if contains_truth:
                        truth_quotes = extracted.get('truth_quotes', [])
                        if truth_quotes:
                            truth_matches = count_matching_quotes(truth_quotes, chunk_content)
                            truth_required = min_required_matches(len(truth_quotes))
                            if truth_matches >= truth_required:
                                return True, extracted
                        # No truth quotes found but item quotes valid
                        # For truth-containing chunks, truth quotes are required
                        continue
                    else:
                        return True, extracted

        except Exception as e:
            print(f"Error verifying chunk {chunk_id} (attempt {attempt + 1}): {e}")
            continue

    return False, extracted if extracted else {}


def process_collect_supporting_item(
    client,
    model: str,
    question: str,
    truth: str,
    clues: str,
    supporting_item: Dict[str, Any],
    items_and_contents: Dict[str, str],
    max_retries: int = 3
) -> Tuple[List[Dict], List[Dict], int, int]:
    """Process additional_chunks for a single supporting item.

    Returns:
        Tuple of (filtered_chunks, filtered_out_chunks, num_original, num_filtered)
    """
    additional_chunks = supporting_item.get('additional_chunks', [])
    if not additional_chunks:
        return [], [], 0, 0

    original_item_quotes = supporting_item.get('item_quotes', [])
    original_truth_quotes = supporting_item.get('truth_quotes', [])
    contains_truth = supporting_item.get('contains_truth', False)

    filtered_chunks = []
    filtered_out_chunks = []

    for chunk in additional_chunks:
        chunk_id = chunk.get('id', '')
        chunk_content = items_and_contents.get(chunk_id, '')

        if not chunk_content:
            filtered_out_chunks.append({**chunk, 'filter_reason': 'no content found'})
            continue

        passed, extracted = verify_additional_chunk(
            client=client,
            model=model,
            question=question,
            truth=truth,
            clues=clues,
            original_item_quotes=original_item_quotes,
            original_truth_quotes=original_truth_quotes,
            contains_truth=contains_truth,
            chunk_id=chunk_id,
            chunk_content=chunk_content,
            max_retries=max_retries
        )

        if passed:
            # Add verification data to the chunk
            chunk['item_quotes'] = extracted.get('item_quotes', [])
            if contains_truth:
                chunk['truth_quotes'] = extracted.get('truth_quotes', [])
            chunk['reasoning'] = extracted.get('reasoning', '')
            filtered_chunks.append(chunk)
        else:
            chunk['filter_reason'] = 'verification failed'
            if extracted:
                chunk['verification_data'] = extracted
            filtered_out_chunks.append(chunk)

    return filtered_chunks, filtered_out_chunks, len(additional_chunks), len(filtered_chunks)


def process_collect_file(
    filepath: str,
    model: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Process a file for collect verification."""
    client = get_anthropic_client()

    with open(filepath, "r") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    if not tasks:
        return {"status": "skipped", "reason": "no tasks"}

    task = get_latest_task(tasks)
    if task is None:
        return {"status": "skipped", "reason": "no tasks"}

    if not task.get("passed_verification"):
        return {"status": "skipped", "reason": "task not verified"}

    question = task.get("question", "")
    truth = task.get("truth", "")
    clues = task.get("clues", "")
    supporting_items = task.get("supporting_items", [])
    bridging_item = task.get("bridging_item")
    items_and_contents = task.get("items_and_contents", {})

    stats = {
        'items_with_chunks': 0,
        'items_processed': 0,
        'total_original': 0,
        'total_filtered': 0,
        'bridging_processed': False
    }

    for item in supporting_items:
        additional_chunks = item.get('additional_chunks', [])
        if not additional_chunks:
            continue

        stats['items_with_chunks'] += 1

        filtered_chunks, filtered_out_chunks, num_original, num_filtered = process_collect_supporting_item(
            client=client,
            model=model,
            question=question,
            truth=truth,
            clues=clues,
            supporting_item=item,
            items_and_contents=items_and_contents,
            max_retries=max_retries
        )

        # Update the supporting item with filtered chunks
        item['additional_chunks'] = filtered_chunks
        item['filtered_out_additional_chunks'] = filtered_out_chunks

        stats['items_processed'] += 1
        stats['total_original'] += num_original
        stats['total_filtered'] += num_filtered

    # Process bridging_item if present and has additional_chunks
    if bridging_item and bridging_item.get('additional_chunks'):
        stats['bridging_processed'] = True

        # Create a temporary supporting_item-like structure for verification
        # Bridging items don't contain truth, so contains_truth is False
        bridging_as_supporting = {
            'id': bridging_item.get('id', ''),
            'clue_quotes': bridging_item.get('clue_quotes', []),
            'item_quotes': bridging_item.get('item_quotes', []),
            'truth_quotes': [],
            'contains_truth': False,
            'additional_chunks': bridging_item.get('additional_chunks', [])
        }

        filtered_chunks, filtered_out_chunks, num_original, num_filtered = process_collect_supporting_item(
            client=client,
            model=model,
            question=question,
            truth=truth,
            clues=clues,
            supporting_item=bridging_as_supporting,
            items_and_contents=items_and_contents,
            max_retries=max_retries
        )

        bridging_item['additional_chunks'] = filtered_chunks
        bridging_item['filtered_out_additional_chunks'] = filtered_out_chunks

        stats['total_original'] += num_original
        stats['total_filtered'] += num_filtered

    # Mark task as having additional_chunks_filtered
    task['additional_chunks_filtered'] = True

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    return {
        "status": "success",
        **stats
    }


def get_collect_eligible_files(input_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Get files eligible for collect verification."""
    all_files = glob(os.path.join(input_dir, "*.json"))
    valid_files = []
    files_to_process = []
    already_processed = []

    for filepath in all_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            tasks = data.get("tasks", [])
            if not tasks:
                continue

            task = get_latest_task(tasks)
            if task is None:
                continue

            if not task.get("passed_verification"):
                continue

            # Must have supporting_items with additional_chunks
            supporting_items = task.get("supporting_items", [])
            has_additional = any(item.get('additional_chunks') for item in supporting_items)

            # Also check bridging_item
            bridging_item = task.get("bridging_item")
            has_bridging_additional = bridging_item and bridging_item.get('additional_chunks')

            if not has_additional and not has_bridging_additional:
                continue

            valid_files.append(filepath)

            if task.get('additional_chunks_filtered'):
                already_processed.append(filepath)
            else:
                files_to_process.append(filepath)

        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return valid_files, files_to_process, already_processed


def run_collect_batch(
    input_dir: str,
    model: str,
    max_workers: int = 8,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Run batch processing for collect verification on all files."""
    valid_files, files_to_process, already_processed = get_collect_eligible_files(input_dir)

    print(f"Total valid files: {len(valid_files)}")
    print(f"Already processed: {len(already_processed)}")
    print(f"To process: {len(files_to_process)}")
    print("-" * 40)

    if not files_to_process:
        print("No files to process.")
        return {
            "total_valid": len(valid_files),
            "already_processed": len(already_processed),
            "processed": 0,
            "successful": 0,
            "total_original_chunks": 0,
            "total_filtered_chunks": 0,
            "filtered_out": 0,
            "errors": []
        }

    results = []
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(
            f"Processing {len(files_to_process)} files",
            total=len(files_to_process)
        )

        def process_with_args(filepath):
            return process_collect_file(filepath, model, max_retries=max_retries)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_with_args, f): f
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

    total_original = sum(r.get('total_original', 0) for r in results)
    total_filtered = sum(r.get('total_filtered', 0) for r in results)
    successful = sum(1 for r in results if r.get('status') == 'success')

    return {
        "total_valid": len(valid_files),
        "already_processed": len(already_processed),
        "processed": len(files_to_process),
        "successful": successful,
        "total_original_chunks": total_original,
        "total_filtered_chunks": total_filtered,
        "filtered_out": total_original - total_filtered,
        "errors": errors
    }


# ============================================================================
# Main Verification Functions
# ============================================================================


def get_task_to_verify(tasks: List[Dict]) -> Tuple[Dict | None, Dict | None]:
    """Find the latest task that needs verification.

    Returns:
        Tuple of (task_to_verify, prev_task) where prev_task is None for level 0 tasks.
    """
    if not tasks:
        return None, None

    # Sort by level
    sorted_tasks = sorted(tasks, key=lambda t: t.get("level", 0))

    # Find the latest task without passed_verification
    for task in reversed(sorted_tasks):
        if "passed_verification" not in task:
            level = task.get("level", 0)
            prev_task = None
            if level > 0:
                # Find the previous level's task
                for t in sorted_tasks:
                    if t.get("level", 0) == level - 1:
                        prev_task = t
                        break
            return task, prev_task

    return None, None


def verify_supporting_item(
    item: Dict[str, Any],
    clues: str,
    items_and_contents: Dict[str, str]
) -> Dict[str, bool]:
    """Verify quotes for a single supporting item."""
    result = {
        'clue_quotes_valid': True,
        'item_quotes_valid': True,
        'truth_quotes_valid': True
    }

    # Verify clue_quotes against clues
    clue_quotes = item.get('clue_quotes', [])
    if clue_quotes:
        matches = count_matching_quotes(clue_quotes, clues)
        required = min_required_matches(len(clue_quotes))
        result['clue_quotes_valid'] = matches >= required

    # Verify item_quotes against chunk content
    item_quotes = item.get('item_quotes', [])
    item_id = item.get('id', '')
    content = items_and_contents.get(item_id, '')

    if item_quotes and content and not content.startswith("Error"):
        matches = count_matching_quotes(item_quotes, content)
        required = min_required_matches(len(item_quotes))
        result['item_quotes_valid'] = matches >= required

    # Verify truth_quotes if contains_truth
    if item.get('contains_truth', False):
        truth_quotes = item.get('truth_quotes', [])
        if truth_quotes and content and not content.startswith("Error"):
            # Filter out "none" entries
            valid_quotes = [q for q in truth_quotes if q.lower() != 'none']
            if valid_quotes:
                all_match = all(text_contains_quote(content, q) for q in valid_quotes)
                result['truth_quotes_valid'] = all_match
            else:
                result['truth_quotes_valid'] = False
        elif not truth_quotes:
            result['truth_quotes_valid'] = False

    return result


def verify_all_items(
    items: List[Dict[str, Any]],
    clues: str,
    items_and_contents: Dict[str, str]
) -> Tuple[bool, List[Dict[str, bool]]]:
    """Verify all supporting items."""
    all_valid = True
    results = []

    for item in items:
        verification = verify_supporting_item(item, clues, items_and_contents)
        results.append(verification)

        if not verification['clue_quotes_valid']:
            all_valid = False
        if not verification['item_quotes_valid']:
            all_valid = False
        if item.get('contains_truth', False) and not verification['truth_quotes_valid']:
            all_valid = False

    return all_valid, results


def check_truth_contained(supporting_items: List[Dict[str, Any]]) -> bool:
    """Check if at least one supporting item contains verifiable truth."""
    for item in supporting_items:
        if item.get('contains_truth', False):
            truth_quotes = item.get('truth_quotes', [])
            valid_quotes = [q for q in truth_quotes if q.lower() != 'none']
            if valid_quotes:
                return True
    return False


def has_not_relevant(items: List[Dict[str, Any]]) -> bool:
    """Check if any item has not_relevant=True."""
    return any(item.get('not_relevant', False) for item in items)


def build_failure_reasons(
    quotes_verified: bool,
    truth_contained: bool,
    verification_results: List[Dict[str, bool]],
    bridging_verification: Dict[str, bool] = None,
    has_not_relevant_items: bool = False,
    has_bridging_not_relevant: bool = False
) -> List[str]:
    """Build human-readable failure reasons."""
    reasons = []

    if has_not_relevant_items:
        reasons.append("One or more supporting items were marked as not relevant by the model")

    if has_bridging_not_relevant:
        reasons.append("Bridging item or its previous item was marked as not relevant by the model")

    for i, verification in enumerate(verification_results):
        item_reasons = []
        if not verification.get('clue_quotes_valid', True):
            item_reasons.append("clue quotes not found in clues")
        if not verification.get('item_quotes_valid', True):
            item_reasons.append("item quotes not found in chunk content")
        if not verification.get('truth_quotes_valid', True):
            item_reasons.append("truth quotes not found in chunk content")
        if item_reasons:
            reasons.append(f"Item {i+1}: {', '.join(item_reasons)}")

    if bridging_verification:
        bridging_reasons = []
        if not bridging_verification.get('item_clue_quotes_valid', True):
            bridging_reasons.append("bridging item clue quotes not found in clues")
        if not bridging_verification.get('item_quotes_valid', True):
            bridging_reasons.append("bridging item quotes not found in chunk content")
        if not bridging_verification.get('prev_item_clue_quotes_valid', True):
            bridging_reasons.append("previous item clue quotes not found in previous clues")
        if not bridging_verification.get('prev_item_quotes_valid', True):
            bridging_reasons.append("previous item quotes not found in previous chunk content")
        if bridging_reasons:
            reasons.append(f"Bridging item: {', '.join(bridging_reasons)}")

    if not truth_contained:
        reasons.append("No supporting item contains verifiable truth quotes")

    return reasons


def has_bridging_not_relevant(bridging_item: Dict[str, Any]) -> bool:
    """Check if bridging item or prev_item has not_relevant=True."""
    item = bridging_item.get('item', {})
    prev_item = bridging_item.get('prev_item', {})
    return item.get('not_relevant', False) or prev_item.get('not_relevant', False)


def parse_bridging_item_from_response(content: str) -> Dict[str, Any]:
    """Parse bridging_item from extraction response.

    Parses chunk_id and relevant_prev_chunk_id from model output and normalizes to 'id' and 'relevant_prev_id' internally.

    New structure has:
    - <item>: bridging chunk with chunk_id, clue_quotes, item_quotes, reasoning
    - <prev_item>: previous chunk with relevant_prev_chunk_id, clue_quotes, prev_item_quotes, reasoning
    """
    outer_match = re.search(r'<bridging_item>(.*?)</bridging_item>', content, re.DOTALL)
    if not outer_match:
        return {}

    bridging_content = outer_match.group(1)

    # Parse <item> section (the bridging chunk)
    item_match = re.search(r'<item>(.*?)</item>', bridging_content, re.DOTALL)
    item_data = {}
    if item_match:
        item_content = item_match.group(1)
        # Parse chunk_id from model output (normalize to 'id' internally)
        chunk_id_match = re.search(r'<chunk_id>(.*?)</chunk_id>', item_content, re.DOTALL)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_content, re.DOTALL)

        clue_quotes = parse_quotes(item_content, 'clue_quotes')
        item_quotes = parse_quotes(item_content, 'item_quotes')

        # Track if quotes are None (model indicated content is not relevant)
        not_relevant = clue_quotes is None or item_quotes is None

        item_data = {
            'id': chunk_id_match.group(1).strip() if chunk_id_match else '',  # Normalize chunk_id -> id
            'clue_quotes': clue_quotes if clue_quotes is not None else [],
            'item_quotes': item_quotes if item_quotes is not None else [],
            'reasoning': reasoning_match.group(1).strip() if reasoning_match else '',
            'not_relevant': not_relevant
        }

    # Parse <prev_item> section (the previous supporting item)
    prev_item_match = re.search(r'<prev_item>(.*?)</prev_item>', bridging_content, re.DOTALL)
    prev_item_data = {}
    if prev_item_match:
        prev_item_content = prev_item_match.group(1)
        # Parse relevant_prev_chunk_id from model output (normalize to 'relevant_prev_id' internally)
        prev_id_match = re.search(r'<relevant_prev_chunk_id>(.*?)</relevant_prev_chunk_id>', prev_item_content, re.DOTALL)
        prev_reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', prev_item_content, re.DOTALL)

        clue_quotes = parse_quotes(prev_item_content, 'clue_quotes')
        prev_item_quotes = parse_quotes(prev_item_content, 'prev_item_quotes')

        # Track if quotes are None (model indicated content is not relevant)
        not_relevant = clue_quotes is None or prev_item_quotes is None

        prev_item_data = {
            'relevant_prev_id': prev_id_match.group(1).strip() if prev_id_match else '',  # Normalize to relevant_prev_id
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
    bridging_item: Dict[str, Any],
    clues: str,
    items_and_contents: Dict[str, str],
    prev_clues: str = "",
    prev_items_and_contents: Dict[str, str] = None
) -> Dict[str, bool]:
    """Verify quotes for a bridging item (no truth check).

    Verifies both:
    - item: bridging chunk quotes against current clues and bridging chunk content
    - prev_item: previous item quotes against previous clues and previous chunk content
    """
    if prev_items_and_contents is None:
        prev_items_and_contents = {}

    result = {
        'item_clue_quotes_valid': True,
        'item_quotes_valid': True,
        'prev_item_clue_quotes_valid': True,
        'prev_item_quotes_valid': True
    }

    # Get item and prev_item from the new structure
    item = bridging_item.get('item', {})
    prev_item = bridging_item.get('prev_item', {})

    # Verify item (bridging chunk) - clue_quotes against current clues
    item_clue_quotes = item.get('clue_quotes', [])
    if item_clue_quotes:
        matches = count_matching_quotes(item_clue_quotes, clues)
        required = min_required_matches(len(item_clue_quotes))
        result['item_clue_quotes_valid'] = matches >= required

    # Verify item - item_quotes against bridging chunk content
    item_quotes = item.get('item_quotes', [])
    chunk_id = item.get('id', '')
    content = items_and_contents.get(chunk_id, '')

    if item_quotes and content and not content.startswith("Error"):
        matches = count_matching_quotes(item_quotes, content)
        required = min_required_matches(len(item_quotes))
        result['item_quotes_valid'] = matches >= required

    # Verify prev_item - clue_quotes against previous clues
    prev_clue_quotes = prev_item.get('clue_quotes', [])
    if prev_clue_quotes and prev_clues:
        matches = count_matching_quotes(prev_clue_quotes, prev_clues)
        required = min_required_matches(len(prev_clue_quotes))
        result['prev_item_clue_quotes_valid'] = matches >= required

    # Verify prev_item - prev_item_quotes against previous chunk content
    prev_item_quotes = prev_item.get('prev_item_quotes', [])
    prev_id = prev_item.get('relevant_prev_id', '')
    prev_content = prev_items_and_contents.get(prev_id, '')

    if prev_item_quotes and prev_content and not prev_content.startswith("Error"):
        matches = count_matching_quotes(prev_item_quotes, prev_content)
        required = min_required_matches(len(prev_item_quotes))
        result['prev_item_quotes_valid'] = matches >= required

    return result


def verify_extension_items(
    bridging_item: Dict[str, Any],
    supporting_items: List[Dict[str, Any]],
    clues: str,
    items_and_contents: Dict[str, str],
    prev_clues: str = "",
    prev_items_and_contents: Dict[str, str] = None
) -> Tuple[bool, Dict[str, bool], List[Dict[str, bool]]]:
    """
    Verify bridging_item and supporting_items for an extension task.
    Only checks truth in supporting_items.
    Returns (all_valid, bridging_verification, supporting_verifications).
    """
    if prev_items_and_contents is None:
        prev_items_and_contents = {}

    all_valid = True

    # Verify bridging item (both item and prev_item parts)
    bridging_verification = verify_bridging_item(
        bridging_item, clues, items_and_contents, prev_clues, prev_items_and_contents
    )

    # Check all four validity flags for bridging item
    if not bridging_verification['item_clue_quotes_valid']:
        all_valid = False
    if not bridging_verification['item_quotes_valid']:
        all_valid = False
    if not bridging_verification['prev_item_clue_quotes_valid']:
        all_valid = False
    if not bridging_verification['prev_item_quotes_valid']:
        all_valid = False

    # Verify supporting items (with truth check)
    supporting_verifications = []
    for item in supporting_items:
        verification = verify_supporting_item(item, clues, items_and_contents)
        supporting_verifications.append(verification)

        if not verification['clue_quotes_valid']:
            all_valid = False
        if not verification['item_quotes_valid']:
            all_valid = False
        if item.get('contains_truth', False) and not verification['truth_quotes_valid']:
            all_valid = False

    return all_valid, bridging_verification, supporting_verifications


def format_bridging_item_for_prompt(bridging_item: Dict, items_and_contents: Dict[str, str]) -> str:
    """Format bridging_item with its content for the prompt.

    Uses chunk_id in the prompt for model clarity (internally stored as 'id').
    """
    item_id = bridging_item.get("id", "")
    relevant_prev_id = bridging_item.get("relevant_prev_id", "")
    reasoning = bridging_item.get("reasoning", "")
    content = items_and_contents.get(item_id, "")

    return f"""<chunk_id>{item_id}</chunk_id>
<relevant_prev_chunk_id>{relevant_prev_id}</relevant_prev_chunk_id>
<reasoning>{reasoning}</reasoning>
<content>
{content}
</content>"""


def format_previous_supporting_item_for_prompt(prev_task: Dict, relevant_prev_id: str) -> str:
    """Format the previous task's supporting item that contains the bridging connection.

    Uses chunk_id in the prompt for model clarity.
    """
    items_and_contents = prev_task.get("items_and_contents", {})
    content = items_and_contents.get(relevant_prev_id, "")
    prev_clues = prev_task.get("clues", "")

    return f"""<chunk_id>{relevant_prev_id}</chunk_id>
<previous_clues>
{prev_clues}
</previous_clues>
<content>
{content}
</content>"""


def run_extension_extraction(client, model: str, prompt: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run extraction for extension tasks, returns (bridging_item, supporting_items)."""
    response = client.messages.create(
        model=model,
        max_tokens=10000,
        thinking={"type": "enabled", "budget_tokens": 2000},
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = ""
    for item in response.content:
        if item.type == "text":
            response_text = item.text
            break

    bridging_item = parse_bridging_item_from_response(response_text)
    supporting_items = parse_supporting_items(response_text, include_quotes=True)

    return bridging_item, supporting_items


def process_extension_task(
    client,
    model: str,
    task: Dict,
    prev_task: Dict,
    max_retries: int = 3
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int, bool]:
    """
    Process an extension task (task with level > 0).
    Returns (extracted_bridging_item, extracted_supporting_items, retry_count, quotes_verified).
    """
    clues = task.get("clues", "")
    question = task.get("question", "")
    truth = task.get("truth", "")
    bridging_item = task.get("bridging_item", {})
    supporting_items = task.get("supporting_items", [])
    items_and_contents = task.get("items_and_contents", {})

    previous_truth = prev_task.get("truth", "")
    previous_clues = prev_task.get("clues", "")
    prev_items_and_contents = prev_task.get("items_and_contents", {})

    # Check if relevant_prev_id is in previous task's items_and_contents
    relevant_prev_id = bridging_item.get("relevant_prev_id", "")
    if relevant_prev_id not in prev_items_and_contents:
        print(f"Warning: relevant_prev_id '{relevant_prev_id}' not found in previous task's items_and_contents")
        return {}, [], 0, False

    if not supporting_items or not items_and_contents:
        return {}, [], 0, False

    # Format inputs for the prompt
    formatted_bridging = format_bridging_item_for_prompt(bridging_item, items_and_contents)
    formatted_prev_item = format_previous_supporting_item_for_prompt(prev_task, relevant_prev_id)
    formatted_items = format_supporting_items_for_prompt(supporting_items, items_and_contents)
    num_items = len(supporting_items)

    prompt = SEC_EXTENSION_VERIFICATION_PROMPT.format(
        clues=clues,
        question=question,
        truth=truth,
        bridging_item=formatted_bridging,
        previous_truth=previous_truth,
        previous_supporting_item=formatted_prev_item,
        supporting_items=formatted_items,
        num_items=num_items
    )

    # Get expected ids for enforcing correct values after extraction
    bridging_id = bridging_item.get("id", "")
    expected_supporting_ids = [item.get("id", "") for item in supporting_items]

    # Try extraction with verification, retry up to max_retries times
    extracted_bridging = {}
    extracted_items = []

    for attempt in range(max_retries):
        extracted_bridging, extracted_items = run_extension_extraction(client, model, prompt)

        if not extracted_bridging or not extracted_items:
            continue

        # Ensure extracted ids match expected values - LLM may output different format/typo
        # Bridging item ids
        if 'item' in extracted_bridging:
            extracted_bridging['item']['id'] = bridging_id
        if 'prev_item' in extracted_bridging:
            extracted_bridging['prev_item']['relevant_prev_id'] = relevant_prev_id

        # Supporting item ids (assume same order as input)
        for i, extracted_item in enumerate(extracted_items):
            if i < len(expected_supporting_ids):
                extracted_item['id'] = expected_supporting_ids[i]

        # If any item has not_relevant=True, the model indicated content is not relevant
        # Don't retry - just return with verification failed
        if has_bridging_not_relevant(extracted_bridging) or has_not_relevant(extracted_items):
            return extracted_bridging, extracted_items, attempt, False

        # Verify the extracted quotes (including previous clues for prev_item verification)
        all_valid, _, _ = verify_extension_items(
            extracted_bridging, extracted_items, clues, items_and_contents,
            previous_clues, prev_items_and_contents
        )

        if all_valid:
            return extracted_bridging, extracted_items, attempt, True

    # All retries exhausted, return last extraction attempt
    return extracted_bridging, extracted_items, max_retries, False


def update_bridging_item(
    original_item: Dict,
    extracted_item: Dict[str, Any]
) -> bool:
    """Update original bridging_item with extracted quotes.

    New structure has:
    - item: bridging chunk with clue_quotes, item_quotes, reasoning
    - prev_item: previous chunk with clue_quotes, prev_item_quotes, reasoning
    """
    if not extracted_item:
        return False

    item = extracted_item.get("item", {})
    prev_item = extracted_item.get("prev_item", {})

    # Update item (bridging chunk) quotes
    original_item["clue_quotes"] = item.get("clue_quotes", [])
    original_item["item_quotes"] = item.get("item_quotes", [])
    original_item["reasoning"] = item.get("reasoning", "")
    # Track if model indicated content is not relevant
    if "not_relevant" in item:
        original_item["not_relevant"] = item["not_relevant"]

    # Add prev_item data
    original_item["prev_item"] = {
        "relevant_prev_id": prev_item.get("relevant_prev_id", ""),
        "clue_quotes": prev_item.get("clue_quotes", []),
        "prev_item_quotes": prev_item.get("prev_item_quotes", []),
        "reasoning": prev_item.get("reasoning", "")
    }
    # Track if model indicated content is not relevant for prev_item
    if "not_relevant" in prev_item:
        original_item["prev_item"]["not_relevant"] = prev_item["not_relevant"]

    return True


def format_supporting_items_for_prompt(supporting_items: List[Dict], items_and_contents: Dict[str, str]) -> str:
    """Format supporting items for the extraction prompt.

    Uses chunk_id in the prompt for model clarity (internally stored as 'id').
    """
    formatted = ""
    for item in supporting_items:
        item_id = item.get("id", "")
        reasoning = item.get("reasoning", "")
        content = items_and_contents.get(item_id, "")

        formatted += f"""    <item>
        <chunk_id>{item_id}</chunk_id>
        <reasoning>{reasoning}</reasoning>
        <content>
{content}
        </content>
    </item>
"""
    return formatted.strip()


def run_extraction(client, model: str, prompt: str) -> List[Dict[str, Any]]:
    """Run the extraction with the LLM."""
    response = client.messages.create(
        model=model,
        max_tokens=10000,
        thinking={"type": "enabled", "budget_tokens": 2000},
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = ""
    for item in response.content:
        if item.type == "text":
            response_text = item.text
            break

    return parse_supporting_items(response_text, include_quotes=True)


def run_single_item_extraction(client, model: str, prompt: str) -> Dict[str, Any] | None:
    """Run the extraction for a single item with the LLM."""
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        thinking={"type": "enabled", "budget_tokens": 2000},
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = ""
    for item in response.content:
        if item.type == "text":
            response_text = item.text
            break

    return parse_single_item(response_text)


def process_single_item(
    client,
    model: str,
    clues: str,
    question: str,
    truth: str,
    supporting_item: Dict,
    items_and_contents: Dict[str, str],
    max_retries: int = 3
) -> Tuple[Dict[str, Any] | None, int, bool]:
    """Process a single supporting item with retries."""
    item_id = supporting_item.get("id", "")
    item_reasoning = supporting_item.get("reasoning", "")
    item_content = items_and_contents.get(item_id, "")

    if not item_content:
        return None, 0, False

    prompt = SEC_SINGLE_ITEM_EXTRACTION_PROMPT.format(
        clues=clues,
        question=question,
        truth=truth,
        chunk_id=item_id,  # Use chunk_id in prompt for model clarity
        item_reasoning=item_reasoning,
        item_content=item_content
    )

    extracted_item = None
    for attempt in range(max_retries):
        extracted_item = run_single_item_extraction(client, model, prompt)

        if not extracted_item:
            continue

        # Ensure extracted item has the correct id - LLM may output different format/typo
        # Since we're asking the LLM to analyze a specific chunk, the id should always match
        extracted_item['id'] = item_id

        # If item has not_relevant=True, don't retry
        if extracted_item.get('not_relevant', False):
            return extracted_item, attempt + 1, False

        # Verify the extracted quotes for this single item
        verification = verify_supporting_item(extracted_item, clues, items_and_contents)

        is_valid = (
            verification['clue_quotes_valid'] and
            verification['item_quotes_valid'] and
            (not extracted_item.get('contains_truth', False) or verification['truth_quotes_valid'])
        )

        if is_valid:
            return extracted_item, attempt + 1, True

    return extracted_item, max_retries, False


def process_task(
    client,
    model: str,
    task: Dict,
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], Dict[str, int], bool]:
    """Process a task by extracting/verifying quotes for each item individually."""
    clues = task.get("clues", "")
    question = task.get("question", "")
    truth = task.get("truth", "")
    supporting_items = task.get("supporting_items", [])
    items_and_contents = task.get("items_and_contents", {})

    if not supporting_items or not items_and_contents:
        return [], {}, False

    extracted_items = []
    retry_counts = {}
    all_items_valid = True

    for supporting_item in supporting_items:
        item_id = supporting_item.get("id", "")
        extracted_item, retries_used, item_valid = process_single_item(
            client, model, clues, question, truth,
            supporting_item, items_and_contents, max_retries
        )

        retry_counts[item_id] = retries_used

        if extracted_item:
            extracted_items.append(extracted_item)
        else:
            # Failed to extract anything for this item
            all_items_valid = False

        if not item_valid:
            all_items_valid = False

    return extracted_items, retry_counts, all_items_valid


def update_supporting_items(
    original_items: List[Dict],
    extracted_items: List[Dict[str, Any]],
    items_and_contents: Dict[str, str]
) -> bool:
    """Update original supporting items with extracted quotes."""
    updated = False
    id_to_extracted = {item["id"]: item for item in extracted_items}

    for item in original_items:
        item_id = item.get("id", "")
        if item_id in id_to_extracted:
            extracted = id_to_extracted[item_id]
            item["clue_quotes"] = extracted["clue_quotes"]
            item["item_quotes"] = extracted["item_quotes"]
            item["reasoning"] = extracted["reasoning"]
            item["truth_quotes"] = extracted["truth_quotes"] if extracted["truth_quotes"] else []

            # contains_truth is true ONLY if truth_quotes has valid quotes AND they match the content
            truth_quotes = extracted["truth_quotes"] or []
            content = items_and_contents.get(item_id, "")
            # Filter out any "none" entries to get valid quotes
            valid_truth_quotes = [q for q in truth_quotes if q.lower() != 'none']
            if valid_truth_quotes and content and not content.startswith("Error"):
                # Check if all valid truth quotes are found in the content
                all_match = all(text_contains_quote(content, q) for q in valid_truth_quotes)
                item["contains_truth"] = all_match
            else:
                item["contains_truth"] = False

            if "not_relevant" in extracted:
                item["not_relevant"] = extracted["not_relevant"]
            updated = True

    return updated


def is_task_processed(task: Dict) -> bool:
    """Check if a task has been processed."""
    if "passed_verification" not in task:
        return False

    supporting_items = task.get("supporting_items", [])

    for item in supporting_items:
        if "clue_quotes" not in item or "item_quotes" not in item or "contains_truth" not in item:
            return False

    return len(supporting_items) > 0


def process_file(filepath: str, model: str, max_retries: int = 3) -> Dict[str, Any]:
    """Process a single file.

    Handles both:
    - Level 0 tasks (3 supporting items, single-item extraction)
    - Extension tasks (2 supporting items + bridging_item)
    """
    client = get_anthropic_client()

    with open(filepath, "r") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    if not tasks:
        return {"status": "skipped", "reason": "no tasks", "passed_verification": None}

    updated = False
    passed_verification = None

    # Find the task to verify (last task by level without passed_verification)
    task_to_verify, prev_task = get_task_to_verify(tasks)

    if task_to_verify is None:
        # All tasks already processed
        return {"status": "skipped", "reason": "already processed", "passed_verification": None}

    try:
        # Check if this is an extension task (has bridging_item and previous task)
        bridging_item = task_to_verify.get("bridging_item", {})
        is_extension = bool(bridging_item) and prev_task is not None

        if is_extension:
            # Process as extension task
            relevant_prev_id = bridging_item.get("relevant_prev_id", "")
            prev_items_and_contents = prev_task.get("items_and_contents", {})

            if relevant_prev_id not in prev_items_and_contents:
                print(f"Warning: relevant_prev_id '{relevant_prev_id}' not in previous task in {filepath}")
                task_to_verify["passed_verification"] = False
                task_to_verify["failure_reasons"] = [f"Bridging ID '{relevant_prev_id}' not found in previous task's items"]
                task_to_verify["extraction_retry_count"] = 0
                passed_verification = False
                updated = True
            else:
                # Process extension task
                extracted_bridging, extracted_items, retry_count, quotes_verified = process_extension_task(
                    client, model, task_to_verify, prev_task, max_retries=max_retries
                )

                if not extracted_items:
                    print(f"Warning: No items extracted for extension task in {filepath}")
                    task_to_verify["passed_verification"] = False
                    task_to_verify["failure_reasons"] = ["No quotes could be extracted from the supporting items"]
                    task_to_verify["extraction_retry_count"] = max_retries
                    passed_verification = False
                    updated = True
                else:
                    # Update bridging_item
                    update_bridging_item(
                        task_to_verify.get("bridging_item", {}),
                        extracted_bridging
                    )

                    # Update supporting_items
                    update_supporting_items(
                        task_to_verify.get("supporting_items", []),
                        extracted_items,
                        task_to_verify.get("items_and_contents", {})
                    )

                    task_to_verify["extraction_retry_count"] = retry_count

                    # Check if at least one supporting_item contains the truth
                    truth_contained = check_truth_contained(task_to_verify.get("supporting_items", []))

                    task_to_verify["passed_verification"] = quotes_verified and truth_contained
                    passed_verification = task_to_verify["passed_verification"]

                    if not passed_verification:
                        # Get verification results to build failure reasons
                        _, bridging_verif, supporting_verifs = verify_extension_items(
                            extracted_bridging,
                            extracted_items,
                            task_to_verify.get("clues", ""),
                            task_to_verify.get("items_and_contents", {}),
                            prev_task.get("clues", ""),
                            prev_items_and_contents
                        )
                        task_to_verify["failure_reasons"] = build_failure_reasons(
                            quotes_verified=quotes_verified,
                            truth_contained=truth_contained,
                            verification_results=supporting_verifs,
                            bridging_verification=bridging_verif,
                            has_not_relevant_items=has_not_relevant(extracted_items),
                            has_bridging_not_relevant=has_bridging_not_relevant(extracted_bridging)
                        )

                    updated = True
        else:
            # Process as regular level 0 task (single-item extraction)
            extracted_items, retry_counts, all_items_valid = process_task(
                client, model, task_to_verify, max_retries=max_retries
            )

            if not extracted_items:
                print(f"Warning: No items extracted for task in {filepath}")
                task_to_verify["passed_verification"] = False
                task_to_verify["failure_reasons"] = ["No quotes could be extracted from the supporting items"]
                task_to_verify["extraction_attempts"] = retry_counts
                passed_verification = False
                updated = True
            else:
                task_updated = update_supporting_items(
                    task_to_verify.get("supporting_items", []),
                    extracted_items,
                    task_to_verify.get("items_and_contents", {})
                )

                task_to_verify["extraction_attempts"] = retry_counts
                truth_contained = check_truth_contained(task_to_verify.get("supporting_items", []))
                task_to_verify["passed_verification"] = all_items_valid and truth_contained
                passed_verification = task_to_verify["passed_verification"]

                if not passed_verification:
                    _, verification_results = verify_all_items(
                        extracted_items,
                        task_to_verify.get("clues", ""),
                        task_to_verify.get("items_and_contents", {})
                    )
                    task_to_verify["failure_reasons"] = build_failure_reasons(
                        quotes_verified=all_items_valid,
                        truth_contained=truth_contained,
                        verification_results=verification_results,
                        has_not_relevant_items=has_not_relevant(extracted_items)
                    )

                if task_updated:
                    updated = True

    except Exception as e:
        print(f"Error processing task in {filepath}: {e}")
        task_to_verify["passed_verification"] = False
        task_to_verify["failure_reasons"] = [f"Exception during processing: {str(e)}"]
        task_to_verify["extraction_attempts"] = {}
        passed_verification = False
        updated = True

    if updated:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    return {"status": "success", "updated": updated, "passed_verification": passed_verification}


def is_file_fully_processed(filepath: str) -> bool:
    """Check if a file has been fully processed."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        tasks = data.get("tasks", [])
        if not tasks:
            return True

        for task in tasks:
            if not is_task_processed(task):
                return False

        return True
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def get_valid_files(input_dir: str) -> List[str]:
    all_files = glob(os.path.join(input_dir, "*.json"))
    valid_files = []

    for filepath in all_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            tasks = data.get("tasks", [])
            if not tasks:
                continue

            # Check each task has required structure
            all_valid = True
            task = get_latest_task(tasks)

            if not task.get("clues") or not task.get("question") or not task.get("truth"):
                all_valid = False
                continue

            if not task.get("supporting_items") or not task.get("items_and_contents"):
                all_valid = False
                continue
            # Determine expected structure based on level
            level = task.get("level", 0)
            num_supporting = len(task.get("supporting_items", []))
            num_items_and_contents = len(task.get("items_and_contents", {}))

            if level == 0:
                # Level 0: 3 supporting items
                if num_supporting != 3:
                    all_valid = False
                    continue
                if num_items_and_contents != 3:
                    all_valid = False
                    continue
            else:
                # Extension tasks: 2 supporting items + bridging_item
                if num_supporting != 2:
                    all_valid = False
                    continue
                if num_items_and_contents != 3:
                    all_valid = False
                    continue
                if not task.get("bridging_item"):
                    all_valid = False
                    continue

            if all_valid:
                valid_files.append(filepath)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return valid_files


def run_batch(
    input_dir: str,
    model: str,
    max_workers: int = 8,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Run batch processing on all files."""
    valid_files = get_valid_files(input_dir)
    files_to_process = [f for f in valid_files if not is_file_fully_processed(f)]

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

        def process_with_model(filepath):
            return process_file(filepath, model, max_retries=max_retries)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_with_model, f): f
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
    parser = argparse.ArgumentParser(description="Verify and extract quotes from supporting items in SEC filings.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing JSON files")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Maximum number of files to process in parallel (default: 8)")
    parser.add_argument("--max-retries", "-r", type=int, default=3, help="Maximum extraction retries per item on verification failure (default: 3)")
    parser.add_argument("--model", "-m", type=str, default="claude-opus-4-5", help="Model to use (default: claude-opus-4-5)")
    parser.add_argument("--mode", type=str, default="main", choices=["main", "collect"],
                        help="Verification mode: 'main' for supporting items, 'collect' for additional chunks (default: main)")
    parser.add_argument("--single-file", "-f", type=str, help="Process a single file instead of a directory")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        exit(1)

    print(f"Input directory: {args.input}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Max workers (files in parallel): {args.max_workers}")
    print(f"Max retries (per item): {args.max_retries}")
    print("-" * 40)

    if args.mode == "collect":
        # Collect verification mode
        if args.single_file:
            print(f"Processing single file: {args.single_file}")
            result = process_collect_file(args.single_file, args.model, max_retries=args.max_retries)
            print(f"Result: {json.dumps(result, indent=2)}")
        else:
            result = run_collect_batch(
                args.input,
                args.model,
                max_workers=args.max_workers,
                max_retries=args.max_retries
            )

            print("-" * 40)
            print(f"Total valid files: {result['total_valid']}")
            print(f"Already processed: {result['already_processed']}")
            print(f"Processed: {result['processed']}")
            print(f"Successful: {result['successful']}")
            print(f"Total original chunks: {result['total_original_chunks']}")
            print(f"Total filtered chunks: {result['total_filtered_chunks']}")
            print(f"Filtered out: {result['filtered_out']}")

            if result['errors']:
                print("\nErrors:")
                for err in result['errors']:
                    print(f"  {err['file']}: {err['error']}")
    else:
        # Main verification mode
        result = run_batch(
            args.input,
            args.model,
            max_workers=args.max_workers,
            max_retries=args.max_retries
        )

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
