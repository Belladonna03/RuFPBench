"""SEC-specific utility functions.

This module contains utilities specific to SEC filings processing.
Common text matching utilities are in core.utils.
"""
import re
from typing import Any, Dict, List


def get_latest_task(tasks: List[Dict]) -> Dict | None:
    """Get the latest task by level from a list of tasks."""
    if not tasks:
        return None
    sorted_tasks = sorted(tasks, key=lambda t: t.get("level", 0))
    return sorted_tasks[-1]


def parse_supporting_items(content: str, include_quotes: bool = False) -> List[Dict[str, Any]]:
    """Parse supporting_items from XML format.

    Parses chunk_id from model output and normalizes to 'id' internally.

    Args:
        content: The XML content to parse
        include_quotes: If True, also parse clue_quotes, item_quotes, truth_quotes,
                       contains_truth, and not_relevant fields (for verification).
                       If False, only parse id and reasoning (for exploration).

    Returns:
        List of dicts. Basic fields: 'id', 'reasoning'.
        With include_quotes=True: also includes 'clue_quotes', 'item_quotes',
        'truth_quotes', 'contains_truth', 'not_relevant'.
    """
    # Import parse_quotes only when needed to avoid circular imports
    if include_quotes:
        from ...core.utils import parse_quotes

    items = []
    outer_match = re.search(r'<supporting_items>(.*?)</supporting_items>', content, re.DOTALL)
    if not outer_match:
        return items

    items_content = outer_match.group(1)
    item_matches = re.findall(r'<item>(.*?)</item>', items_content, re.DOTALL)

    for item_match in item_matches:
        # Parse chunk_id from model output (normalize to 'id' internally)
        chunk_id_match = re.search(r'<chunk_id>(.*?)</chunk_id>', item_match, re.DOTALL)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_match, re.DOTALL)

        item_dict = {
            'id': chunk_id_match.group(1).strip() if chunk_id_match else '',
            'reasoning': reasoning_match.group(1).strip() if reasoning_match else ''
        }

        if include_quotes:
            contains_truth_match = re.search(r'<contains_truth>(.*?)</contains_truth>', item_match, re.DOTALL)

            clue_quotes = parse_quotes(item_match, 'clue_quotes')
            item_quotes = parse_quotes(item_match, 'item_quotes')
            truth_quotes = parse_quotes(item_match, 'truth_quotes')

            contains_truth = False
            if contains_truth_match:
                contains_truth_str = contains_truth_match.group(1).strip().lower()
                contains_truth = contains_truth_str == 'true'

            # Track if quotes are None (model indicated content is not relevant)
            not_relevant = clue_quotes is None or item_quotes is None

            item_dict.update({
                'clue_quotes': clue_quotes if clue_quotes is not None else [],
                'item_quotes': item_quotes if item_quotes is not None else [],
                'contains_truth': contains_truth,
                'truth_quotes': truth_quotes if truth_quotes is not None else [],
                'not_relevant': not_relevant
            })

        items.append(item_dict)

    return items


def parse_single_item(content: str) -> Dict[str, Any] | None:
    """Parse a single item from extraction response.

    Parses chunk_id from model output and normalizes to 'id' internally.

    Args:
        content: The XML content to parse

    Returns:
        Dict with 'id', 'clue_quotes', 'item_quotes', 'reasoning',
        'contains_truth', 'truth_quotes', 'not_relevant' fields.
        Returns None if no <item> tag found.
    """
    from ...core.utils import parse_quotes

    item_match = re.search(r'<item>(.*?)</item>', content, re.DOTALL)
    if not item_match:
        return None

    item_content = item_match.group(1)

    chunk_id_match = re.search(r'<chunk_id>(.*?)</chunk_id>', item_content, re.DOTALL)
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_content, re.DOTALL)
    contains_truth_match = re.search(r'<contains_truth>(.*?)</contains_truth>', item_content, re.DOTALL)

    clue_quotes = parse_quotes(item_content, 'clue_quotes')
    item_quotes = parse_quotes(item_content, 'item_quotes')
    truth_quotes = parse_quotes(item_content, 'truth_quotes')

    contains_truth = False
    if contains_truth_match:
        contains_truth = contains_truth_match.group(1).strip().lower() == 'true'

    not_relevant = clue_quotes is None or item_quotes is None

    return {
        'id': chunk_id_match.group(1).strip() if chunk_id_match else '',
        'clue_quotes': clue_quotes if clue_quotes is not None else [],
        'item_quotes': item_quotes if item_quotes is not None else [],
        'reasoning': reasoning_match.group(1).strip() if reasoning_match else '',
        'contains_truth': contains_truth,
        'truth_quotes': truth_quotes if truth_quotes is not None else [],
        'not_relevant': not_relevant
    }


def format_chunks(chunks: List[Dict[str, Any]], include_ticker: bool = False) -> str:
    """Format a list of chunks for display.

    Args:
        chunks: List of chunk dicts with 'id' and 'document' keys.
                Optionally 'ticker' key if include_ticker=True.
        include_ticker: If True, include ticker in the output format.

    Returns:
        Formatted string representation of chunks
    """
    formatted_str = ""
    for chunk in chunks:
        if include_ticker:
            ticker = chunk.get("ticker", "Unknown")
            formatted_str += f"\n\n---- Chunk ID: {chunk['id']} | Ticker: {ticker} ----\n{chunk['document']}\n"
        else:
            formatted_str += f"\n\n---- Chunk ID: {chunk['id']} ----\n{chunk['document']}\n"
    return formatted_str
