import os
import argparse
from typing import Dict, Any, List
from dotenv import load_dotenv

from ...core.distract import BaseDistractorAgent
from ...core.utils import get_anthropic_client
from .prompts import WEB_DISTRACTORS_PROMPT, WEB_DISTRACTORS_FORCE_OUTPUT_PROMPT
from .utils import (
    get_page, search, format_search_results,
    TOOLS, handle_long_page, normalize_item, denormalize_item,
    truncate_long_page
)
from ...core.utils import strip_links

load_dotenv()


class WebDistractorAgent(BaseDistractorAgent):
    """Web-specific distractor mining agent."""

    item_id_tag = "url"
    system_prompt = "You are a helpful assistant that finds distractor pages for search tasks."

    def __init__(self, model: str = "claude-sonnet-4-5", max_iterations: int = 15):
        client = get_anthropic_client()
        super().__init__(client, model, max_iterations)

    def get_tools(self) -> List[Dict[str, Any]]:
        return TOOLS

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], iteration: int, context: Dict[str, Any]) -> str:
        input_messages = context.get("input_messages", [])
        long_page_contents = context.get("long_page_contents", {})

        if tool_name == "get_page":
            url = tool_args.get("url")
            if not url:
                return "Error: 'url' parameter is required for get_page tool"

            page_content = get_page(url)

            output, was_long_page, search_query, content_for_save = handle_long_page(
                self.client, url, page_content, input_messages, iteration
            )

            # Save content for later use (truncated if it was a long page)
            if was_long_page and content_for_save:
                long_page_contents[url] = strip_links(content_for_save)
            elif not page_content.startswith("Error"):
                long_page_contents[url] = strip_links(truncate_long_page(page_content))

            return output

        elif tool_name == "search":
            query = tool_args.get("query")
            if not query:
                return "Error: 'query' parameter is required for search tool"
            search_results = search(query)
            return f"[Tool call #{iteration+1}] {format_search_results(search_results)}"

        return f"Unknown tool: {tool_name}"

    def format_distractors_prompt(self, clues: str, question: str, truth: str, formatted_items_and_contents: str, prev_info: str) -> str:
        return WEB_DISTRACTORS_PROMPT.format(
            clues=clues,
            question=question,
            truth=truth,
            supporting_items=formatted_items_and_contents,
            prev_info=prev_info
        )

    def get_force_output_message(self) -> str:
        return WEB_DISTRACTORS_FORCE_OUTPUT_PROMPT

    def get_item_content(self, item_id: str, context: Dict[str, Any]) -> str:
        long_page_contents = context.get("long_page_contents", {})
        if item_id in long_page_contents:
            return long_page_contents[item_id]

        # Fallback: fetch and truncate if not in cache
        content = get_page(item_id)
        if not content.startswith("Error"):
            return strip_links(truncate_long_page(content))
        return None

    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert url -> id for JSON storage."""
        return normalize_item(item)

    def denormalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert id -> url for internal processing."""
        return denormalize_item(item)


def main():
    parser = argparse.ArgumentParser(description="Run the WebDistractorAgent to find distractors for search tasks.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input file path (JSON) or directory containing JSON files")
    parser.add_argument("--max-iterations", "-n", type=int, default=15, help="Maximum iterations per task (default: 15)")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Maximum number of parallel workers (default: 8)")
    parser.add_argument("--model", "-m", type=str, default="claude-sonnet-4-5", help="Model to use (default: claude-sonnet-4-5)")
    parser.add_argument("--level-filter", "-l", type=int, default=None, help="Level filter (default: None)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
        exit(1)

    agent = WebDistractorAgent(model=args.model, max_iterations=args.max_iterations)

    if os.path.isdir(args.input):
        print(f"Running batch mode on directory: {args.input}")
        print(f"Model: {args.model}")
        print(f"Max workers: {args.max_workers}")
        print(f"Max iterations: {args.max_iterations}")
        print("-" * 40)

        result = agent.run_batch(args.input, level_filter=args.level_filter, max_workers=args.max_workers)

        print("-" * 40)
        print(f"Total: {result['total']}")
        print(f"Skipped: {result['skipped']}")
        print(f"Successful: {result['successful']}")
        print(f"Failed: {result['failed']}")
        if result['errors']:
            print("\nErrors:")
            for err in result['errors']:
                print(f"  {err['file']}: {err['error']}")
    else:
        if not agent.is_valid(args.input, level_filter=args.level_filter):
            print(f"Error: Input file is not valid (missing tasks array or tasks have missing values)")
            exit(1)

        if agent.is_processed(args.input):
            print(f"File already processed (all tasks have distractors)")
            exit(0)

        print(f"Processing: {args.input}")
        print(f"Model: {args.model}")
        print(f"Max iterations: {args.max_iterations}")
        print("-" * 40)

        result = agent.run_single(args.input)

        processed_count = sum(1 for task in result["tasks"] if "distractors" in task and len(task.get("distractors", [])) > 0)
        print(f"Done. {processed_count}/{len(result['tasks'])} tasks have distractors.")


if __name__ == "__main__":
    main()
