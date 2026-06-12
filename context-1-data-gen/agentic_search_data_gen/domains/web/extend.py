import os
import argparse
import json
import random
from typing import Dict, Any, List

from ...core.extend import BaseExtenderAgent, get_latest_task, get_latest_verified_task
from ...core.utils import get_anthropic_client, strip_links
from .prompts import WEB_TRUTH_TYPES, WEB_EXTENSION_PROMPT, WEB_EXTENSION_FORCE_OUTPUT_MESSAGE
from .utils import (
    get_page, search, format_search_results, truncate_long_page,
    handle_long_page, count_tokens, TOOLS
)


class WebExtensionAgent(BaseExtenderAgent):
    """Extension agent for web-based multi-hop retrieval tasks."""

    item_id_tag = "url"
    system_prompt = "You are a helpful assistant that creates challenging questions for search based on real information."

    def __init__(self, model: str = "claude-sonnet-4-5", max_iterations: int = 20):
        client = get_anthropic_client()
        super().__init__(client, model, max_iterations)
        self.long_page_contents: Dict[str, str] = {}

    def get_tools(self) -> List[Dict[str, Any]]:
        return TOOLS

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], iteration: int, context: Dict[str, Any]) -> str:
        input_messages = context.get("input_messages", [])
        surfaced_urls = context.get("surfaced_urls", [])
        visited_urls = context.get("visited_urls", [])

        if tool_name == "get_page":
            url = tool_args.get("url")
            if not url:
                return "Error: 'url' parameter is required for get_page tool"

            page_content = get_page(url)
            visited_urls.append(url)

            output, was_long_page, search_query, content_for_save = handle_long_page(
                self.client, url, page_content, input_messages, iteration
            )

            # Save content for later use (truncated if it was a long page)
            if was_long_page and content_for_save:
                self.long_page_contents[url] = strip_links(content_for_save)
            elif not page_content.startswith("Error"):
                self.long_page_contents[url] = strip_links(truncate_long_page(page_content))

            return output

        elif tool_name == "search":
            query = tool_args.get("query")
            if not query:
                return "Error: 'query' parameter is required for search tool"

            search_results = search(query)
            surfaced_urls.extend([r["link"] for r in search_results])
            return f"[Tool call #{iteration+1}] {format_search_results(search_results)}"

        return f"Unknown tool: {tool_name}"

    def on_tool_result(self, tool_name: str, tool_args: Dict[str, Any], output: str, context: Dict[str, Any]) -> None:
        if tool_name == "get_page":
            url = tool_args.get("url")
            if url:
                context.get("visited_urls", []).append(url)

    def format_initial_prompt(self, prev_task: Dict[str, Any], **kwargs) -> str:
        truth_type = kwargs.get("truth_type", random.choice(WEB_TRUTH_TYPES))
        selected_prev_url = kwargs.get("selected_prev_url")
        items_and_contents = prev_task.get("items_and_contents", {})

        selected_content = items_and_contents.get(selected_prev_url, "")

        return WEB_EXTENSION_PROMPT.format(
            prev_clues=prev_task.get("clues", ""),
            prev_truth=prev_task.get("truth", ""),
            selected_prev_url=selected_prev_url,
            selected_prev_content=selected_content,
            truth_type=truth_type
        )

    def get_force_output_message(self) -> str:
        return WEB_EXTENSION_FORCE_OUTPUT_MESSAGE

    def get_item_content(self, item_id: str, context: Dict[str, Any]) -> str:
        """Fetch page content for a URL, using cached long page content if available."""
        if item_id in self.long_page_contents:
            return self.long_page_contents[item_id]

        content = get_page(item_id)
        if content.startswith("Error fetching page:"):
            return content

        return strip_links(truncate_long_page(content))

    def run_single(self, input_filepath: str) -> Dict[str, Any]:
        with open(input_filepath, "r") as f:
            data = json.load(f)

        tasks = data.get("tasks", [])
        prev_task = get_latest_verified_task(tasks)
        if prev_task is None:
            raise ValueError(f"No verified tasks found in {input_filepath}")

        # Remove any existing task at the next level (for reprocessing invalid extensions)
        next_level = prev_task.get("level", 0) + 1
        data["tasks"] = [t for t in tasks if t.get("level") != next_level]

        # Select a random URL from previous task's supporting items to build the bridge from
        prev_supporting_items = prev_task.get("supporting_items", [])
        prev_urls = [item.get("url", "") or item.get("id", "") for item in prev_supporting_items if item.get("contains_truth")]
        prev_urls = [u for u in prev_urls if u]
        if not prev_urls:
            raise ValueError(f"No URLs in previous task's supporting items in {input_filepath}")

        selected_prev_url = random.choice(prev_urls)

        truth_type = random.choice(WEB_TRUTH_TYPES)
        self.long_page_contents = {}

        trajectory = []
        surfaced_urls = data.get("surfaced_urls", [])
        visited_urls = data.get("visited_urls", [])

        formatted_prompt = self.format_initial_prompt(
            prev_task, truth_type=truth_type, selected_prev_url=selected_prev_url
        )

        input_messages = [{"role": "user", "content": formatted_prompt}]
        trajectory.append({
            "type": "input_text",
            "output": formatted_prompt
        })

        context = {
            "input_messages": input_messages,
            "surfaced_urls": surfaced_urls,
            "visited_urls": visited_urls
        }

        parsed = self.run_agent_loop(input_messages, trajectory, context)

        if parsed is None or parsed.get("new_clues") is None:
            parsed = self.force_output(input_messages, trajectory)

        new_task = self.build_result(parsed, prev_task, context)
        new_task["truth_type"] = truth_type

        # Set relevant_prev_url directly (not from model output)
        if new_task.get("bridging_item"):
            new_task["bridging_item"]["relevant_prev_url"] = selected_prev_url

        data["tasks"].append(new_task)
        data["surfaced_urls"] = list(set(surfaced_urls))
        data["visited_urls"] = list(set(visited_urls))

        with open(input_filepath, "w") as f:
            json.dump(data, f, indent=4)

        return data


def main():
    parser = argparse.ArgumentParser(description="Run the WebExtensionAgent to extend search questions with a second hop.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing JSON files from explore.py")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Maximum number of parallel workers (default: 8)")
    parser.add_argument("--max-iterations", "-n", type=int, default=20, help="Maximum iterations per file (default: 20)")
    parser.add_argument("--model", "-m", type=str, default="claude-sonnet-4-5", help="Model to use (default: claude-sonnet-4-5)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        exit(1)

    print(f"Input directory: {args.input}")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max iterations: {args.max_iterations}")
    print("-" * 40)

    agent = WebExtensionAgent(model=args.model, max_iterations=args.max_iterations)
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
