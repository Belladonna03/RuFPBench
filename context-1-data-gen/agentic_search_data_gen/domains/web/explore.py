import os
import re
import argparse
import json
import random
from typing import Dict, Any, List
from dotenv import load_dotenv

from ...core.explore import BaseExplorerAgent
from ...core.utils import get_anthropic_client, strip_links
from .prompts import WEB_TRUTH_TYPES, WEB_EXPLORATION_PROMPT, WEB_FORCE_OUTPUT_PROMPT
from .utils import (
    get_page, search, format_search_results,
    truncate_long_page, TOOLS, handle_long_page,
    count_tokens, normalize_item, denormalize_item
)

load_dotenv()


class WebExplorerAgent(BaseExplorerAgent):
    item_id_tag = "url"
    system_prompt = "You are a helpful assistant that creates challenging questions for web search based on real information."

    def __init__(self, model: str = "claude-sonnet-4-5", max_iterations: int = 20):
        client = get_anthropic_client()
        super().__init__(client, model, max_iterations)

    def get_tools(self) -> List[Dict[str, Any]]:
        return TOOLS

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], iteration: int, context: Dict[str, Any]) -> str:
        # Reset long page tracking for this tool call
        context["was_long_page"] = False
        context["search_query"] = None
        context["page_token_count"] = 0

        input_messages = context.get("input_messages", [])
        long_page_contents = context.get("long_page_contents", {})

        if tool_name == "get_page":
            url = tool_args.get("url")
            if not url:
                return "Error: 'url' parameter is required for get_page tool"

            page_content = get_page(url)
            context.setdefault("visited_urls", []).append(url)

            output, was_long_page, search_query, content_for_save = handle_long_page(
                self.client, url, page_content, input_messages, iteration
            )

            # Save content for later use (truncated if it was a long page)
            if was_long_page and content_for_save:
                long_page_contents[url] = strip_links(content_for_save)
                context["was_long_page"] = True
                context["search_query"] = search_query
                context["page_token_count"] = count_tokens(page_content)
            elif not page_content.startswith("Error"):
                long_page_contents[url] = strip_links(truncate_long_page(page_content))

            return output

        elif tool_name == "search":
            query = tool_args.get("query")
            if not query:
                return "Error: 'query' parameter is required for search tool"
            search_results = search(query)
            context.setdefault("surfaced_urls", []).extend([r["link"] for r in search_results])
            return f"[Tool call #{iteration+1}] {format_search_results(search_results)}"

        return f"Unknown tool: {tool_name}"

    def build_tool_trajectory_entry(self, tool_name: str, tool_args: Dict[str, Any], output: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "get_page" and context.get("was_long_page") and context.get("search_query"):
            return {
                "type": "long_page_search",
                "tool_name": "get_page",
                "arguments": {"url": tool_args.get("url"), "search_query": context["search_query"]},
                "token_count": context.get("page_token_count", 0),
                "output": output
            }
        return super().build_tool_trajectory_entry(tool_name, tool_args, output, context)

    def format_initial_prompt(self, seed_topic: str, initial_search_results: str, truth_type: str) -> str:
        return WEB_EXPLORATION_PROMPT.format(
            seed_topic=seed_topic,
            initial_search_results=initial_search_results,
            truth_type=truth_type
        )

    def get_force_output_message(self) -> str:
        return WEB_FORCE_OUTPUT_PROMPT

    def get_item_content(self, item_id: str, context: Dict[str, Any]) -> str:
        long_page_contents = context.get("long_page_contents", {})
        if item_id in long_page_contents:
            return long_page_contents[item_id]

        content = get_page(item_id)
        if not content.startswith("Error fetching page:"):
            return strip_links(truncate_long_page(content))
        return None

    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return normalize_item(item)

    def denormalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return denormalize_item(item)

    def run_single(self, seed: str, output_dir: str = "../data/web/test") -> Dict[str, Any]:
        trajectory = []
        context = {
            "surfaced_urls": [],
            "visited_urls": [],
            "long_page_contents": {}
        }

        seed_search_results = search(seed)
        formatted_seed_search_results = format_search_results(seed_search_results)
        context["surfaced_urls"].extend([result["link"] for result in seed_search_results])

        truth_type = random.choice(WEB_TRUTH_TYPES)
        formatted_prompt = self.format_initial_prompt(seed, formatted_seed_search_results, truth_type)

        input_messages = [{"role": "user", "content": formatted_prompt}]
        context["input_messages"] = input_messages
        trajectory.append({
            "type": "input_text",
            "output": formatted_prompt
        })

        parsed = self.run_agent_loop(input_messages, trajectory, context)

        if parsed is None or parsed.get("question") is None:
            parsed = self.force_output(input_messages, trajectory)

        result = self.build_result(parsed, context, extra_fields={
            "level": 0,
            "truth_type": truth_type
        })

        result["surfaced_urls"] = list(set(context["surfaced_urls"]))
        result["visited_urls"] = list(set(context["visited_urls"]))

        os.makedirs(output_dir, exist_ok=True)
        safe_filename = re.sub(r'[^\w\-_.]', '_', seed)
        with open(os.path.join(output_dir, f"{safe_filename}.json"), "w") as f:
            json.dump(result, f, indent=4)

        return result

    def run_batch(self, seeds: List[str], output_dir: str = "../data/web/test", max_workers: int = 8) -> Dict[str, Any]:
        def get_output_path(seed, output_dir):
            safe_filename = re.sub(r'[^\w\-_.]', '_', seed)
            return os.path.join(output_dir, f"{safe_filename}.json")

        return super().run_batch(
            seeds=seeds,
            output_dir=output_dir,
            max_workers=max_workers,
            get_seed_id=lambda s: s,
            get_output_path=get_output_path
        )


def main():
    parser = argparse.ArgumentParser(description="Run the WebExplorerAgent to generate search questions from seed topics.")
    parser.add_argument("--seeds", "-s", type=str, default="seeds.txt", help="Path to a txt file containing seed topics (one per line)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for generated trajectories")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Maximum number of parallel workers (default: 8)")
    parser.add_argument("--max-iterations", "-i", type=int, default=20, help="Maximum iterations per seed (default: 20)")
    parser.add_argument("--model", "-m", type=str, default="claude-sonnet-4-5", help="Model to use (default: claude-sonnet-4-5)")

    args = parser.parse_args()

    if not os.path.exists(args.seeds):
        print(f"Error: Seeds file not found: {args.seeds}")
        exit(1)

    with open(args.seeds, "r") as f:
        seeds = [line.strip() for line in f if line.strip()]

    if not seeds:
        print("Error: No seeds found in the input file")
        exit(1)

    print(f"Loaded {len(seeds)} seeds from {args.seeds}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max iterations: {args.max_iterations}")
    print("-" * 40)

    agent = WebExplorerAgent(model=args.model, max_iterations=args.max_iterations)
    result = agent.run_batch(seeds, output_dir=args.output, max_workers=args.max_workers)

    print("-" * 40)
    print(f"Total: {result['total']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    if result['errors']:
        print("\nErrors:")
        for err in result['errors']:
            print(f"  {err['seed']}: {err['error']}")


if __name__ == "__main__":
    main()
