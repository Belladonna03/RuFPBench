import os
import re
import argparse
import json
import random
import time
import uuid
from typing import Dict, Any, List
from dotenv import load_dotenv

from ...core.explore import BaseExplorerAgent
from .prompts import EPSTEIN_TRUTH_TYPES, EPSTEIN_EXPLORATION_PROMPT
from .utils import (
    hybrid_search_across_all,
    grep_across_all,
    search_across_person,
    get_random_across_person,
    get_thread,
    get_random_seed_threads,
    get_anthropic_client,
)

load_dotenv()

TOOLS = [
    {
        "name": "hybrid_search_across_all",
        "description": "Semantic + keyword search across all emails. Use natural language queries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The natural language query to search for"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "grep_across_all",
        "description": "Regex search for exact patterns/names across all emails. Use for finding specific people, phrases, or keywords.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for"}
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "search_across_person",
        "description": "Search within a specific person's emails. Use when you've identified a person of interest.",
        "input_schema": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "The person's name to search within."},
                "query": {"type": "string", "description": "The query to search for within that person's emails"}
            },
            "required": ["person", "query"]
        }
    },
    {
        "name": "get_random_across_person",
        "description": "Get random email samples from a person. Use to explore what a person discusses without a specific query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "The person's name to get random emails from."}
            },
            "required": ["person"]
        }
    },
    {
        "name": "get_thread",
        "description": "Read the full content of a thread by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thread_id": {"type": "string", "description": "The thread ID to retrieve"}
            },
            "required": ["thread_id"]
        }
    }
]

FORCE_OUTPUT_MESSAGE = """You have reached the maximum number of tool calls. Based on all the information you have gathered so far, you MUST now provide your final output immediately.

Output your response in the following format, no other text or formatting:
<clues>
{The 3 subtle clues which point to the common truth.}
</clues>
<question>
{The question which asks for the common truth.}
</question>
<truth>
{The one and only exact truth to the question.}
</truth>
<supporting_items>
    <item>
        <id>{ID of the thread for clue 1}</id>
        <reasoning>{Reasoning for why this thread supports the clue}</reasoning>
    </item>
    <item>
        <id>{ID of the thread for clue 2}</id>
        <reasoning>{Reasoning for why this thread supports the clue}</reasoning>
    </item>
    <item>
        <id>{ID of the thread for clue 3}</id>
        <reasoning>{Reasoning for why this thread supports the clue}</reasoning>
    </item>
</supporting_items>"""


class EpsteinExplorerAgent(BaseExplorerAgent):
    item_id_tag = "id"
    system_prompt = "You are a helpful assistant that creates challenging search questions based on email threads."

    def __init__(self, model: str = "claude-sonnet-4-5", max_iterations: int = 20):
        client = get_anthropic_client()
        super().__init__(client, model, max_iterations)

    def get_tools(self) -> List[Dict[str, Any]]:
        return TOOLS

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], iteration: int, context: Dict[str, Any]) -> str:
        collection_name = context.get("collection_name", "")

        if tool_name == "hybrid_search_across_all":
            query = tool_args.get("query")
            if not query:
                return "Error: 'query' parameter is required"
            result = hybrid_search_across_all(query)
            return f"[Tool call #{iteration+1}] hybrid_search_across_all('{query}'):\n{result}"

        elif tool_name == "grep_across_all":
            pattern = tool_args.get("pattern")
            if not pattern:
                return "Error: 'pattern' parameter is required"
            result = grep_across_all(pattern)
            return f"[Tool call #{iteration+1}] grep_across_all('{pattern}'):\n{result}"

        elif tool_name == "search_across_person":
            person = tool_args.get("person")
            query = tool_args.get("query")
            if not person or not query:
                return "Error: 'person' and 'query' parameters are required"
            result = search_across_person(person, query, collection_name)
            return f"[Tool call #{iteration+1}] search_across_person('{person}', '{query}'):\n{result}"

        elif tool_name == "get_random_across_person":
            person = tool_args.get("person")
            if not person:
                return "Error: 'person' parameter is required"
            result = get_random_across_person(person)
            return f"[Tool call #{iteration+1}] get_random_across_person('{person}'):\n{result}"

        elif tool_name == "get_thread":
            thread_id = tool_args.get("thread_id")
            if not thread_id:
                return "Error: 'thread_id' parameter is required"
            try:
                result = get_thread(thread_id)
                return f"[Tool call #{iteration+1}] get_thread('{thread_id}'):\n{result}"
            except KeyError:
                return f"Error: Thread ID '{thread_id}' not found"

        return f"Unknown tool: {tool_name}"

    def on_tool_result(self, tool_name: str, tool_args: Dict[str, Any], output: str, context: Dict[str, Any]) -> None:
        if tool_name == "get_thread":
            thread_id = tool_args.get("thread_id")
            if thread_id:
                context.setdefault("visited_threads", []).append(thread_id)

    def format_initial_prompt(self, seed_threads: str, truth_type: str) -> str:
        return EPSTEIN_EXPLORATION_PROMPT.format(
            seed_threads=seed_threads,
            truth_type=truth_type
        )

    def get_force_output_message(self) -> str:
        return FORCE_OUTPUT_MESSAGE

    def get_item_content(self, item_id: str, context: Dict[str, Any]) -> str:
        try:
            return get_thread(item_id)
        except KeyError:
            return f"Error: Thread ID '{item_id}' not found"

    def run_single(self, seed: str = None, output_dir: str = "../data/epstein/test") -> Dict[str, Any]:
        trajectory = []
        context = {
            "collection_name": f"temp_{uuid.uuid4().hex}",
            "visited_threads": []
        }

        seed_threads = get_random_seed_threads()
        formatted_seed_threads = "\n\n".join(seed_threads)
        truth_type = random.choice(EPSTEIN_TRUTH_TYPES)

        formatted_prompt = self.format_initial_prompt(formatted_seed_threads, truth_type)

        input_messages = [{"role": "user", "content": formatted_prompt}]
        trajectory.append({
            "type": "input_text",
            "output": formatted_prompt
        })

        parsed = self.run_agent_loop(input_messages, trajectory, context)

        if parsed is None or parsed.get("question") is None:
            parsed = self.force_output(input_messages, trajectory)

        result = self.build_result(parsed, context, extra_fields={"truth_type": truth_type})

        os.makedirs(output_dir, exist_ok=True)
        if seed:
            filename = f"{seed}.json"
        else:
            filename = f"epstein_{int(time.time())}_{random.randint(0, 999):03d}.json"

        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(result, f, indent=4)

        return result

    def run_batch(self, num_explorations: int, output_dir: str = "../data/epstein/test", max_workers: int = 8) -> Dict[str, Any]:
        seed_ids = [f"epstein_{i}" for i in range(num_explorations)]
        return super().run_batch(
            seeds=seed_ids,
            output_dir=output_dir,
            max_workers=max_workers,
            get_seed_id=lambda s: s,
            get_output_path=lambda s, d: os.path.join(d, f"{s}.json")
        )


def main():
    parser = argparse.ArgumentParser(description="Run the EpsteinExplorerAgent to generate search questions from email threads.")
    parser.add_argument("--num", "-n", type=int, default=1, help="Number of explorations to run (default: 1)")
    parser.add_argument("--output", "-o", type=str, default="../data/epstein/test", help="Output directory")
    parser.add_argument("--max-workers", "-w", type=int, default=8, help="Maximum parallel workers (default: 8)")
    parser.add_argument("--max-iterations", "-i", type=int, default=20, help="Maximum iterations per exploration (default: 20)")
    parser.add_argument("--model", "-m", type=str, default="claude-sonnet-4-5", help="Model to use (default: claude-sonnet-4-5)")

    args = parser.parse_args()

    print(f"Running {args.num} explorations")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output}")
    print(f"Max workers: {args.max_workers}")
    print(f"Max iterations: {args.max_iterations}")
    print("-" * 40)

    agent = EpsteinExplorerAgent(model=args.model, max_iterations=args.max_iterations)

    if args.num == 1:
        result = agent.run_single(output_dir=args.output)
        print("Exploration complete!")
        print(f"Question: {result['tasks'][0].get('question', 'N/A')}")
        print(f"Truth: {result['tasks'][0].get('truth', 'N/A')}")
    else:
        result = agent.run_batch(args.num, output_dir=args.output, max_workers=args.max_workers)
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
