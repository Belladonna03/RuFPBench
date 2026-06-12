import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .utils import parse_tag


class BaseExplorerAgent(ABC):
    item_id_tag: str = "id"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self, client, model: str, max_iterations: int = 20):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations

    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return the list of tools available to this agent."""
        pass

    @abstractmethod
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], iteration: int, context: Dict[str, Any]) -> str:
        """Execute a tool and return the output string."""
        pass

    @abstractmethod
    def format_initial_prompt(self, **kwargs) -> str:
        """Format the initial prompt for the agent."""
        pass

    @abstractmethod
    def get_force_output_message(self) -> str:
        """Return the message to force output when max iterations reached."""
        pass

    @abstractmethod
    def get_item_content(self, item_id: str, context: Dict[str, Any]) -> str:
        """Fetch the content for a supporting item by its ID."""
        pass

    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert domain-specific item format to normalized format with 'id' key.
        Override this in subclasses that use different identifiers."""
        return item

    def denormalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert normalized format with 'id' key back to domain-specific format.
        Override this in subclasses that use different identifiers."""
        return item

    def parse_supporting_items(self, content: str) -> List[Dict[str, Any]]:
        """Parse supporting items from the agent output using domain-specific tag."""
        items = []
        outer_match = re.search(r'<supporting_items>(.*?)</supporting_items>', content, re.DOTALL)
        if outer_match:
            items_content = outer_match.group(1)
            item_matches = re.findall(r'<item>(.*?)</item>', items_content, re.DOTALL)
            for item_match in item_matches:
                id_match = re.search(rf'<{self.item_id_tag}>(.*?)</{self.item_id_tag}>', item_match, re.DOTALL)
                reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_match, re.DOTALL)
                if id_match:
                    items.append({
                        self.item_id_tag: id_match.group(1).strip(),
                        'reasoning': reasoning_match.group(1).strip() if reasoning_match else ''
                    })
        return items

    def parse_structured_output(self, content: str) -> Dict[str, Any]:
        """Parse the structured output from the agent."""
        return {
            "clues": parse_tag(content, "clues"),
            "question": parse_tag(content, "question"),
            "truth": parse_tag(content, "truth"),
            "supporting_items": self.parse_supporting_items(content)
        }

    def on_tool_result(self, tool_name: str, tool_args: Dict[str, Any], output: str, context: Dict[str, Any]) -> None:
        """Hook called after each tool execution. Override to track visited items, etc."""
        pass

    def build_tool_trajectory_entry(self, tool_name: str, tool_args: Dict[str, Any], output: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build a trajectory entry for a tool call. Override to customize."""
        return {
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": tool_args,
            "output": output
        }

    def run_agent_loop(self, input_messages: List, trajectory: List, context: Dict[str, Any]) -> Dict[str, Any] | None:
        """Run the main agent loop with tools."""
        request_body = {
            "model": self.model,
            "system": self.system_prompt,
            "max_tokens": 20000,
            "tools": self.get_tools(),
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000}
        }
        parsed = None

        for i in range(self.max_iterations):
            request_body["messages"] = input_messages
            response = self.client.messages.create(**request_body)

            tool_use_items = [item for item in response.content if getattr(item, 'type', None) == 'tool_use']
            thinking_items = [item for item in response.content if getattr(item, 'type', None) == 'thinking']
            text_items = [item for item in response.content if getattr(item, 'type', None) == 'text']

            if thinking_items:
                for thinking_item in thinking_items:
                    trajectory.append({
                        "type": "thinking",
                        "output": thinking_item.thinking
                    })

            if not tool_use_items:
                for item in text_items:
                    if item.type == "text":
                        content = item.text
                        parsed = self.parse_structured_output(content)
                        trajectory.append({
                            "type": "output_text",
                            "output": content,
                            **parsed
                        })
                break

            serialized_items = []
            for item in response.content:
                serialized_item = item.model_dump(mode="python")
                if 'status' in serialized_item:
                    del serialized_item['status']
                serialized_items.append(serialized_item)

            input_messages.append({"role": "assistant", "content": serialized_items})

            tool_results = []
            for tool_call in tool_use_items:
                tool_args = tool_call.input
                tool_name = tool_call.name

                output = self.execute_tool(tool_name, tool_args, i, context)
                self.on_tool_result(tool_name, tool_args, output, context)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": output
                })

                trajectory_entry = self.build_tool_trajectory_entry(tool_name, tool_args, output, context)
                trajectory.append(trajectory_entry)

            input_messages.append({"role": "user", "content": tool_results})

        return parsed

    def force_output(self, input_messages: List, trajectory: List) -> Dict[str, Any] | None:
        """Force the agent to produce output when max iterations reached."""
        input_messages.append({"role": "user", "content": self.get_force_output_message()})

        response = self.client.messages.create(
            model=self.model,
            system=self.system_prompt,
            max_tokens=20000,
            messages=input_messages,
            thinking={"type": "enabled", "budget_tokens": 2000}
        )

        thinking_items = [item for item in response.content if getattr(item, 'type', None) == 'thinking']
        text_items = [item for item in response.content if getattr(item, 'type', None) == 'text']

        if thinking_items:
            for thinking_item in thinking_items:
                trajectory.append({
                    "type": "thinking",
                    "output": thinking_item.thinking
                })

        parsed = None
        for item in text_items:
            if item.type == "text":
                content = item.text
                parsed = self.parse_structured_output(content)
                trajectory.append({
                    "type": "forced_output",
                    "output": content,
                    **parsed
                })

        return parsed

    def is_processed(self, filepath: str) -> bool:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            if not data.get("tasks") or len(data["tasks"]) == 0:
                return False

            task = data["tasks"][0]
            if task.get("clues") is None or task.get("question") is None or task.get("truth") is None:
                return False

            supporting_items = task.get("supporting_items", [])
            if len(supporting_items) != 3:
                return False


            unique_ids = set([item['id'] for item in supporting_items])

            if len(unique_ids) != 3:
                return False

            items_and_contents = task.get("items_and_contents", {})
            for item in supporting_items:
                item_id = item.get("id", "")
                if item_id and item_id not in items_and_contents:
                    return False

            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    @abstractmethod
    def run_single(self, **kwargs) -> Dict[str, Any]:
        """Run a single exploration. Implementation varies by domain."""
        pass

    def build_result(self, parsed: Dict[str, Any] | None, context: Dict[str, Any], extra_fields: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build the final result dict with normalized items."""
        clues = parsed["clues"] if parsed else None
        question = parsed["question"] if parsed else None
        truth = parsed["truth"] if parsed else None
        supporting_items = parsed["supporting_items"] if parsed else []

        items_and_contents = {}
        normalized_items = []
        for item in supporting_items:
            item_id = item.get(self.item_id_tag, "")
            normalized_item = self.normalize_item(item)
            normalized_items.append(normalized_item)

            if not item_id:
                continue
            content = self.get_item_content(item_id, context)
            if content:
                normalized_id = normalized_item.get("id", item_id)
                items_and_contents[normalized_id] = content

        task = {
            "clues": clues,
            "question": question,
            "truth": truth,
            "supporting_items": normalized_items,
            "items_and_contents": items_and_contents
        }
        if extra_fields:
            task.update(extra_fields)

        return {"tasks": [task]}

    def run_batch(self, seeds: List[Any], output_dir: str, max_workers: int = 8, get_seed_id=None, get_output_path=None) -> Dict[str, Any]:
        """Run batch processing with parallel workers."""
        results = []
        errors = []

        os.makedirs(output_dir, exist_ok=True)

        if get_seed_id is None:
            get_seed_id = lambda s: str(s)
        if get_output_path is None:
            get_output_path = lambda s, d: os.path.join(d, f"{get_seed_id(s)}.json")

        seeds_to_process = []
        for seed in seeds:
            output_path = get_output_path(seed, output_dir)
            if not os.path.exists(output_path) or not self.is_processed(output_path):
                seeds_to_process.append(seed)

        skipped = len(seeds) - len(seeds_to_process)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"Processing {len(seeds_to_process)}/{len(seeds)} explorations",
                total=len(seeds_to_process)
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_seed = {
                    executor.submit(self.run_single, seed=seed, output_dir=output_dir): seed
                    for seed in seeds_to_process
                }

                for future in as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    try:
                        result = future.result()
                        results.append({"seed": get_seed_id(seed), "status": "success", "result": result})
                    except Exception as e:
                        errors.append({"seed": get_seed_id(seed), "error": str(e)})
                    progress.advance(task)

        return {
            "total": len(seeds),
            "successful": len(results),
            "failed": len(errors),
            "skipped": skipped,
            "errors": errors
        }
