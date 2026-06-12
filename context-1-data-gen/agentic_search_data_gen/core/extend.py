"""Base class for extension agents that create multi-hop retrieval tasks."""
import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .utils import parse_tag


def get_latest_task(tasks: List[Dict]) -> Dict | None:
    """Get the latest task by level from a list of tasks."""
    if not tasks:
        return None
    sorted_tasks = sorted(tasks, key=lambda t: t.get("level", 0))
    return sorted_tasks[-1]


def get_latest_verified_task(tasks: List[Dict]) -> Dict | None:
    """Get the latest task by level that has passed_verification=True."""
    if not tasks:
        return None
    verified_tasks = [t for t in tasks if t.get("passed_verification") is True]
    if not verified_tasks:
        return None
    sorted_tasks = sorted(verified_tasks, key=lambda t: t.get("level", 0))
    return sorted_tasks[-1]


class BaseExtenderAgent(ABC):
    """Base class for agents that extend existing tasks with additional hops."""

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
    def format_initial_prompt(self, prev_task: Dict[str, Any], **kwargs) -> str:
        """Format the initial prompt for extending from a previous task."""
        pass

    @abstractmethod
    def get_force_output_message(self) -> str:
        """Return the message to force output when max iterations reached."""
        pass

    @abstractmethod
    def get_item_content(self, item_id: str, context: Dict[str, Any]) -> str:
        """Fetch the content for a supporting item by its ID."""
        pass

    def parse_bridging_item(self, content: str) -> Dict[str, Any] | None:
        """Parse bridging_item from XML format."""
        outer_match = re.search(r'<bridging_item>(.*?)</bridging_item>', content, re.DOTALL)
        if outer_match:
            item_content = outer_match.group(1)
            id_match = re.search(rf'<{self.item_id_tag}>(.*?)</{self.item_id_tag}>', item_content, re.DOTALL)
            relevant_prev_match = re.search(rf'<relevant_prev_{self.item_id_tag}>(.*?)</relevant_prev_{self.item_id_tag}>', item_content, re.DOTALL)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', item_content, re.DOTALL)

            if id_match:
                return {
                    self.item_id_tag: id_match.group(1).strip(),
                    f'relevant_prev_{self.item_id_tag}': relevant_prev_match.group(1).strip() if relevant_prev_match else '',
                    'reasoning': reasoning_match.group(1).strip() if reasoning_match else ''
                }
        return None

    def parse_supporting_items(self, content: str) -> List[Dict[str, Any]]:
        """Parse supporting_items from XML format."""
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
            "new_clues": parse_tag(content, "new_clues"),
            "question": parse_tag(content, "question"),
            "truth": parse_tag(content, "truth"),
            "bridging_item": self.parse_bridging_item(content),
            "supporting_items": self.parse_supporting_items(content)
        }

    def on_tool_result(self, tool_name: str, tool_args: Dict[str, Any], output: str, context: Dict[str, Any]) -> None:
        """Hook called after each tool execution. Override to track visited items, etc."""
        pass

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

            # Execute tools in parallel when multiple tools are called
            if len(tool_use_items) > 1:
                tool_results = []
                with ThreadPoolExecutor(max_workers=min(len(tool_use_items), 10)) as executor:
                    # Submit all tool calls
                    future_to_tool = {
                        executor.submit(self.execute_tool, tool_call.name, tool_call.input, i, context): tool_call
                        for tool_call in tool_use_items
                    }
                    # Collect results
                    results_map = {}
                    for future in as_completed(future_to_tool):
                        tool_call = future_to_tool[future]
                        results_map[tool_call.id] = future.result()
                    # Process in original order for tool_results, on_tool_result hooks, and trajectory
                    for tool_call in tool_use_items:
                        output = results_map[tool_call.id]
                        self.on_tool_result(tool_call.name, tool_call.input, output, context)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": output
                        })
                        trajectory.append({
                            "type": "tool_call",
                            "tool_name": tool_call.name,
                            "arguments": tool_call.input,
                            "output": output
                        })
            else:
                # Single tool call - no need for parallelization overhead
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

                    trajectory.append({
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "output": output
                    })

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

    def is_valid(self, filepath: str) -> bool:
        """Check if file is valid for extension (has at least one task with passed_verification=True)."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            tasks = data.get("tasks", [])
            if not tasks:
                return False

            latest_task = get_latest_task(tasks)
            if not latest_task["passed_verification"]:
                return False

            if not "distractors_and_contents" in latest_task:
                return False

            if not latest_task["distractors_passed_verification"]:
                return False

            return True

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def is_processed(self, filepath: str) -> bool:
        """Check if file has been fully processed (has a complete task at next level)."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            tasks = data.get("tasks", [])
            if not tasks:
                return False

            latest_task = get_latest_task(tasks)
            if latest_task is None:
                return False

            if latest_task.get("clues") is None or latest_task.get("question") is None or latest_task.get("truth") is None:
                return False

            bridging_item = latest_task.get("bridging_item")
            if bridging_item is None or not bridging_item.get(self.item_id_tag):
                return False

            supporting_items = latest_task.get("supporting_items", [])
            if len(supporting_items) != 2:
                return False
            for item in supporting_items:
                if not item.get(self.item_id_tag):
                    return False

            items_and_contents = latest_task.get("items_and_contents", {})
            if len(items_and_contents) != 3:
                return False

            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    @abstractmethod
    def run_single(self, input_filepath: str) -> Dict[str, Any]:
        """Run extension on a single file. Implementation varies by domain."""
        pass

    def build_result(self, parsed: Dict[str, Any] | None, prev_task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the new task dict from parsed output."""
        new_clues = parsed["new_clues"] if parsed else None
        question = parsed["question"] if parsed else None
        truth = parsed["truth"] if parsed else None
        bridging_item = parsed["bridging_item"] if parsed else None
        supporting_items = parsed["supporting_items"] if parsed else []

        items_and_contents = {}
        failed_items = []

        all_item_ids = []
        if bridging_item and bridging_item.get(self.item_id_tag):
            all_item_ids.append(bridging_item[self.item_id_tag])
        for item in supporting_items:
            if item.get(self.item_id_tag):
                all_item_ids.append(item[self.item_id_tag])

        for item_id in all_item_ids:
            content = self.get_item_content(item_id, context)
            if content and not content.startswith("Error"):
                items_and_contents[item_id] = content
            else:
                failed_items.append({"id": item_id, "error": content or "Unknown error"})

        prev_level = prev_task.get("level", 0)

        return {
            "level": prev_level + 1,
            "clues": new_clues,
            "question": question,
            "truth": truth,
            "bridging_item": bridging_item,
            "supporting_items": supporting_items,
            "items_and_contents": items_and_contents,
            "failed_items": failed_items
        }

    def run_batch(self, input_dir: str, max_workers: int = 8) -> Dict[str, Any]:
        """Run batch processing with parallel workers."""
        from glob import glob

        all_files = glob(os.path.join(input_dir, "*.json"))
        valid_files = [f for f in all_files if self.is_valid(f)]
        files_to_process = [f for f in valid_files if not self.is_processed(f)]

        results = []
        errors = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"Processing {len(files_to_process)}/{len(valid_files)} valid files",
                total=len(files_to_process)
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.run_single, f): f
                    for f in files_to_process
                }

                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        result = future.result()
                        results.append({"file": filepath, "status": "success", "result": result})
                    except Exception as e:
                        errors.append({"file": filepath, "error": str(e)})
                    progress.advance(task)

        return {
            "total": len(all_files),
            "valid": len(valid_files),
            "processed": len(files_to_process),
            "successful": len(results),
            "failed": len(errors),
            "errors": errors
        }
