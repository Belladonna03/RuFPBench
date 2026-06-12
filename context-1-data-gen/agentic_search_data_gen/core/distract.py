"""Base class for distractor mining agents."""
import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


class BaseDistractorAgent(ABC):
    """Base class for distractor mining agents."""

    item_id_tag: str = "id"
    system_prompt: str = "You are a helpful assistant that finds distractor pages for search tasks."

    def __init__(self, client, model: str, max_iterations: int = 15):
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
    def format_distractors_prompt(self, clues: str, question: str, truth: str, formatted_items_and_contents: str, prev_info: str) -> str:
        """Format the prompt for finding distractors."""
        pass

    @abstractmethod
    def get_force_output_message(self) -> str:
        """Return the message to force output when max iterations reached."""
        pass

    @abstractmethod
    def get_item_content(self, item_id: str, context: Dict[str, Any]) -> str:
        """Fetch the content for a distractor item by its ID.
        Should check context['long_page_contents'] first before fetching."""
        pass

    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert domain-specific item format to normalized format with 'id' key.
        Override this in subclasses that use different identifiers (e.g., url -> id)."""
        return item

    def denormalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert normalized format with 'id' key back to domain-specific format.
        Override this in subclasses that use different identifiers (e.g., id -> url)."""
        return item

    def parse_distractors(self, content: str) -> List[Dict[str, Any]]:
        """Parse distractor items from the agent output."""
        distractors = []
        distractors_match = re.search(r'<distractors>(.*?)</distractors>', content, re.DOTALL)
        if not distractors_match:
            return distractors

        distractors_content = distractors_match.group(1)
        distractor_matches = re.findall(r'<distractor>(.*?)</distractor>', distractors_content, re.DOTALL)

        for match in distractor_matches:
            id_match = re.search(rf'<{self.item_id_tag}>(.*?)</{self.item_id_tag}>', match, re.DOTALL)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', match, re.DOTALL)

            if id_match:
                distractors.append({
                    self.item_id_tag: id_match.group(1).strip(),
                    "reasoning": reasoning_match.group(1).strip() if reasoning_match else ""
                })

        return distractors

    def run_agent_loop(self, input_messages: List, context: Dict[str, Any]) -> List[Dict[str, Any]] | None:
        """Run the main agent loop with tools."""
        request_body = {
            "model": self.model,
            "system": self.system_prompt,
            "max_tokens": 20000,
            "tools": self.get_tools(),
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000}
        }
        distractors_output = None

        for i in range(self.max_iterations):
            request_body["messages"] = input_messages
            response = self.client.messages.create(**request_body)

            tool_use_items = [item for item in response.content if getattr(item, 'type', None) == 'tool_use']
            text_items = [item for item in response.content if getattr(item, 'type', None) == 'text']

            if not tool_use_items:
                for item in text_items:
                    if item.type == "text":
                        distractors_output = self.parse_distractors(item.text)
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
                    # Collect results in submission order
                    results_map = {}
                    for future in as_completed(future_to_tool):
                        tool_call = future_to_tool[future]
                        results_map[tool_call.id] = future.result()
                    # Preserve original order
                    for tool_call in tool_use_items:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": results_map[tool_call.id]
                        })
            else:
                # Single tool call - no need for parallelization overhead
                tool_results = []
                for tool_call in tool_use_items:
                    output = self.execute_tool(tool_call.name, tool_call.input, i, context)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": output
                    })

            input_messages.append({"role": "user", "content": tool_results})

        return distractors_output

    def force_output(self, input_messages: List) -> List[Dict[str, Any]] | None:
        """Force the agent to produce output when max iterations reached."""
        input_messages.append({"role": "user", "content": self.get_force_output_message()})

        response = self.client.messages.create(
            model=self.model,
            system=self.system_prompt,
            max_tokens=20000,
            messages=input_messages,
            thinking={"type": "enabled", "budget_tokens": 2000}
        )

        text_items = [item for item in response.content if getattr(item, 'type', None) == 'text']
        distractors_output = None

        for item in text_items:
            if item.type == "text":
                distractors_output = self.parse_distractors(item.text)

        return distractors_output

    def fetch_distractor_contents(self, distractors: List[Dict[str, Any]], context: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Fetch content for each distractor and build distractors_and_contents.

        Returns:
            tuple: (distractors_list, distractors_and_contents_dict)
        """
        valid_distractors = []
        distractors_and_contents = {}

        for d in distractors:
            item_id = d.get(self.item_id_tag, "")
            content = self.get_item_content(item_id, context) if item_id else ""
            if content and not content.startswith("Error fetching page:"):
                normalized = self.normalize_item(d)
                valid_distractors.append(normalized)
                # Use normalized 'id' key for distractors_and_contents
                normalized_id = normalized.get("id", item_id)
                distractors_and_contents[normalized_id] = content

        return valid_distractors, distractors_and_contents

    def _find_distractors_for_task(self, clues: str, question: str, truth: str, formatted_items_and_contents: str, prev_info: str) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Find distractors for a single task.

        Returns:
            tuple: (distractors_list, distractors_and_contents_dict)
        """
        formatted_prompt = self.format_distractors_prompt(
            clues=clues,
            question=question,
            truth=truth,
            formatted_items_and_contents=formatted_items_and_contents,
            prev_info=prev_info
        )

        input_messages = [{"role": "user", "content": formatted_prompt}]
        context = {
            "input_messages": input_messages,
            "long_page_contents": {}
        }

        distractors_output = self.run_agent_loop(input_messages, context)

        if distractors_output is None:
            distractors_output = self.force_output(input_messages)

        if distractors_output:
            return self.fetch_distractor_contents(distractors_output, context)

        return [], {}

    def format_items_and_contents(self, items_and_contents: Dict[str, str], max_content_length: int = 3000) -> str:
        """Format items_and_contents dict for the prompt."""
        formatted = ""
        for item_id, content in items_and_contents.items():
            truncated_content = content[:max_content_length] if len(content) > max_content_length else content
            formatted += f"{self.item_id_tag.upper()} ({item_id}):\n{truncated_content}\n\n"
        return formatted

    def run_single(self, input_filepath: str) -> Dict[str, Any]:
        """Process a single file to find distractors for all tasks."""
        with open(input_filepath, "r") as f:
            data = json.load(f)

        # Denormalize items_and_contents keys when reading
        for task in data.get("tasks", []):
            if "items_and_contents" in task:
                task["items_and_contents"] = {
                    self.denormalize_item({"id": k}).get(self.item_id_tag, k): v
                    for k, v in task["items_and_contents"].items()
                }

        prev_info = ""

        for task_idx, task in enumerate(data["tasks"]):
            # Skip tasks that already have >= 8 distractors
            if "distractors" in task and len(task.get("distractors", [])) >= 8:
                distractors = task.get("distractors", [])
                all_complete = True
                for distractor in distractors:
                    if not distractor.get("id") or not distractor.get("content") or not distractor.get("reasoning"):
                        all_complete = False
                        break
                if all_complete:
                    continue

            items_and_contents = task.get("items_and_contents", {})
            clues = task.get("clues", "")
            question = task.get("question", "")
            truth = task.get("truth", "")

            if not clues or not question or not truth or not items_and_contents:
                continue

            formatted_items_and_contents = self.format_items_and_contents(items_and_contents)
            distractors, distractors_and_contents = self._find_distractors_for_task(clues, question, truth, formatted_items_and_contents, prev_info)

            prev_info += f"Clues: {clues}\nTruth: {truth}\n\n ----------------------------------------\n\n"
            data["tasks"][task_idx]["distractors"] = distractors
            data["tasks"][task_idx]["distractors_and_contents"] = distractors_and_contents

        # Normalize items_and_contents keys when writing
        for task in data.get("tasks", []):
            if "items_and_contents" in task:
                task["items_and_contents"] = {
                    self.normalize_item({self.item_id_tag: k}).get("id", k): v
                    for k, v in task["items_and_contents"].items()
                }

        with open(input_filepath, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Processed {input_filepath}")

        return data

    def is_valid(self, filepath: str, level_filter: int = None) -> bool:
        """Check if a file is valid for processing."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return False

        if "tasks" not in data or len(data["tasks"]) == 0:
            return False

        for task in data["tasks"]:
            if not isinstance(task, dict):
                return False
            if task.get("clues") is None or task.get("question") is None or task.get("truth") is None:
                return False
            if task.get("items_and_contents") is None or len(task.get("items_and_contents", {})) == 0:
                return False
            if not task.get("passed_verification"):
                return False

        if level_filter is not None and level_filter != len(data["tasks"]) - 1:
            return False

        return True

    def is_processed(self, filepath: str) -> bool:
        """Check if file has been fully processed with distractors for all tasks."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return False

        if "tasks" not in data or len(data["tasks"]) == 0:
            return False

        for task in data["tasks"]:
            if "distractors_passed_verification" not in task:
                # Each non-filtered task should have at least 8 distractors
                distractors = task.get("distractors", [])
                if len(distractors) < 8:
                    return False

                # Check distractors_and_contents exists and has content for all distractors
                distractors_and_contents = task.get("distractors_and_contents", {})
                if not distractors_and_contents:
                    return False

                # Each distractor should have non-empty id, reasoning, and content in distractors_and_contents
                for distractor in distractors:
                    distractor_id = distractor.get("id")
                    if not distractor_id:
                        return False
                    if not distractor.get("reasoning"):
                        return False
                    if distractor_id not in distractors_and_contents:
                        return False

        return True

    def run_batch(self, input_dir: str, level_filter: int = None, max_workers: int = 8) -> Dict[str, Any]:
        """Run batch processing on all valid files in directory."""
        results = []
        errors = []

        json_files = glob(os.path.join(input_dir, "*.json"))
        files_to_process = [f for f in json_files if self.is_valid(f, level_filter) and not self.is_processed(f)]
        skipped = len(json_files) - len(files_to_process)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                f"Processing {len(files_to_process)}/{len(json_files)} files",
                total=len(files_to_process)
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.run_single, filepath): filepath
                    for filepath in files_to_process
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
            "total": len(json_files),
            "successful": len(results),
            "failed": len(errors),
            "skipped": skipped,
            "errors": errors
        }
