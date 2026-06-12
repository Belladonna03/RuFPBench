from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
import json
from pathlib import Path
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
import xml.etree.ElementTree as ET
from ...core.utils import get_anthropic_client
from .prompts import (
    NON_FINAL_REJECTION_EXTRACTION_SYSTEM_PROMPT,
    NON_FINAL_REJECTION_EXTRACTION_PROMPT
)

load_dotenv()


class Extractor:
    # Pre-compiled regex patterns
    _XML_CODE_BLOCK_START = re.compile(r'^```xml\s*')
    _CODE_BLOCK_END = re.compile(r'```\s*$')
    _REJECTIONS_TAG = re.compile(r'<rejections>(.*?)</rejections>', re.DOTALL)

    def __init__(self, anthropic_client, model="claude-opus-4-5"):
        self.anthropic_client = anthropic_client
        self.model = model

    def _load_json(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load JSON file, returning None on failure."""
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _check_file_status(self, filepath: str) -> tuple[bool, bool]:
        """Check if file has rejection and is processed in one read.

        Returns: (has_rejection, is_processed)
        """
        data = self._load_json(filepath)
        if data is None:
            return False, False
        has_rejection = "CTNF" in data and len(data.get("CTNF", {}).get("text", "")) > 0
        is_processed = "extraction_result" in data
        return has_rejection, is_processed

    def parse_rejections(self, xml_string):
        xml_string = self._XML_CODE_BLOCK_START.sub('', xml_string.strip())
        xml_string = self._CODE_BLOCK_END.sub('', xml_string.strip())

        # Extract content between <rejections> tags
        match = self._REJECTIONS_TAG.search(xml_string)
        if match:
            xml_content = f"<rejections>{match.group(1)}</rejections>"
        else:
            # If no tags found, assume the entire string is the XML content
            xml_content = xml_string.strip()

        root = ET.fromstring(xml_content)
        rejections = []

        for rejection in root.findall('rejection'):
            rej_dict = {
                'type': rejection.findtext('type'),
                'claim_number': rejection.findtext('claim_number'),
                'claim_text': rejection.findtext('claim_text'),
                'reasoning': rejection.findtext('reasoning'),
                'claim_element_mappings': []
            }


            # Parse claim element mappings
            for mapping in rejection.findall('.//claim_element_mappings/mapping'):
                mapping_dict = {
                    'claim_element': mapping.findtext('claim_element'),
                    'prior_art_element': mapping.findtext('prior_art_element'),
                    'mapping_strength': mapping.findtext('mapping_strength'),
                    'citations': []
                }
                for citation in mapping.findall('.//citations/citation'):
                    mapping_dict['citations'].append({
                        'name': citation.findtext('name'),
                        'locations': citation.findtext('locations')
                    })
                rej_dict['claim_element_mappings'].append(mapping_dict)

            rejections.append(rej_dict)


        return rejections

    def run_single(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r") as f:
            data = json.load(f)

        application_no = os.path.splitext(os.path.basename(filepath))[0]

        non_final_rejection = data["CTNF"]["text"]
        claims = data["CLM"]["text"]
        formatted_prompt = NON_FINAL_REJECTION_EXTRACTION_PROMPT.format(rejected_patent_claims=claims, non_final_rejection=non_final_rejection)

        with self.anthropic_client.messages.stream(
            model=self.model,
            system=NON_FINAL_REJECTION_EXTRACTION_SYSTEM_PROMPT,
            max_tokens=40000,
            messages=[{"role": "user", "content": formatted_prompt}],
            thinking={
                "type": "enabled",
                "budget_tokens": 5000
            }
        ) as stream:
            response = stream.get_final_message()

        text_items = [item for item in response.content if getattr(item, 'type', None) == 'text']
        text_only = text_items[0].text

        if text_only.lower() == "none":
            output = {
                "num_extracted": 0,
                "extracted": None
            }
        else:
            # Parse XML - raises on error, caught by _process_file
            extracted = self.parse_rejections(text_only)
            output = {
                "num_extracted": len(extracted),
                "extracted": extracted
            }

        data["application_no"] = application_no
        data["extraction_result"] = output

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        return output

    def _process_file(self, filepath: Path):
        try:
            result = self.run_single(str(filepath))
            return {"filepath": filepath, "success": True, "result": result}
        except Exception as e:
            return {"filepath": filepath, "success": False, "error": str(e)}

    def run_batch(self, input_dir: str, max_workers: int = 8):
        console = Console()
        input_path = Path(input_dir)

        if not input_path.exists():
            console.print(f"[red]Error: Directory {input_dir} does not exist[/red]")
            return

        json_files = list(input_path.glob("*.json"))
        if not json_files:
            console.print(f"[red]No JSON files found in {input_dir}[/red]")
            return

        # Check file status in one pass (single file read per file)
        unprocessed_files = []
        rejection_count = 0
        for f in json_files:
            has_rejection, is_processed = self._check_file_status(str(f))
            if has_rejection:
                rejection_count += 1
                if not is_processed:
                    unprocessed_files.append(f)

        console.print(f"Found {len(json_files)} total JSON files")
        console.print(f"With rejection: {rejection_count}")
        console.print(f"Already processed: {rejection_count - len(unprocessed_files)}")
        console.print(f"To process: {len(unprocessed_files)}")

        if not unprocessed_files:
            console.print("[green]All files already extracted![/green]")
            return

        console.print()

        successful = []
        failed = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=10,
        ) as progress:
            overall_task = progress.add_task(
                f"[bold blue]Extracting {len(unprocessed_files)} files",
                total=len(unprocessed_files)
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_file, filepath): filepath for filepath in unprocessed_files}

                for future in as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        successful.append(result["filepath"].name)
                    else:
                        failed.append({"file": result["filepath"].name, "error": result["error"]})
                    progress.advance(overall_task)

        console.print()
        console.print(f"[bold green]Successful:[/bold green] {len(successful)}")
        console.print(f"[bold red]Failed:[/bold red] {len(failed)}")

        if failed:
            console.print("\n[bold red]Failed files:[/bold red]")
            for item in failed:
                console.print(f"  {item['file']}: {item['error']}")


def main():
    parser = argparse.ArgumentParser(description="Extract from USPTO JSON files")
    parser.add_argument("--input-dir", "-i", required=True, help="Path to input directory containing JSON files")
    parser.add_argument("--max-workers", "-w", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--model", default="claude-opus-4-5", help="Anthropic model for extraction (default: claude-opus-4-5)")
    args = parser.parse_args()

    console = Console()
    anthropic_client = get_anthropic_client()

    console.print("[bold]Starting Extraction [/bold]\n")
    extraction_agent = Extractor(anthropic_client, model=args.model)
    extraction_agent.run_batch(args.input_dir, max_workers=args.max_workers)

    console.print("\n[bold green]Extraction complete![/bold green]")


if __name__ == "__main__":
    main()
