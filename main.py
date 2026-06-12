import typer
import asyncio
import json
import pandas as pd
from typing import Optional
from models import BaseItem, FamilyConfig, GenerationRoute, Stage1Result
from llm import MockLLMClient
from core import FamilyRouter, CandidateGenerator, Stage1Pipeline

app = typer.Typer()

@app.command()
def run(
    input_path: str = typer.Option(..., help="Path to base_items JSON/CSV"),
    output_path: str = typer.Option("stage1_results.json", help="Path to save results"),
    config_path: Optional[str] = typer.Option(None, help="Path to family_configs JSON"),
    mock: bool = typer.Option(True, help="Use mock LLM client"),
    resume: bool = typer.Option(False, help="Resume from existing output_path")
):
    async def _run():
        # Load existing results if resume
        existing_ids = set()
        all_candidates = []
        if resume and os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_data = json.load(f)
                result_obj = Stage1Result(**old_data)
                all_candidates = result_obj.candidates
                existing_ids = {c.base_item_id for c in all_candidates}
            typer.echo(f"Resuming: {len(existing_ids)} items already processed.")

        # Load items
        if input_path.endswith(".json"):
            with open(input_path, "r") as f:
                data = json.load(f)
                items = [BaseItem(**item) for item in data if item["id"] not in existing_ids]
        else:
            df = pd.read_csv(input_path)
            items = [BaseItem(**row) for _, row in df.iterrows() if str(row["id"]) not in existing_ids]

        if not items:
            typer.echo("No new items to process.")
            return

        # Setup router configs
        configs = []
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg_data = json.load(f)
                configs = [FamilyConfig(**c) for c in cfg_data]
        else:
            families = list(set(item.family for item in items))
            configs = [FamilyConfig(family=f, route=GenerationRoute.DIRECT_EXPANSION) for f in families]
        
        router = FamilyRouter(configs)
        llm_client = MockLLMClient() if mock else None
        generator = CandidateGenerator(llm_client)
        pipeline = Stage1Pipeline(router, generator)

        typer.echo(f"Running Stage 1 for {len(items)} new items...")
        
        # Incremental processing could be added here, but for now we run on remaining items
        new_result = await pipeline.run(items)
        
        final_result = Stage1Result(
            candidates=all_candidates + new_result.candidates,
            rejects=new_result.rejects
        )
        
        with open(output_path, "w") as f:
            f.write(final_result.model_dump_json(indent=2))
        
        typer.echo(f"Done! Total candidates: {len(final_result.candidates)}. Results saved to {output_path}")

    import os
    asyncio.run(_run())

if __name__ == "__main__":
    app()
