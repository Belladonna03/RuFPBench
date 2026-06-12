import typer
import asyncio
import pandas as pd
from .io import load_raw_data, normalize_base_items, save_jsonl, save_json
from .llm.base import LLMClient
from .llm.mock_client import MockLLMClient
from .schemas import RunManifest, FamilyBatch, FamilyRouteDecision, Candidate, PromptLength, NaturalizedCandidate, RefinedCandidate, AcceptedPrompt, RejectedPrompt
from .nodes.ingestion import create_family_batches
from .nodes.family_router import FamilyRouterNode
from .nodes.candidate_generator import CandidateGeneratorNode
from .nodes.ru_naturalizer import RUNaturalizerNode
from .nodes.borderline_refiner import BorderlineRefinerNode
from .nodes.filter_packager import FilterPackagerNode
from .pipeline.orchestrator import run_stage1_pipeline
from datetime import datetime
import uuid
import os
import json
import logging

app = typer.Typer()

@app.command()
def run_stage1(
    input_path: str = typer.Option(..., "--input", help="Path to base_items CSV/XLSX"),
    output_dir: str = typer.Option("artifacts/stage1", help="Output directory"),
    mock: bool = typer.Option(True, help="Use mock LLM client"),
    resume: bool = typer.Option(False, help="Resume from existing artifacts"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run End-to-End Stage 1 Pipeline."""
    run_stage1_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        mock=mock,
        resume=resume,
        debug=debug
    )

@app.command()
def run_filter_packager(
    input_path: str = typer.Option("artifacts/stage1/refined_candidates.jsonl", help="Path to refined_candidates.jsonl"),
    raw_path: str = typer.Option("artifacts/stage1/raw_candidates.jsonl", help="Path to raw_candidates.jsonl"),
    output_dir: str = typer.Option("artifacts/stage1", help="Output directory"),
    mock: bool = typer.Option(True, help="Use mock LLM client"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 5: Filter + Packager."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    async def _run():
        typer.echo(f"Running Filter + Packager on {input_path}...")
        
        if not os.path.exists(input_path) or not os.path.exists(raw_path):
            typer.echo("Error: Input files not found.")
            return

        # Load refined candidates
        refined = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                refined.append(RefinedCandidate(**json.loads(line)))
        
        # Load raw candidates for metadata
        raw_candidates = {}
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                c = Candidate(**json.loads(line))
                raw_candidates[c.candidate_id] = c
        
        # Setup node
        client = MockLLMClient()
        if mock:
            class MockFilterClient(MockLLMClient):
                async def generate(self, prompt: str, **kwargs) -> str:
                    # Randomly accept or reject for testing
                    import random
                    decision = "accept_raw" if random.random() > 0.2 else "reject"
                    reason = None if decision == "accept_raw" else "too_bland"
                    return json.dumps({
                        "decision": decision,
                        "reason": reason,
                        "comment": "Mock filter comment"
                    })
            client = MockFilterClient()

        node = FilterPackagerNode(client, "src/rufp_stage1/prompts/cheap_filter.jinja2")
        
        accepted, rejected = await node.process_candidates(refined, raw_candidates)
        
        # Save results
        save_jsonl(os.path.join(output_dir, "prompt_candidate_bank_raw.jsonl"), accepted)
        save_jsonl(os.path.join(output_dir, "stage1_rejects_log.jsonl"), rejected)
        
        # Family to prompt map
        f_map = {}
        for a in accepted:
            if a.family_id not in f_map:
                f_map[a.family_id] = []
            f_map[a.family_id].append(a.prompt_id)
        
        with open(os.path.join(output_dir, "family_to_prompt_map.jsonl"), "w", encoding="utf-8") as f:
            for fid, pids in f_map.items():
                f.write(json.dumps({"family_id": fid, "prompt_ids": pids}, ensure_ascii=False) + "\n")
        
        # Summary
        summary = node.generate_summary(accepted, rejected)
        with open(os.path.join(output_dir, "stage1_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        typer.echo(f"Done! Accepted: {len(accepted)}, Rejected: {len(rejected)}")
        typer.echo(f"Summary saved to {os.path.join(output_dir, 'stage1_summary.json')}")

    asyncio.run(_run())

@app.command()
def run_borderline_refiner(
    input_path: str = typer.Option("artifacts/stage1/naturalized_candidates.jsonl", help="Path to naturalized_candidates.jsonl"),
    routes_path: str = typer.Option("artifacts/stage1/family_routes.jsonl", help="Path to family_routes.jsonl"),
    candidates_path: str = typer.Option("artifacts/stage1/raw_candidates.jsonl", help="Path to raw_candidates.jsonl"),
    output_path: str = typer.Option("artifacts/stage1/refined_candidates.jsonl", help="Output path"),
    mock: bool = typer.Option(True, help="Use mock LLM client"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 4: Borderline Refiner."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    async def _run():
        typer.echo(f"Running Borderline Refiner on {input_path}...")
        
        if not os.path.exists(input_path) or not os.path.exists(routes_path):
            typer.echo("Error: Input files not found.")
            return

        # Load naturalized candidates
        naturalized = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                naturalized.append(NaturalizedCandidate(**json.loads(line)))
        
        # Load routes
        routes = {}
        with open(routes_path, "r", encoding="utf-8") as f:
            for line in f:
                r = FamilyRouteDecision(**json.loads(line))
                routes[r.family_id] = r

        # Load raw candidates to get family_id for each candidate_id
        candidate_to_family = {}
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                c = Candidate(**json.loads(line))
                candidate_to_family[c.candidate_id] = c.family_id
        
        # Setup node
        client = MockLLMClient()
        if mock:
            class MockRefinerClient(MockLLMClient):
                async def generate(self, prompt: str, **kwargs) -> str:
                    import re
                    match = re.search(r"candidate_id\": \"([\w-]+)\"", prompt)
                    cid = match.group(1) if match else "unknown"
                    return json.dumps({
                        "candidate_id": cid,
                        "refined_text": f"Усиленный (refined) текст для {cid}",
                        "borderline_strategy": "lexical_trigger_strengthened",
                        "refinement_note": "Добавлено больше триггерных слов"
                    })
            client = MockRefinerClient()

        node = BorderlineRefinerNode(client, "src/rufp_stage1/prompts/borderline_refiner.jinja2")
        
        all_refined = []
        for nc in naturalized:
            fid = candidate_to_family.get(nc.candidate_id)
            route_decision = routes.get(fid) if fid else None
            
            if not route_decision:
                logger.warning(f"No route decision for candidate {nc.candidate_id}, skipping.")
                continue
                
            refined = await node.process_candidate(nc, route_decision)
            if refined:
                all_refined.append(refined)
        
        # Save results
        save_jsonl(output_path, all_refined)
        typer.echo(f"Refined {len(all_refined)} candidates. Results saved to {output_path}")

    asyncio.run(_run())

@app.command()
def run_ru_naturalizer(
    input_path: str = typer.Option("artifacts/stage1/raw_candidates.jsonl", help="Path to raw_candidates.jsonl"),
    output_path: str = typer.Option("artifacts/stage1/naturalized_candidates.jsonl", help="Output path"),
    batch_size: int = typer.Option(10, help="Batch size for processing"),
    mock: bool = typer.Option(True, help="Use mock LLM client"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 3: RU Naturalizer."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    async def _run():
        typer.echo(f"Running RU Naturalizer on {input_path}...")
        
        if not os.path.exists(input_path):
            typer.echo(f"Error: Input file {input_path} not found.")
            return

        # Load candidates
        candidates = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                candidates.append(Candidate(**json.loads(line)))
        
        # Setup node
        client = MockLLMClient()
        if mock:
            class MockNaturalizerClient(MockLLMClient):
                async def generate(self, prompt: str, **kwargs) -> str:
                    import re
                    # Extract IDs from prompt
                    ids = re.findall(r"ID: ([\w-]+)", prompt)
                    
                    results = []
                    for cid in ids:
                        results.append({
                            "candidate_id": cid,
                            "rewritten_text": f"Натурализованный текст для {cid}",
                            "change_type": "улучшен стиль"
                        })
                    return json.dumps({"results": results})
            client = MockNaturalizerClient()

        node = RUNaturalizerNode(client, "src/rufp_stage1/prompts/ru_naturalizer.jinja2")
        
        all_naturalized = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            naturalized = await node.process_batch(batch)
            all_naturalized.extend(naturalized)
        
        # Save results
        save_jsonl(output_path, all_naturalized)
        typer.echo(f"Naturalized {len(all_naturalized)} candidates. Results saved to {output_path}")

    asyncio.run(_run())

@app.command()
def run_candidate_generator(
    batches_path: str = typer.Option("artifacts/stage1/family_batches.jsonl", help="Path to family_batches.jsonl"),
    routes_path: str = typer.Option("artifacts/stage1/family_routes.jsonl", help="Path to family_routes.jsonl"),
    output_path: str = typer.Option("artifacts/stage1/raw_candidates.jsonl", help="Output path"),
    mock: bool = typer.Option(True, help="Use mock LLM client"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 2: Candidate Generator."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    async def _run():
        typer.echo(f"Running Candidate Generator...")
        
        # Load batches
        batches = {}
        with open(batches_path, "r", encoding="utf-8") as f:
            for line in f:
                b = FamilyBatch(**json.loads(line))
                batches[b.family_id] = b
        
        # Load routes
        routes = {}
        with open(routes_path, "r", encoding="utf-8") as f:
            for line in f:
                r = FamilyRouteDecision(**json.loads(line))
                routes[r.family_id] = r
        
        # Setup node
        client = MockLLMClient()
        if mock:
            class MockCandidateClient(MockLLMClient):
                async def generate(self, prompt: str, **kwargs) -> str:
                    import re
                    match = re.search(r"ID: ([\w_]+)", prompt)
                    fid = match.group(1) if match else "unknown"
                    
                    candidates = []
                    # Use unique texts to avoid deduplication in the node
                    for i, lb in enumerate(["short", "short", "medium", "medium", "long", "long"]):
                        candidates.append({
                            "length_bucket": lb,
                            "prompt_text": f"Пример промпта {i} ({lb}) для семейства {fid}",
                            "rationale": "Тестовое обоснование"
                        })
                    return json.dumps({"candidates": candidates})
            client = MockCandidateClient()

        node = CandidateGeneratorNode(client, "src/rufp_stage1/prompts/candidate_generator.jinja2")
        
        all_candidates = []
        for fid, batch in batches.items():
            route_decision = routes.get(fid)
            if not route_decision:
                logger.warning(f"No route decision for family {fid}, skipping.")
                continue
                
            candidates = await node.process_batch(batch, route_decision)
            all_candidates.extend(candidates)
        
        # Save results
        save_jsonl(output_path, all_candidates)
        typer.echo(f"Generated {len(all_candidates)} candidates. Results saved to {output_path}")

    asyncio.run(_run())

@app.command()
def run_family_router(
    input_path: str = typer.Option("artifacts/stage1/family_batches.jsonl", help="Path to family_batches.jsonl"),
    output_path: str = typer.Option("artifacts/stage1/family_routes.jsonl", help="Output path"),
    mock: bool = typer.Option(True, help="Use mock LLM client"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 1: Family Router."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    async def _run():
        typer.echo(f"Running Family Router on {input_path}...")
        
        # Load batches
        batches = []
        if not os.path.exists(input_path):
            typer.echo(f"Error: Input file {input_path} not found.")
            return

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                batches.append(FamilyBatch(**json.loads(line)))
        
        # Setup node
        client = MockLLMClient()
        if mock:
            class MockJSONClient(MockLLMClient):
                async def generate(self, prompt: str, **kwargs) -> str:
                    import re
                    match = re.search(r"ID: ([\w_]+)", prompt)
                    fid = match.group(1) if match else "unknown"
                    return json.dumps({
                        "family_id": fid,
                        "primary_route": "direct_expansion",
                        "safe_core_meaning": "Безопасное использование пограничного термина",
                        "must_keep": ["термин"],
                        "must_avoid": ["насилие"],
                        "red_flags": ["trigger_word"]
                    })
            client = MockJSONClient()

        node = FamilyRouterNode(client, "src/rufp_stage1/prompts/family_router.jinja2")
        
        decisions = []
        for batch in batches:
            decision = await node.process_batch(batch)
            if decision:
                decisions.append(decision)
        
        # Save results
        save_jsonl(output_path, decisions)
        typer.echo(f"Processed {len(decisions)} families. Results saved to {output_path}")

    asyncio.run(_run())

@app.command()
def build_batches(
    input_path: str = typer.Option(..., "--input", help="Path to base_items CSV/XLSX"),
    output_dir: str = typer.Option("artifacts/stage1", help="Output directory")
):
    """Ingest Stage 0 data and build family batches."""
    typer.echo(f"Ingesting data from {input_path}...")
    
    df = load_raw_data(input_path)
    records = normalize_base_items(df)
    batches = create_family_batches(records)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSONL
    jsonl_path = os.path.join(output_dir, "family_batches.jsonl")
    save_jsonl(jsonl_path, batches)
    
    # Save as CSV (flattened)
    csv_path = os.path.join(output_dir, "family_batches.csv")
    flattened = []
    for b in batches:
        for item in b.items:
            d = item.model_dump()
            d["batch_id"] = b.batch_id
            flattened.append(d)
    pd.DataFrame(flattened).to_csv(csv_path, index=False)
    
    typer.echo(f"Created {len(batches)} batches from {len(records)} items.")
    typer.echo(f"Saved to {jsonl_path} and {csv_path}")

@app.command()
def run(
    input_path: str = typer.Option(..., help="Path to base_items.json"),
    mock: bool = typer.Option(True, help="Use mock LLM client")
):
    """Run Stage 1 Pipeline."""
    async def _run():
        typer.echo(f"Starting Stage 1 with input: {input_path}")
        items = load_base_items(input_path)
        
        # Placeholder for pipeline execution
        typer.echo(f"Loaded {len(items)} items. Processing...")
        
        manifest = RunManifest(
            run_id=str(uuid.uuid4()),
            started_at=datetime.now(),
            finished_at=datetime.now(),
            total_items=len(items),
            total_candidates=len(items) * 3,
            total_accepted=len(items) * 3,
            total_rejected=0,
            config_snapshot={"mock": mock}
        )
        
        output_path = "artifacts/stage1/manifest.json"
        save_json(output_path, manifest)
        typer.echo(f"Pipeline finished. Manifest saved to {output_path}")

    asyncio.run(_run())

if __name__ == "__main__":
    app()
