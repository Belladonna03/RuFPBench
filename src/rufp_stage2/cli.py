import typer
import asyncio
import logging
import os
import json
from .pipeline.orchestrator import run_stage2_pipeline
from .analysis.failure_report import generate_failure_report
from .nodes.safety_judge import SafetyJudgeNode
from .nodes.naturalness_judge import NaturalnessJudgeNode
from .nodes.borderline_judge import BorderlineJudgeNode
from .nodes.refusal_probe_runner import RefusalProbeRunnerNode
from .nodes.decision_aggregator import DecisionAggregatorNode
from .llm.mock_client import MockLLMClient, MockProbeClient
from .schemas import (
    Stage2InputPrompt, SafetyJudgment, NaturalnessJudgment, 
    BorderlineJudgment, ProbeResult
)
from .io import save_jsonl, save_json, load_jsonl

app = typer.Typer()

@app.command()
def run_stage2(
    input_path: str = typer.Option("artifacts/stage1/prompt_candidate_bank_raw.jsonl", help="Path to Stage 1 bank"),
    run_id: str = typer.Option(..., help="Run ID for output directory"),
    models: str = typer.Option("gpt-3.5,claude-3", help="Comma-separated model names"),
    mock: bool = typer.Option(True, help="Use mock clients"),
    resume: bool = typer.Option(False, help="Resume from existing artifacts"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run End-to-End Stage 2 Pipeline."""
    run_stage2_pipeline(
        input_path=input_path,
        run_id=run_id,
        mock=mock,
        resume=resume,
        debug=debug,
        models=models
    )
    generate_failure_report(run_id)
    typer.echo(f"Stage 2 E2E finished. Report: artifacts/stage2/{run_id}/reports/stage2_failure_report.md")

@app.command()
def run_safety_judge(
    input_path: str = typer.Option("artifacts/stage1/prompt_candidate_bank_raw.jsonl", help="Path to Stage 1 bank"),
    run_id: str = typer.Option(..., help="Run ID for output directory"),
    mock: bool = typer.Option(True, help="Use mock clients"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 1: Safety Judge."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    output_dir = f"artifacts/stage2/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    async def _run():
        typer.echo(f"Running Safety Judge on {input_path}...")
        
        # Load inputs
        prompts = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                # Stage 1 output has slightly different schema, we map it to Stage2InputPrompt
                data = json.loads(line)
                prompts.append(Stage2InputPrompt(
                    prompt_id=data.get("prompt_id", data.get("candidate_id")),
                    family_id=data["family_id"],
                    category=data["category"],
                    text=data.get("text", data.get("prompt_text")),
                    metadata=data.get("metadata", {})
                ))
        
        # Setup node
        client = MockLLMClient()
        if mock:
            class MockSafetyClient(MockLLMClient):
                async def generate(self, prompt: str, **kwargs) -> str:
                    return json.dumps({
                        "safety_label": "safe",
                        "confidence": 0.98,
                        "rationale": "Mock: Промпт использует метафору в безопасном контексте.",
                        "failure_modes": ["lexical_trigger_refusal"]
                    })
            client = MockSafetyClient()

        node = SafetyJudgeNode(client, "src/rufp_stage2/prompts/safety_judge.jinja2", model_name="mock-judge")
        
        judgments = []
        for p in prompts:
            res = await node.judge_prompt(p)
            if res:
                judgments.append(res)
        
        output_path = os.path.join(output_dir, "semantic_safety_labels.jsonl")
        save_jsonl(output_path, judgments)
        typer.echo(f"Processed {len(judgments)} prompts. Results saved to {output_path}")

    asyncio.run(_run())

@app.command()
def run_naturalness_judge(
    input_path: str = typer.Option("artifacts/stage1/prompt_candidate_bank_raw.jsonl", help="Path to Stage 1 bank"),
    run_id: str = typer.Option(..., help="Run ID for output directory"),
    mock: bool = typer.Option(True, help="Use mock clients"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 2: RU Naturalness Judge."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    output_dir = f"artifacts/stage2/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    async def _run():
        typer.echo(f"Running Naturalness Judge on {input_path}...")
        
        # Load inputs
        prompts = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                prompts.append(Stage2InputPrompt(
                    prompt_id=data.get("prompt_id", data.get("candidate_id")),
                    family_id=data["family_id"],
                    category=data["category"],
                    text=data.get("text", data.get("prompt_text")),
                    metadata=data.get("metadata", {})
                ))
        
        # Setup node
        client = MockLLMClient()
        if mock:
            class MockNaturalnessClient(MockLLMClient):
                async def generate(self, prompt: str, **kwargs) -> str:
                    return json.dumps({
                        "naturalness_label": "natural_ru",
                        "confidence": 0.95,
                        "rationale": "Mock: Текст звучит естественно, без признаков кальки.",
                        "issue_tags": []
                    })
            client = MockNaturalnessClient()

        node = NaturalnessJudgeNode(client, "src/rufp_stage2/prompts/naturalness_judge.jinja2", model_name="mock-naturalness-judge")
        
        judgments = []
        for p in prompts:
            res = await node.judge_prompt(p)
            if res:
                judgments.append(res)
        
        output_path = os.path.join(output_dir, "ru_naturalness_labels.jsonl")
        save_jsonl(output_path, judgments)
        typer.echo(f"Processed {len(judgments)} prompts. Results saved to {output_path}")

    asyncio.run(_run())

@app.command()
def run_borderline_judge(
    input_path: str = typer.Option("artifacts/stage1/prompt_candidate_bank_raw.jsonl", help="Path to Stage 1 bank"),
    run_id: str = typer.Option(..., help="Run ID for output directory"),
    mock: bool = typer.Option(True, help="Use mock clients"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 3: Borderline Judge."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    output_dir = f"artifacts/stage2/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    async def _run():
        typer.echo(f"Running Borderline Judge on {input_path}...")
        
        # Load inputs
        prompts = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                prompts.append(Stage2InputPrompt(
                    prompt_id=data.get("prompt_id", data.get("candidate_id")),
                    family_id=data["family_id"],
                    category=data["category"],
                    text=data.get("text", data.get("prompt_text")),
                    metadata=data.get("metadata", {})
                ))
        
        # Setup node
        client = MockLLMClient()
        if mock:
            class MockBorderlineClient(MockLLMClient):
                async def generate(self, prompt: str, **kwargs) -> str:
                    return json.dumps({
                        "borderline_label": "good_borderline",
                        "confidence": 0.9,
                        "rationale": "Mock: Промпт содержит сильный триггер в безопасном контексте.",
                        "issue_tags": []
                    })
            client = MockBorderlineClient()

        node = BorderlineJudgeNode(client, "src/rufp_stage2/prompts/borderline_judge.jinja2", model_name="mock-borderline-judge")
        
        judgments = []
        for p in prompts:
            res = await node.judge_prompt(p)
            if res:
                judgments.append(res)
        
        output_path = os.path.join(output_dir, "borderline_labels.jsonl")
        save_jsonl(output_path, judgments)
        typer.echo(f"Processed {len(judgments)} prompts. Results saved to {output_path}")

    asyncio.run(_run())

@app.command()
def run_refusal_probes(
    input_path: str = typer.Option("artifacts/stage1/prompt_candidate_bank_raw.jsonl", help="Path to Stage 1 bank"),
    run_id: str = typer.Option(..., help="Run ID for output directory"),
    models: str = typer.Option("gpt-3.5,claude-3", help="Comma-separated model names"),
    mock: bool = typer.Option(True, help="Use mock clients"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 4: Refusal Probe Runner."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    output_dir = f"artifacts/stage2/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    model_list = [m.strip() for m in models.split(",")]
    
    async def _run():
        typer.echo(f"Running Refusal Probes for models: {model_list}...")
        
        # Load inputs
        prompts = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                prompts.append(Stage2InputPrompt(
                    prompt_id=data.get("prompt_id", data.get("candidate_id")),
                    family_id=data["family_id"],
                    category=data["category"],
                    text=data.get("text", data.get("prompt_text")),
                    metadata=data.get("metadata", {})
                ))
        
        # Setup clients
        clients = {}
        for m in model_list:
            clients[m] = MockProbeClient(model_name=m) if mock else None # Real client here
            
        node = RefusalProbeRunnerNode(clients)
        
        # Run in batches of 5 for safety
        all_results = []
        batch_size = 5
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            results = await node.run_batch(batch, model_list)
            all_results.extend(results)
        
        output_path = os.path.join(output_dir, "refusal_probe_results.jsonl")
        save_jsonl(output_path, all_results)
        typer.echo(f"Probed {len(prompts)} prompts across {len(model_list)} models. Results saved to {output_path}")

    asyncio.run(_run())

@app.command()
def run_stage2_aggregator(
    run_id: str = typer.Option(..., help="Run ID for input/output directory"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Node 5: Decision Aggregator."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    
    output_dir = f"artifacts/stage2/{run_id}"
    
    async def _run():
        typer.echo(f"Running Decision Aggregator for run {run_id}...")
        
        # Load all previous nodes' outputs
        try:
            # 1. Base Prompts (re-loaded from stage 1 for full metadata)
            prompts = {}
            input_path = "artifacts/stage1/prompt_candidate_bank_raw.jsonl"
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    p = Stage2InputPrompt(
                        prompt_id=data.get("prompt_id", data.get("candidate_id")),
                        family_id=data["family_id"],
                        category=data["category"],
                        text=data.get("text", data.get("prompt_text")),
                        metadata=data.get("metadata", {})
                    )
                    prompts[p.prompt_id] = p

            # 2. Safety
            safety = {j.prompt_id: j for j in load_jsonl(os.path.join(output_dir, "semantic_safety_labels.jsonl"), SafetyJudgment)}
            
            # 3. Naturalness
            naturalness = {j.prompt_id: j for j in load_jsonl(os.path.join(output_dir, "ru_naturalness_labels.jsonl"), NaturalnessJudgment)}
            
            # 4. Borderline
            borderline = {j.prompt_id: j for j in load_jsonl(os.path.join(output_dir, "borderline_labels.jsonl"), BorderlineJudgment)}
            
            # 5. Probes
            probes = {}
            for p in load_jsonl(os.path.join(output_dir, "refusal_probe_results.jsonl"), ProbeResult):
                if p.prompt_id not in probes:
                    probes[p.prompt_id] = []
                probes[p.prompt_id].append(p)
        except Exception as e:
            typer.echo(f"Error loading input files: {e}")
            return

        node = DecisionAggregatorNode()
        accepted, positives, reviews, rejects = node.aggregate(prompts, safety, naturalness, borderline, probes)
        
        # Save results
        save_jsonl(os.path.join(output_dir, "validated_semantic_set.jsonl"), accepted)
        save_jsonl(os.path.join(output_dir, "probe_positive_set.jsonl"), positives)
        save_jsonl(os.path.join(output_dir, "review_queue.jsonl"), reviews)
        save_jsonl(os.path.join(output_dir, "reject_set.jsonl"), rejects)
        
        summary = node.generate_summary(accepted, positives, reviews, rejects)
        save_json(os.path.join(output_dir, "stage2_summary.json"), summary)
        
        typer.echo(f"Aggregation complete for {len(prompts)} prompts.")
        typer.echo(f"Accepted Semantic: {len(accepted)}")
        typer.echo(f"Probe Positive: {len(positives)}")
        typer.echo(f"Review Queue: {len(reviews)}")
        typer.echo(f"Reject Set: {len(rejects)}")

    asyncio.run(_run())

@app.command()
def run(
    input_path: str = typer.Option("artifacts/stage1/prompt_candidate_bank_raw.jsonl", help="Path to Stage 1 bank"),
    output_dir: str = typer.Option("artifacts/stage2", help="Output directory"),
    mock: bool = typer.Option(True, help="Use mock clients"),
    resume: bool = typer.Option(False, help="Resume from existing artifacts"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Run Stage 2: Gating & Validation."""
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    asyncio.run(run_stage2_pipeline(input_path, output_dir, mock, resume))

if __name__ == "__main__":
    app()
