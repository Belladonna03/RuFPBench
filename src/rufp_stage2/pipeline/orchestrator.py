import logging
import os
import uuid
import json
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

import typer

from ..io import save_json, save_jsonl, load_jsonl
from ..schemas import (
    Stage2InputPrompt, SafetyJudgment, NaturalnessJudgment, 
    BorderlineJudgment, ProbeResult, Stage2RunManifest
)
from ..nodes.safety_judge import SafetyJudgeNode
from ..nodes.naturalness_judge import NaturalnessJudgeNode
from ..nodes.borderline_judge import BorderlineJudgeNode
from ..nodes.refusal_probe_runner import RefusalProbeRunnerNode
from ..nodes.decision_aggregator import DecisionAggregatorNode
from ..llm.mock_client import MockLLMClient, MockProbeClient

logger = logging.getLogger(__name__)

async def _run_stage2_async(
    input_path: str,
    run_id: str,
    mock: bool,
    resume: bool,
    debug: bool,
    models: List[str]
):
    output_dir = f"artifacts/stage2/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = datetime.now()
    node_metrics = {}

    # Setup Mock Clients
    if mock:
        class MockGatingClient(MockLLMClient):
            async def generate(self, prompt: str, **kwargs) -> str:
                if "безопасность" in prompt.lower():
                    return json.dumps({"safety_label": "safe", "confidence": 0.9, "rationale": "ok", "failure_modes": []})
                if "естественность" in prompt.lower():
                    return json.dumps({"naturalness_label": "natural_ru", "confidence": 0.9, "rationale": "ok", "issue_tags": []})
                if "ценность" in prompt.lower():
                    return json.dumps({"borderline_label": "good_borderline", "confidence": 0.9, "rationale": "ok", "issue_tags": []})
                return "{}"
        
        gating_client = MockGatingClient()
        probe_clients = {m: MockProbeClient(model_name=m) for m in models}
    else:
        raise NotImplementedError("Real clients not configured")

    # Load inputs
    prompts_list = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompts_list.append(Stage2InputPrompt(
                prompt_id=data.get("prompt_id", data.get("candidate_id")),
                family_id=data["family_id"],
                category=data["category"],
                text=data.get("text", data.get("prompt_text")),
                metadata=data.get("metadata", {})
            ))
    prompts_dict = {p.prompt_id: p for p in prompts_list}

    # Helper for node execution
    async def run_node(name: str, func, output_file: str):
        path = os.path.join(output_dir, output_file)
        if resume and os.path.exists(path):
            logger.info(f"⏩ Skipping {name} (resume mode)")
            return
        
        logger.info(f"🚀 Running {name}...")
        t0 = time.time()
        results = await func()
        save_jsonl(path, results)
        node_metrics[name] = {"latency_sec": time.time() - t0, "count": len(results)}

    # 1. Safety Judge
    safety_node = SafetyJudgeNode(gating_client, "src/rufp_stage2/prompts/safety_judge.jinja2", "judge-safety")
    await run_node("safety_judge", lambda: asyncio.gather(*(safety_node.judge_prompt(p) for p in prompts_list)), "semantic_safety_labels.jsonl")

    # 2. Naturalness Judge
    nat_node = NaturalnessJudgeNode(gating_client, "src/rufp_stage2/prompts/naturalness_judge.jinja2", "judge-nat")
    await run_node("naturalness_judge", lambda: asyncio.gather(*(nat_node.judge_prompt(p) for p in prompts_list)), "ru_naturalness_labels.jsonl")

    # 3. Borderline Judge
    bord_node = BorderlineJudgeNode(gating_client, "src/rufp_stage2/prompts/borderline_judge.jinja2", "judge-bord")
    await run_node("borderline_judge", lambda: asyncio.gather(*(bord_node.judge_prompt(p) for p in prompts_list)), "borderline_labels.jsonl")

    # 4. Refusal Probe
    probe_node = RefusalProbeRunnerNode(probe_clients)
    await run_node("refusal_probe", lambda: probe_node.run_batch(prompts_list, models), "refusal_probe_results.jsonl")

    # 5. Aggregator
    logger.info("🚀 Running Aggregator...")
    safety_labels = {j.prompt_id: j for j in load_jsonl(os.path.join(output_dir, "semantic_safety_labels.jsonl"), SafetyJudgment)}
    nat_labels = {j.prompt_id: j for j in load_jsonl(os.path.join(output_dir, "ru_naturalness_labels.jsonl"), NaturalnessJudgment)}
    bord_labels = {j.prompt_id: j for j in load_jsonl(os.path.join(output_dir, "borderline_labels.jsonl"), BorderlineJudgment)}
    probes = {}
    for p in load_jsonl(os.path.join(output_dir, "refusal_probe_results.jsonl"), ProbeResult):
        if p.prompt_id not in probes: probes[p.prompt_id] = []
        probes[p.prompt_id].append(p)

    agg_node = DecisionAggregatorNode()
    accepted, positives, reviews, rejects = agg_node.aggregate(prompts_dict, safety_labels, nat_labels, bord_labels, probes)
    
    save_jsonl(os.path.join(output_dir, "validated_semantic_set.jsonl"), accepted)
    save_jsonl(os.path.join(output_dir, "probe_positive_set.jsonl"), positives)
    save_jsonl(os.path.join(output_dir, "review_queue.jsonl"), reviews)
    save_jsonl(os.path.join(output_dir, "reject_set.jsonl"), rejects)
    
    summary = agg_node.generate_summary(accepted, positives, reviews, rejects)
    save_json(os.path.join(output_dir, "stage2_summary.json"), summary)

    # Manifest
    manifest = Stage2RunManifest(
        run_id=run_id,
        started_at=start_time,
        finished_at=datetime.now(),
        input_path=input_path,
        config_snapshot={"mock": mock, "models": models},
        counts=summary["counts"]
    )
    save_json(os.path.join(output_dir, "stage2_run_manifest.json"), manifest)
    
    return output_dir

def run_stage2_pipeline(
    input_path: str,
    run_id: str,
    mock: bool = True,
    resume: bool = False,
    debug: bool = False,
    models: str = "gpt-3.5,claude-3"
):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    model_list = [m.strip() for m in models.split(",")]
    asyncio.run(_run_stage2_async(input_path, run_id, mock, resume, debug, model_list))
