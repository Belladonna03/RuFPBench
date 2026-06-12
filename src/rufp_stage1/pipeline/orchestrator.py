import logging
import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional

import typer

from ..io import save_json, load_raw_data, normalize_base_items, save_jsonl
from ..schemas import RunManifest, FamilyBatch, FamilyRouteDecision, Candidate, NaturalizedCandidate, RefinedCandidate
from ..nodes.ingestion import create_family_batches
from ..nodes.family_router import FamilyRouterNode
from ..nodes.candidate_generator import CandidateGeneratorNode
from ..nodes.ru_naturalizer import RUNaturalizerNode
from ..nodes.borderline_refiner import BorderlineRefinerNode
from ..nodes.filter_packager import FilterPackagerNode
from ..llm.mock_client import MockLLMClient

logger = logging.getLogger(__name__)

async def _run_pipeline_async(
    input_path: str,
    output_dir: str,
    mock: bool,
    resume: bool,
    debug: bool
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Mock Clients with specific behaviors
    if mock:
        class MockRouterClient(MockLLMClient):
            async def generate(self, prompt: str, **kwargs) -> str:
                import re
                match = re.search(r"ID: ([\w_]+)", prompt)
                fid = match.group(1) if match else "unknown"
                return json.dumps({
                    "family_id": fid,
                    "primary_route": "direct_expansion",
                    "safe_core_meaning": "Безопасное использование",
                    "must_keep": ["термин"],
                    "must_avoid": ["насилие"],
                    "red_flags": ["trigger"]
                })

        class MockCandidateClient(MockLLMClient):
            async def generate(self, prompt: str, **kwargs) -> str:
                import re
                match = re.search(r"ID: ([\w_]+)", prompt)
                fid = match.group(1) if match else "unknown"
                candidates = []
                for i, lb in enumerate(["short", "short", "medium", "medium", "long", "long"]):
                    candidates.append({
                        "length_bucket": lb,
                        "prompt_text": f"Промпт {i} ({lb}) для {fid}",
                        "rationale": "Тест"
                    })
                return json.dumps({"candidates": candidates})

        class MockNaturalizerClient(MockLLMClient):
            async def generate(self, prompt: str, **kwargs) -> str:
                import re
                ids = re.findall(r"ID: ([\w-]+)", prompt)
                results = [{"candidate_id": cid, "rewritten_text": f"Натуральный {cid}", "change_type": "стиль"} for cid in ids]
                return json.dumps({"results": results})

        class MockRefinerClient(MockLLMClient):
            async def generate(self, prompt: str, **kwargs) -> str:
                import re
                match = re.search(r"candidate_id\": \"([\w-]+)\"", prompt)
                cid = match.group(1) if match else "unknown"
                return json.dumps({
                    "candidate_id": cid,
                    "refined_text": f"Refined {cid}",
                    "borderline_strategy": "lexical",
                    "refinement_note": "усилено"
                })

        class MockFilterClient(MockLLMClient):
            async def generate(self, prompt: str, **kwargs) -> str:
                return json.dumps({"decision": "accept_raw", "reason": None, "comment": "ok"})

    # 1. Ingestion
    batches_path = os.path.join(output_dir, "family_batches.jsonl")
    if not (resume and os.path.exists(batches_path)):
        df = load_raw_data(input_path)
        records = normalize_base_items(df)
        batches = create_family_batches(records)
        save_jsonl(batches_path, batches)
        # Also save CSV for convenience
        flattened = []
        for b in batches:
            for item in b.items:
                d = item.model_dump()
                d["batch_id"] = b.batch_id
                flattened.append(d)
        import pandas as pd
        pd.DataFrame(flattened).to_csv(os.path.join(output_dir, "family_batches.csv"), index=False)

    # Load batches for subsequent steps
    batches = []
    with open(batches_path, "r", encoding="utf-8") as f:
        for line in f:
            batches.append(FamilyBatch(**json.loads(line)))

    # 2. Router
    routes_path = os.path.join(output_dir, "family_routes.jsonl")
    if not (resume and os.path.exists(routes_path)):
        node = FamilyRouterNode(MockRouterClient() if mock else None, "src/rufp_stage1/prompts/family_router.jinja2")
        decisions = []
        for b in batches:
            decisions.append(await node.process_batch(b))
        save_jsonl(routes_path, decisions)

    routes = {}
    with open(routes_path, "r", encoding="utf-8") as f:
        for line in f:
            r = FamilyRouteDecision(**json.loads(line))
            routes[r.family_id] = r

    # 3. Generator
    raw_candidates_path = os.path.join(output_dir, "raw_candidates.jsonl")
    if not (resume and os.path.exists(raw_candidates_path)):
        node = CandidateGeneratorNode(MockCandidateClient() if mock else None, "src/rufp_stage1/prompts/candidate_generator.jinja2")
        all_raw = []
        for b in batches:
            all_raw.extend(await node.process_batch(b, routes[b.family_id]))
        save_jsonl(raw_candidates_path, all_raw)

    raw_candidates = {c.candidate_id: c for c in []}
    with open(raw_candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            c = Candidate(**json.loads(line))
            raw_candidates[c.candidate_id] = c

    # 4. Naturalizer
    naturalized_path = os.path.join(output_dir, "naturalized_candidates.jsonl")
    if not (resume and os.path.exists(naturalized_path)):
        node = RUNaturalizerNode(MockNaturalizerClient() if mock else None, "src/rufp_stage1/prompts/ru_naturalizer.jinja2")
        all_nat = await node.process_batch(list(raw_candidates.values()))
        save_jsonl(naturalized_path, all_nat)

    naturalized = []
    with open(naturalized_path, "r", encoding="utf-8") as f:
        for line in f:
            naturalized.append(NaturalizedCandidate(**json.loads(line)))

    # 5. Refiner
    refined_path = os.path.join(output_dir, "refined_candidates.jsonl")
    if not (resume and os.path.exists(refined_path)):
        node = BorderlineRefinerNode(MockRefinerClient() if mock else None, "src/rufp_stage1/prompts/borderline_refiner.jinja2")
        all_ref = []
        for nc in naturalized:
            fid = raw_candidates[nc.candidate_id].family_id
            all_ref.append(await node.process_candidate(nc, routes[fid]))
        save_jsonl(refined_path, all_ref)

    refined = []
    with open(refined_path, "r", encoding="utf-8") as f:
        for line in f:
            refined.append(RefinedCandidate(**json.loads(line)))

    # 6. Filter & Packager
    final_bank_path = os.path.join(output_dir, "prompt_candidate_bank_raw.jsonl")
    if not (resume and os.path.exists(final_bank_path)):
        node = FilterPackagerNode(MockFilterClient() if mock else None, "src/rufp_stage1/prompts/cheap_filter.jinja2")
        accepted, rejected = await node.process_candidates(refined, raw_candidates)
        save_jsonl(final_bank_path, accepted)
        save_jsonl(os.path.join(output_dir, "stage1_rejects_log.jsonl"), rejected)
        
        summary = node.generate_summary(accepted, rejected)
        with open(os.path.join(output_dir, "stage1_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

def run_stage1_pipeline(
    input_path: str,
    output_dir: str = "artifacts/stage1",
    mock: bool = True,
    resume: bool = False,
    debug: bool = False
):
    """Orchestrates the full Stage 1 pipeline."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    start_time = datetime.now()
    typer.echo(f"🚀 Starting Stage 1 Pipeline at {start_time}")
    
    asyncio.run(_run_pipeline_async(input_path, output_dir, mock, resume, debug))

    end_time = datetime.now()
    typer.echo(f"✅ Stage 1 Pipeline completed in {end_time - start_time}")
