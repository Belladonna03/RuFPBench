"""
Node 4: re-run Stage 2 safety / naturalness / borderline judges + refusal probes on repaired prompts.

Uses `rufp_stage2` nodes as-is with `Stage2InputPrompt` built from `RepairedPrompt`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Sequence, Set

from rufp_stage2.llm.base import LLMClient
from rufp_stage2.llm.mock_client import MockProbeClient
from rufp_stage2.nodes.safety_judge import SafetyJudgeNode
from rufp_stage2.nodes.naturalness_judge import NaturalnessJudgeNode
from rufp_stage2.nodes.borderline_judge import BorderlineJudgeNode
from rufp_stage2.nodes.refusal_probe_runner import RefusalProbeRunnerNode
from rufp_stage2.schemas import Stage2InputPrompt

from ..schemas import RepairedPrompt, RepairRevalidationResult

logger = logging.getLogger(__name__)

STAGE2_PROMPTS_DIR = "src/rufp_stage2/prompts"


def repaired_to_stage2_input(rp: RepairedPrompt) -> Stage2InputPrompt:
    """Map repaired row to Stage 2 input; prompt_id = repaired id for traceability."""
    meta = dict(rp.metadata or {})
    meta.setdefault("subtype", rp.subtype)
    meta.setdefault("route", rp.generation_route)
    meta["original_prompt_id"] = rp.original_prompt_id
    meta["stage25_revalidation"] = True
    return Stage2InputPrompt(
        prompt_id=rp.repaired_prompt_id,
        family_id=rp.family_id,
        category=rp.category,
        text=rp.repaired_text,
        metadata=meta,
    )


def _build_mock_judge_clients() -> tuple[LLMClient, LLMClient, LLMClient]:
    """Return (safety, naturalness, borderline) clients producing valid Stage 2 JSON."""

    class MockSafety(LLMClient):
        async def generate(self, prompt: str, **kwargs) -> str:
            await asyncio.sleep(0.005)
            return json.dumps(
                {
                    "safety_label": "safe",
                    "confidence": 0.92,
                    "rationale": "mock revalidation safety",
                    "failure_modes": [],
                }
            )

    class MockNaturalness(LLMClient):
        async def generate(self, prompt: str, **kwargs) -> str:
            await asyncio.sleep(0.005)
            return json.dumps(
                {
                    "naturalness_label": "natural_ru",
                    "confidence": 0.9,
                    "rationale": "mock revalidation naturalness",
                    "issue_tags": [],
                }
            )

    class MockBorderline(LLMClient):
        async def generate(self, prompt: str, **kwargs) -> str:
            await asyncio.sleep(0.005)
            return json.dumps(
                {
                    "borderline_label": "good_borderline",
                    "confidence": 0.85,
                    "rationale": "mock revalidation borderline",
                    "issue_tags": [],
                }
            )

    return MockSafety(), MockNaturalness(), MockBorderline()


def build_stage2_nodes(
    *,
    mock: bool,
    safety_client: Optional[LLMClient] = None,
    naturalness_client: Optional[LLMClient] = None,
    borderline_client: Optional[LLMClient] = None,
    probe_clients: Optional[Dict[str, LLMClient]] = None,
    model_names: Optional[Sequence[str]] = None,
) -> tuple[SafetyJudgeNode, NaturalnessJudgeNode, BorderlineJudgeNode, RefusalProbeRunnerNode, List[str]]:
    """Wire Stage 2 judge/probe nodes; in mock mode use JSON-compatible mocks."""
    mnames = list(model_names or ["probe_a", "probe_b"])
    if mock:
        s, n, b = _build_mock_judge_clients()
        probes = {m: MockProbeClient(model_name=m) for m in mnames}
    else:
        if not all([safety_client, naturalness_client, borderline_client, probe_clients]):
            raise ValueError("Non-mock mode requires safety, naturalness, borderline, and probe_clients")
        s, n, b = safety_client, naturalness_client, borderline_client  # type: ignore
        probes = probe_clients or {m: MockProbeClient(model_name=m) for m in mnames}

    safety_node = SafetyJudgeNode(s, f"{STAGE2_PROMPTS_DIR}/safety_judge.jinja2", "revalidation-safety")
    nat_node = NaturalnessJudgeNode(n, f"{STAGE2_PROMPTS_DIR}/naturalness_judge.jinja2", "revalidation-naturalness")
    bord_node = BorderlineJudgeNode(b, f"{STAGE2_PROMPTS_DIR}/borderline_judge.jinja2", "revalidation-borderline")
    probe_node = RefusalProbeRunnerNode(probes)
    return safety_node, nat_node, bord_node, probe_node, mnames


async def revalidate_one(
    rp: RepairedPrompt,
    *,
    safety_node: SafetyJudgeNode,
    nat_node: NaturalnessJudgeNode,
    bord_node: BorderlineJudgeNode,
    probe_node: RefusalProbeRunnerNode,
    model_names: Sequence[str],
    run_probes: bool = True,
) -> RepairRevalidationResult:
    p = repaired_to_stage2_input(rp)
    breakdown: Dict[str, float] = {}

    t_all = time.perf_counter()

    t0 = time.perf_counter()
    safety = await safety_node.judge_prompt(p)
    breakdown["safety_ms"] = (time.perf_counter() - t0) * 1000
    if safety is None:
        raise RuntimeError(f"safety judgment missing for {rp.repaired_prompt_id}")

    t0 = time.perf_counter()
    naturalness = await nat_node.judge_prompt(p)
    breakdown["naturalness_ms"] = (time.perf_counter() - t0) * 1000
    if naturalness is None:
        raise RuntimeError(f"naturalness judgment missing for {rp.repaired_prompt_id}")

    t0 = time.perf_counter()
    borderline = await bord_node.judge_prompt(p)
    breakdown["borderline_ms"] = (time.perf_counter() - t0) * 1000
    if borderline is None:
        raise RuntimeError(f"borderline judgment missing for {rp.repaired_prompt_id}")

    t0 = time.perf_counter()
    probe_results = []
    if run_probes:
        for m in model_names:
            probe_results.append(await probe_node.probe_prompt(p, m))
    breakdown["probes_ms"] = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_all) * 1000

    logger.info(
        "revalidation_timing repaired_prompt_id=%s total_ms=%.2f breakdown=%s",
        rp.repaired_prompt_id,
        total_ms,
        {k: round(v, 2) for k, v in breakdown.items()},
    )

    return RepairRevalidationResult(
        repaired_prompt_id=rp.repaired_prompt_id,
        original_prompt_id=rp.original_prompt_id,
        safety_label=safety.safety_label,
        naturalness_label=naturalness.naturalness_label,
        borderline_label=borderline.borderline_label,
        safety=safety,
        naturalness=naturalness,
        borderline=borderline,
        probe_results=probe_results,
        timing_ms_total=total_ms,
        timing_ms_breakdown=breakdown,
    )


async def run_revalidation_batch(
    repaired: Sequence[RepairedPrompt],
    *,
    mock: bool = True,
    model_names: Optional[Sequence[str]] = None,
    only_ids: Optional[Set[str]] = None,
    run_probes: bool = True,
) -> List[RepairRevalidationResult]:
    """Revalidate each repaired prompt; optional subset filter by repaired_prompt_id."""
    subset = list(repaired)
    if only_ids:
        subset = [r for r in subset if r.repaired_prompt_id in only_ids]
        logger.info("subset mode: %d / %d prompts", len(subset), len(repaired))

    safety_node, nat_node, bord_node, probe_node, mnames = build_stage2_nodes(
        mock=mock, model_names=model_names
    )
    out: List[RepairRevalidationResult] = []
    for rp in subset:
        out.append(
            await revalidate_one(
                rp,
                safety_node=safety_node,
                nat_node=nat_node,
                bord_node=bord_node,
                probe_node=probe_node,
                model_names=mnames,
                run_probes=run_probes,
            )
        )
    return out
