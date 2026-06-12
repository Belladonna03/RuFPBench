"""
End-to-end Stage 2.5 orchestrator: selector → planner → repairer → revalidation → aggregator.

Supports resume (skip completed steps), dry-run (forces mock + structured dry-run logging),
per-node timing, manifest, and post-run lineage finalization + failure analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..policy.loader import load_stage25_policy
from ..policy.schema import LoggingVerbosity, Stage25Policy
from ..analysis.repair_failure_report import write_repair_failure_artifacts
from ..io import load_jsonl, save_json, save_jsonl
from ..llm.mock_client import MockLLMClient
from ..nodes.lineage_finalize import finalize_repair_lineage
from ..nodes.prompt_repairer import PromptRepairerNode
from ..nodes.repair_candidate_selector import RepairCandidateSelectorNode
from ..nodes.repair_decision_aggregator import run_aggregator
from ..nodes.repair_planner import RepairPlannerNode
from ..nodes.revalidation_runner import run_revalidation_batch
from ..schemas import (
    RepairCandidate,
    RepairPlan,
    RepairedPrompt,
    Stage25DecisionSummary,
    Stage25RunManifest,
)
from ..analysis.repair_audit_report import write_audit_artifacts

logger = logging.getLogger(__name__)


def _apply_logging_verbosity(verbosity: LoggingVerbosity) -> None:
    level = getattr(logging, verbosity.value.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logging.getLogger("rufp_stage25").setLevel(level)
    logging.getLogger("rufp_stage2").setLevel(level)


def _artifacts_root() -> str:
    return os.environ.get("RUFP_ARTIFACTS_ROOT", ".")


def _stage25_dir(run_id: str) -> str:
    return os.path.join(_artifacts_root(), "artifacts", "stage25", run_id)


def _stage2_dir(stage2_run_id: str) -> str:
    return os.path.join(_artifacts_root(), "artifacts", "stage2", stage2_run_id)


def _struct_event(event: str, **fields: Any) -> str:
    payload = {"event": event, **fields}
    return json.dumps(payload, ensure_ascii=False, default=str)


def _nonempty_jsonl(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _count_jsonl(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


async def _timed_node(
    name: str,
    coro_factory: Callable[[], Any],
    node_metrics: Dict[str, Dict[str, Any]],
    *,
    dry_run: bool,
) -> Any:
    logger.info(_struct_event("node_start", node=name, dry_run=dry_run))
    t0 = time.perf_counter()
    status = "ok"
    err: Optional[str] = None
    try:
        out = await coro_factory()
        return out
    except Exception as e:
        status = "error"
        err = f"{type(e).__name__}: {e}"
        logger.exception(_struct_event("node_failed", node=name, error=err))
        raise
    finally:
        ms = (time.perf_counter() - t0) * 1000.0
        node_metrics[name] = {
            "duration_ms": round(ms, 3),
            "status": status,
            "error": err,
        }
        logger.info(
            _struct_event("node_complete", node=name, duration_ms=round(ms, 3), status=status, error=err)
        )


def _planner_client(mock: bool) -> MockLLMClient:
    client: MockLLMClient = MockLLMClient()
    if mock:

        class MockPlannerClient(MockLLMClient):
            async def generate(self, prompt: str, **kwargs: Any) -> str:
                return json.dumps(
                    {
                        "repair_strategy": "strengthen_borderline_surface",
                        "repair_goals": ["Добавить слово 'убить'"],
                        "must_preserve": ["Безопасный смысл"],
                        "must_avoid": ["Реальное насилие"],
                        "expected_risk": "low",
                        "reasoning": "Mock reasoning",
                        "instructions": "Замените 'провести время' на 'убить время'",
                    }
                )

        client = MockPlannerClient()
    return client


def _repairer_client(mock: bool) -> MockLLMClient:
    client: MockLLMClient = MockLLMClient()
    if mock:

        class M(MockLLMClient):
            async def generate(self, prompt: str, **kwargs: Any) -> str:
                if "ИСХОДНЫЙ ТЕКСТ" in prompt or "text" in prompt.lower():
                    for line in prompt.splitlines():
                        if line.strip().startswith('"') and len(line) > 3:
                            inner = line.strip().strip('"')
                            return inner + " [repaired]"
                return "repaired output"

        client = M()
    return client


async def run_stage25_pipeline(
    input_run_id: str,
    run_id: str,
    mock: Optional[bool] = None,
    resume: bool = False,
    dry_run: bool = False,
    models: Optional[str] = None,
    run_post_analysis: bool = True,
    config_path: Optional[str] = None,
) -> Stage25RunManifest:
    """
    Run full Stage 2.5. Policy is loaded from ``config_path`` / ``RUFP_STAGE25_CONFIG`` / ``configs/stage25.yaml``.
    ``dry_run`` forces mock clients and tags structured logs with dry_run=true.
    ``mock`` overrides ``runtime.use_mock_clients`` when set (not None).
    """
    policy: Stage25Policy = load_stage25_policy(config_path)
    _apply_logging_verbosity(policy.runtime.logging_verbosity)

    use_mock = dry_run or (mock if mock is not None else policy.runtime.use_mock_clients)
    started = datetime.now(timezone.utc)
    out_dir = _stage25_dir(run_id)
    os.makedirs(out_dir, exist_ok=True)

    policy_dump_path = os.path.join(out_dir, "stage25_policy_resolved.json")
    with open(policy_dump_path, "w", encoding="utf-8") as pf:
        pf.write(policy.model_dump_json(indent=2))

    node_metrics: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {}

    model_list = [m.strip() for m in (models or ",".join(policy.revalidation.model_names)).split(",") if m.strip()]
    effective_resume = resume and policy.resume.skip_completed_nodes

    config_snapshot: Dict[str, Any] = {
        "stage2_run_id": input_run_id,
        "stage25_run_id": run_id,
        "mock": use_mock,
        "dry_run": dry_run,
        "resume": resume,
        "effective_resume": effective_resume,
        "models": model_list,
        "stage2_path": _stage2_dir(input_run_id),
        "stage25_path": out_dir,
        "config_path": config_path or os.environ.get("RUFP_STAGE25_CONFIG") or "configs/stage25.yaml",
        "policy_version": policy.version,
        "policy_file": policy_dump_path,
    }

    manifest_path = os.path.join(out_dir, "stage25_run_manifest.json")

    paths = {
        "repair_candidates": os.path.join(out_dir, "repair_candidates.jsonl"),
        "repair_plans": os.path.join(out_dir, "repair_plans.jsonl"),
        "repaired_prompts": os.path.join(out_dir, "repaired_prompts.jsonl"),
        "repair_lineage": os.path.join(out_dir, "repair_lineage.jsonl"),
        "repair_revalidation": os.path.join(out_dir, "repair_revalidation_results.jsonl"),
        "stage25_summary": os.path.join(out_dir, "stage25_summary.json"),
    }

    # --- 1. Repair candidate selector ---
    async def step_selector() -> int:
        if effective_resume and _nonempty_jsonl(paths["repair_candidates"]):
            n = _count_jsonl(paths["repair_candidates"])
            logger.info(_struct_event("node_skipped_resume", node="repair_selector", rows=n))
            return n
        node = RepairCandidateSelectorNode(input_run_id)
        candidates = node.select_candidates(policy)
        save_jsonl(paths["repair_candidates"], candidates)
        return len(candidates)

    n_cand = await _timed_node("repair_selector", step_selector, node_metrics, dry_run=dry_run)
    counts["repair_candidates"] = n_cand
    if n_cand == 0:
        logger.warning(
            _struct_event(
                "pipeline_warning",
                message="zero repair candidates; downstream steps may produce empty outputs",
                stage2_run_id=input_run_id,
            )
        )
        if policy.resume.fail_on_empty_upstream:
            raise RuntimeError(
                "repair_candidates is empty and policy.resume.fail_on_empty_upstream=true — aborting."
            )

    allowed_strat = policy.repair_strategies.as_set()

    # --- 2. Repair planner ---
    async def step_planner() -> int:
        if effective_resume and _nonempty_jsonl(paths["repair_plans"]):
            n = _count_jsonl(paths["repair_plans"])
            logger.info(_struct_event("node_skipped_resume", node="repair_planner", rows=n))
            return n
        cands = load_jsonl(paths["repair_candidates"], RepairCandidate)
        client = _planner_client(use_mock)
        planner = RepairPlannerNode(client, "src/rufp_stage25/prompts/repair_planner.jinja2")
        plans: List[RepairPlan] = []
        for c in cands:
            plan = await planner.plan_repair(c, allowed_strategies=allowed_strat)
            if plan:
                plans.append(plan)
        save_jsonl(paths["repair_plans"], plans)
        return len(plans)

    n_plans = await _timed_node("repair_planner", step_planner, node_metrics, dry_run=dry_run)
    counts["repair_plans"] = n_plans

    # --- 3. Prompt repairer ---
    async def step_repairer() -> Tuple[int, int]:
        if effective_resume and _nonempty_jsonl(paths["repaired_prompts"]) and _nonempty_jsonl(paths["repair_lineage"]):
            np = _count_jsonl(paths["repaired_prompts"])
            nl = _count_jsonl(paths["repair_lineage"])
            logger.info(
                _struct_event("node_skipped_resume", node="prompt_repairer", repaired=np, lineage=nl)
            )
            return np, nl
        plans = load_jsonl(paths["repair_plans"], RepairPlan)
        cands = load_jsonl(paths["repair_candidates"], RepairCandidate)
        by_id = {c.original_prompt_id: c for c in cands}
        client = _repairer_client(use_mock)
        pnode = PromptRepairerNode(client, "src/rufp_stage25/prompts/prompt_repairer.jinja2")
        repaired_rows: List[RepairedPrompt] = []
        lineage_rows = []
        for plan in plans:
            c = by_id.get(plan.original_prompt_id)
            if not c:
                logger.error(
                    _struct_event(
                        "missing_candidate",
                        original_prompt_id=plan.original_prompt_id,
                        message="planner produced plan without matching candidate",
                    )
                )
                raise ValueError(f"No candidate for plan {plan.original_prompt_id}")
            rp, lin = await pnode.repair_one(
                c,
                plan,
                parent_stage2_run_id=input_run_id,
                parent_stage25_run_id=run_id,
            )
            repaired_rows.append(rp)
            lineage_rows.append(lin)
        save_jsonl(paths["repaired_prompts"], repaired_rows)
        save_jsonl(paths["repair_lineage"], lineage_rows)
        reports_dir = os.path.join(out_dir, "reports")
        write_audit_artifacts(run_id, paths["repair_lineage"], reports_dir)
        return len(repaired_rows), len(lineage_rows)

    n_rep, n_lin = await _timed_node("prompt_repairer", lambda: step_repairer(), node_metrics, dry_run=dry_run)
    counts["repaired_prompts"] = n_rep
    counts["repair_lineage_rows"] = n_lin

    # --- 4. Revalidation ---
    async def step_reval() -> int:
        if effective_resume and _nonempty_jsonl(paths["repair_revalidation"]):
            n = _count_jsonl(paths["repair_revalidation"])
            logger.info(_struct_event("node_skipped_resume", node="revalidation_runner", rows=n))
            return n
        repaired = load_jsonl(paths["repaired_prompts"], RepairedPrompt)
        results = await run_revalidation_batch(
            repaired,
            mock=use_mock,
            model_names=model_list,
            only_ids=None,
            run_probes=policy.revalidation.run_probes,
        )
        save_jsonl(paths["repair_revalidation"], results)
        return len(results)

    n_rev = await _timed_node("revalidation_runner", step_reval, node_metrics, dry_run=dry_run)
    counts["revalidation_rows"] = n_rev

    # --- 5. Aggregator ---
    base_stage25 = os.path.join(_artifacts_root(), "artifacts", "stage25")

    def step_agg() -> Stage25DecisionSummary:
        if effective_resume and os.path.isfile(paths["stage25_summary"]):
            logger.info(_struct_event("node_skipped_resume", node="repair_decision_aggregator"))
            with open(paths["stage25_summary"], "r", encoding="utf-8") as f:
                return Stage25DecisionSummary.model_validate_json(f.read())
        return run_aggregator(
            run_id,
            base_dir=base_stage25,
            aggregator_policy=policy.aggregator,
        )

    async def step_agg_async() -> Any:
        return await asyncio.to_thread(step_agg)

    _ = await _timed_node("repair_decision_aggregator", step_agg_async, node_metrics, dry_run=dry_run)

    # --- Lineage finalize (original → repaired → final decision + revalidation snapshot) ---
    async def step_finalize() -> None:
        await asyncio.to_thread(finalize_repair_lineage, run_id, base_stage25)

    await _timed_node("lineage_finalize", step_finalize, node_metrics, dry_run=dry_run)

    analysis_paths: Dict[str, str] = {}
    if run_post_analysis:

        async def step_analysis() -> Dict[str, str]:
            return await asyncio.to_thread(write_repair_failure_artifacts, run_id, out_dir)

        analysis_paths = await _timed_node("repair_failure_report", step_analysis, node_metrics, dry_run=dry_run)

    finished = datetime.now(timezone.utc)
    manifest = Stage25RunManifest(
        run_id=run_id,
        input_run_id=input_run_id,
        started_at=started,
        finished_at=finished,
        config_snapshot=config_snapshot,
        counts=counts,
        node_metrics=node_metrics,
        lineage_finalized=True,
        analysis_artifacts=analysis_paths,
    )
    save_json(manifest_path, manifest)
    logger.info(_struct_event("pipeline_complete", run_id=run_id, manifest=manifest_path))
    return manifest


def preview_lineage_chain(run_id: str, repaired_prompt_id: str) -> str:
    """Human-readable lineage for CLI / debugging."""
    from ..audit.show_lineage import find_lineage_by_repaired_id, format_lineage_human

    found = find_lineage_by_repaired_id(repaired_prompt_id, run_id=run_id)
    if not found:
        return f"(no lineage for {repaired_prompt_id})"
    _path, rec = found
    return format_lineage_human(rec)
