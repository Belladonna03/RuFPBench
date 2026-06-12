"""End-to-end Stage 3: pool → QC → dedup → balance → hard subset → splits → export."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..analysis.final_report import write_stage3_analysis_bundle
from ..config import effective_artifacts_root, stage3_run_dir
from ..io import file_nonempty, load_jsonl, save_json, save_jsonl
from ..nodes.balancer_slice_builder import run_balancer_slice_builder
from ..nodes.candidate_merger import merge_candidate_pool_with_summary
from ..nodes.hard_subset import run_hard_subset
from ..nodes.qc_dedup_cluster import run_qc_dedup_cluster
from ..nodes.split_builder_exporter import run_splits, write_stage3_split_outputs
from ..policy.loader import load_stage3_policy
from ..policy.schema import Stage3Policy
from ..schemas import (
    BalancedRecord,
    DedupClusterRecord,
    HardSubsetRecord,
    QCLabelRecord,
    SplitAssignmentRecord,
    Stage3CandidateRecord,
    Stage3RunManifest,
    Stage3Summary,
)

logger = logging.getLogger(__name__)


def _log_event(event: str, **fields: Any) -> str:
    return json.dumps({"event": event, **fields}, ensure_ascii=False, default=str)


def _apply_log_level(verbosity: str) -> None:
    level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logging.getLogger("rufp_stage3").setLevel(level)


async def _run_node(name: str, fn: Callable[[], Any], metrics: Dict[str, Dict[str, Any]]) -> Any:
    logger.info(_log_event("node_start", node=name))
    t0 = time.perf_counter()
    err: Optional[str] = None
    status = "ok"
    try:
        out = await asyncio.to_thread(fn)
        return out
    except Exception as e:
        status = "error"
        err = f"{type(e).__name__}: {e}"
        logger.exception(_log_event("node_failed", node=name, error=err))
        raise RuntimeError(f"Stage 3 node {name!r} failed: {err}") from e
    finally:
        ms = (time.perf_counter() - t0) * 1000.0
        entry = metrics.setdefault(name, {})
        entry.update({"duration_ms": round(ms, 3), "status": status, "error": err})


def _filter_pool_qc(pool: List[Stage3CandidateRecord], qc_pass_ids: set) -> List[Stage3CandidateRecord]:
    return [r for r in pool if r.item_id in qc_pass_ids]


def _balanced_file_is_candidates(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
        if not line.strip():
            return False
        d = json.loads(line)
        return "prompt_text" in d or "text" in d
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _pool_by_split(
    pool: List[Stage3CandidateRecord], splits: List[SplitAssignmentRecord], split_name: str
) -> List[Stage3CandidateRecord]:
    ids = {s.item_id for s in splits if s.split == split_name}
    return [r for r in pool if r.item_id in ids]


def _apply_stage_run_paths(
    policy: Stage3Policy,
    *,
    stage2_run_id: Optional[str],
    stage25_run_id: Optional[str],
) -> Stage3Policy:
    """Override ``artifacts/stage2/<id>`` and/or ``artifacts/stage25/<id>`` when run ids are set."""
    if not stage2_run_id and not stage25_run_id:
        return policy
    root = Path(effective_artifacts_root())
    p = policy.model_copy(deep=True)
    if stage2_run_id:
        p.inputs.stage2_dir = str(root / "artifacts" / "stage2" / stage2_run_id)
    if stage25_run_id:
        p.inputs.stage25_dir = str(root / "artifacts" / "stage25" / stage25_run_id)
    return p


async def run_stage3_pipeline(
    run_id: str,
    *,
    config_path: Optional[str] = None,
    resume: bool = False,
    dry_run: bool = False,
    stage2_run_id: Optional[str] = None,
    stage25_run_id: Optional[str] = None,
) -> Stage3Summary:
    try:
        policy: Stage3Policy = load_stage3_policy(config_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load Stage 3 policy: {e}") from e
    policy = _apply_stage_run_paths(policy, stage2_run_id=stage2_run_id, stage25_run_id=stage25_run_id)
    _apply_log_level(policy.runtime.logging_verbosity)

    started_at = datetime.now(timezone.utc)
    base = str(stage3_run_dir(run_id))
    Path(base).mkdir(parents=True, exist_ok=True)
    exports_dir = str(Path(base) / "stage3_exports")
    metrics_dir = str(Path(base) / "metrics")

    paths = {
        "pool": str(Path(base) / "stage3_candidate_pool.jsonl"),
        "candidate_merge_summary": str(Path(base) / "stage3_candidate_merge_summary.json"),
        "qc": str(Path(base) / "stage3_qc_labels.jsonl"),
        "dedup": str(Path(base) / "stage3_dedup_clusters.jsonl"),
        "balanced": str(Path(base) / "stage3_balanced_pool.jsonl"),
        "balance_metadata": str(Path(base) / "stage3_balance_metadata.jsonl"),
        "slice_index": str(Path(base) / "stage3_slice_index.json"),
        "hard": str(Path(base) / "stage3_hard_subset.jsonl"),
        "split_assignment": str(Path(base) / "split_assignment.jsonl"),
        "splits": str(Path(base) / "stage3_splits.jsonl"),
        "export_manifest": str(Path(base) / "export_manifest.json"),
        "dev": str(Path(base) / "stage3_dev_set.jsonl"),
        "test": str(Path(base) / "stage3_test_set.jsonl"),
        "holdout": str(Path(base) / "stage3_review_holdout.jsonl"),
        "summary": str(Path(base) / "stage3_summary.json"),
        "metrics": metrics_dir,
        "manifest": str(Path(base) / "stage3_run_manifest.json"),
        "report": str(Path(base) / "stage3_final_report.md"),
        "policy_resolved": str(Path(base) / "stage3_policy_resolved.json"),
    }

    with open(paths["policy_resolved"], "w", encoding="utf-8") as f:
        f.write(policy.model_dump_json(indent=2))

    metrics: Dict[str, Dict[str, Any]] = {}
    effective_resume = resume and policy.resume.skip_completed_nodes

    def step_pool() -> List[Stage3CandidateRecord]:
        if effective_resume and file_nonempty(paths["pool"]):
            logger.info(_log_event("resume_skip", node="build_pool"))
            return load_jsonl(paths["pool"], Stage3CandidateRecord)
        p, merge_summary = merge_candidate_pool_with_summary(policy, dry_run=dry_run)
        save_jsonl(paths["pool"], p)
        save_json(paths["candidate_merge_summary"], merge_summary)
        return p

    pool = await _run_node("build_pool", step_pool, metrics)
    metrics["build_pool"]["outputs"] = {"candidate_pool_rows": len(pool)}

    def step_qc_dedup() -> tuple[
        List[QCLabelRecord], List[DedupClusterRecord], List[Stage3CandidateRecord]
    ]:
        if effective_resume and file_nonempty(paths["qc"]) and file_nonempty(paths["dedup"]):
            logger.info(_log_event("resume_skip", node="qc_dedup_cluster"))
            qc = load_jsonl(paths["qc"], QCLabelRecord)
            clusters = load_jsonl(paths["dedup"], DedupClusterRecord)
            qp = {q.item_id for q in qc if q.qc_pass}
            kept = _filter_pool_qc(pool, qp)
            return qc, clusters, kept
        qc, clusters, kept = run_qc_dedup_cluster(pool, policy)
        save_jsonl(paths["qc"], qc)
        save_jsonl(paths["dedup"], clusters)
        return qc, clusters, kept

    _qc_labels, dedup_clusters, pool_dedup = await _run_node("qc_dedup_cluster", step_qc_dedup, metrics)
    metrics["qc_dedup_cluster"]["outputs"] = {
        "qc_label_rows": len(_qc_labels),
        "dedup_cluster_records": len(dedup_clusters),
        "survivor_rows": len(pool_dedup),
    }

    def step_balance() -> tuple[List[Stage3CandidateRecord], List[BalancedRecord]]:
        if effective_resume and file_nonempty(paths["balanced"]):
            if _balanced_file_is_candidates(paths["balanced"]):
                rows = load_jsonl(paths["balanced"], Stage3CandidateRecord)
                meta_path = paths["balance_metadata"]
                if file_nonempty(meta_path):
                    meta = load_jsonl(meta_path, BalancedRecord)
                else:
                    meta = []
                return rows, meta
            meta = load_jsonl(paths["balanced"], BalancedRecord)
            ids = {m.item_id for m in meta}
            return [r for r in pool_dedup if r.item_id in ids], meta
        rows, meta, slice_index = run_balancer_slice_builder(pool_dedup, policy)
        save_jsonl(paths["balanced"], rows)
        save_jsonl(paths["balance_metadata"], meta)
        save_json(paths["slice_index"], slice_index)
        return rows, meta

    balanced_pool, _bal_meta = await _run_node("balance", step_balance, metrics)
    metrics["balance"]["outputs"] = {"balanced_pool_rows": len(balanced_pool)}

    def step_hard() -> List[HardSubsetRecord]:
        if effective_resume and file_nonempty(paths["hard"]):
            return load_jsonl(paths["hard"], HardSubsetRecord)
        h = run_hard_subset(balanced_pool, policy)
        save_jsonl(paths["hard"], h)
        return h

    hard_rows = await _run_node("hard_subset", step_hard, metrics)
    metrics["hard_subset"]["outputs"] = {"hard_subset_rows": len(hard_rows)}

    def step_split_builder_exporter() -> List[SplitAssignmentRecord]:
        if effective_resume and file_nonempty(paths["split_assignment"]):
            logger.info(_log_event("resume_skip", node="split_builder_exporter"))
            return load_jsonl(paths["split_assignment"], SplitAssignmentRecord)
        sp = run_splits(balanced_pool, policy, dedup_clusters)
        write_stage3_split_outputs(base, exports_dir, balanced_pool, sp, hard_rows, policy)
        return sp

    splits = await _run_node("split_builder_exporter", step_split_builder_exporter, metrics)
    metrics["split_builder_exporter"]["outputs"] = {
        "split_assignment_rows": len(splits),
        "dev": len(_pool_by_split(balanced_pool, splits, "dev")),
        "test": len(_pool_by_split(balanced_pool, splits, "test")),
        "review_holdout": len(_pool_by_split(balanced_pool, splits, "review_holdout")),
    }

    summary = Stage3Summary(
        run_id=run_id,
        pool_size=len(pool),
        qc_passed=len(pool_dedup),
        dedup_clusters=len(dedup_clusters),
        balanced_size=len(balanced_pool),
        hard_subset_size=len(hard_rows),
        dev_size=len(_pool_by_split(balanced_pool, splits, "dev")),
        test_size=len(_pool_by_split(balanced_pool, splits, "test")),
        review_holdout_size=len(_pool_by_split(balanced_pool, splits, "review_holdout")),
        exports_dir=exports_dir,
    )

    write_stage3_analysis_bundle(
        base,
        paths,
        summary,
        policy,
        pool,
        _qc_labels,
        len(dedup_clusters),
        balanced_pool,
        hard_rows,
        splits,
        metrics,
    )

    finished_at = datetime.now(timezone.utc)
    manifest = Stage3RunManifest(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        config_snapshot=policy.model_dump(mode="json"),
        input_paths={
            "stage2": policy.inputs.stage2_dir,
            "stage25": policy.inputs.stage25_dir or "",
            "stage2_run_id": stage2_run_id or "",
            "stage25_run_id": stage25_run_id or "",
        },
        output_paths=paths,
        node_metrics=metrics,
        dry_run=dry_run,
    )
    save_json(paths["manifest"], manifest)

    logger.info(
        _log_event(
            "pipeline_complete",
            run_id=run_id,
            duration_ms=round((finished_at - started_at).total_seconds() * 1000.0, 3),
        )
    )
    return summary
