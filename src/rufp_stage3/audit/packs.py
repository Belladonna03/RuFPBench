"""Export small JSONL audit packs for human spot-check (optional; does not affect pipeline)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..config import ensure_stage3_run_dir
from ..io import file_nonempty, load_jsonl, save_json, save_jsonl
from ..policy.schema import AuditPackConfig, Stage3Policy
from ..schemas import AuditPackRecord, HardSubsetRecord, QCLabelRecord, SplitAssignmentRecord, Stage3CandidateRecord


def _repair_flag(r: Stage3CandidateRecord) -> bool:
    return "repaired_accept" in set(r.source_statuses or [])


def _source_stage_str(r: Stage3CandidateRecord) -> str:
    ss = list(r.source_stages or [])
    return "|".join(sorted(ss)) if ss else "unknown"


def _qc_block(q: Optional[QCLabelRecord]) -> Dict[str, Any]:
    if q is None:
        return {"note": "missing_qc_row_for_item"}
    return {
        "qc_label": q.qc_label,
        "qc_pass": q.qc_pass,
        "qc_flags": list(q.qc_flags or []),
        "cluster_id": q.cluster_id,
        "notes": q.notes,
    }


def _to_record(r: Stage3CandidateRecord, qc_by_id: Dict[str, QCLabelRecord]) -> AuditPackRecord:
    return AuditPackRecord(
        item_id=r.item_id,
        prompt_text=r.prompt_text,
        category=r.category,
        subtype=r.subtype,
        source_stage=_source_stage_str(r),
        source_stages=list(r.source_stages or []),
        probe_profile=dict(r.probe_profile or {}),
        repair_flag=_repair_flag(r),
        qc_metadata=_qc_block(qc_by_id.get(r.item_id)),
    )


def _split_item_ids(base: Path, split_name: str) -> Set[str]:
    p = base / "split_assignment.jsonl"
    if not p.is_file():
        alt = base / "stage3_splits.jsonl"
        p = alt if alt.is_file() else p
    if not file_nonempty(str(p)):
        return set()
    rows = load_jsonl(str(p), SplitAssignmentRecord)
    return {s.item_id for s in rows if s.split == split_name}


def export_stage3_audit_packs(
    run_id: str,
    policy: Stage3Policy,
    *,
    random_count: Optional[int] = None,
    hard_count: Optional[int] = None,
    repaired_count: Optional[int] = None,
    random_seed: Optional[int] = None,
    sample_random_from_split: Optional[str] = None,
    hard_order: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Writes under ``artifacts/stage3/<run_id>/stage3_exports/``:

    - ``audit_pack_random.jsonl``
    - ``audit_pack_hard.jsonl``
    - ``audit_pack_repaired.jsonl``
    - ``audit_packs_manifest.json``
    """
    base = ensure_stage3_run_dir(run_id)
    exp = base / "stage3_exports"
    exp.mkdir(parents=True, exist_ok=True)

    cfg: AuditPackConfig = policy.audit_packs
    seed = cfg.random_seed if random_seed is None else random_seed
    rng = random.Random(seed)

    rc = cfg.random_count if random_count is None else random_count
    hc = cfg.hard_count if hard_count is None else hard_count
    repc = cfg.repaired_count if repaired_count is None else repaired_count
    split_mode = cfg.sample_random_from_split if sample_random_from_split is None else sample_random_from_split
    h_order = cfg.hard_order if hard_order is None else hard_order

    bal_path = base / "stage3_balanced_pool.jsonl"
    if not file_nonempty(str(bal_path)):
        raise FileNotFoundError(f"Balanced pool required: {bal_path}")

    balanced = load_jsonl(str(bal_path), Stage3CandidateRecord)
    by_id = {r.item_id: r for r in balanced}

    qc_path = base / "stage3_qc_labels.jsonl"
    qc_by_id: Dict[str, QCLabelRecord] = {}
    if file_nonempty(str(qc_path)):
        for q in load_jsonl(str(qc_path), QCLabelRecord):
            qc_by_id[q.item_id] = q

    # --- random pack ---
    pool_rand = list(balanced)
    if split_mode != "all":
        allowed = _split_item_ids(base, split_mode)
        if allowed:
            pool_rand = [r for r in pool_rand if r.item_id in allowed]
    random_rows: List[Stage3CandidateRecord] = []
    if rc > 0 and pool_rand:
        k = min(rc, len(pool_rand))
        random_rows = rng.sample(sorted(pool_rand, key=lambda x: x.item_id), k=k)

    # --- hard pack ---
    hard_path = base / "stage3_hard_subset.jsonl"
    hard_rows: List[HardSubsetRecord] = (
        load_jsonl(str(hard_path), HardSubsetRecord) if file_nonempty(str(hard_path)) else []
    )
    hard_candidates: List[Stage3CandidateRecord] = []
    if hc > 0 and hard_rows:
        if h_order == "score_desc":
            hard_sorted = sorted(hard_rows, key=lambda h: -h.hard_score)
        else:
            hard_sorted = list(hard_rows)
            rng.shuffle(hard_sorted)
        seen: Set[str] = set()
        for h in hard_sorted:
            if len(hard_candidates) >= hc:
                break
            r = by_id.get(h.item_id)
            if r is None or h.item_id in seen:
                continue
            seen.add(h.item_id)
            hard_candidates.append(r)

    # --- repaired pack ---
    rep_pool = [r for r in balanced if _repair_flag(r)]
    repaired_rows: List[Stage3CandidateRecord] = []
    if repc > 0 and rep_pool:
        k = min(repc, len(rep_pool))
        repaired_rows = rng.sample(sorted(rep_pool, key=lambda x: x.item_id), k=k)

    out_random = [_to_record(r, qc_by_id) for r in random_rows]
    out_hard = [_to_record(r, qc_by_id) for r in hard_candidates]
    out_rep = [_to_record(r, qc_by_id) for r in repaired_rows]

    paths = {
        "audit_pack_random": str(exp / "audit_pack_random.jsonl"),
        "audit_pack_hard": str(exp / "audit_pack_hard.jsonl"),
        "audit_pack_repaired": str(exp / "audit_pack_repaired.jsonl"),
    }
    save_jsonl(paths["audit_pack_random"], out_random)
    save_jsonl(paths["audit_pack_hard"], out_hard)
    save_jsonl(paths["audit_pack_repaired"], out_rep)

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "random_seed": seed,
        "counts": {
            "audit_pack_random": len(out_random),
            "audit_pack_hard": len(out_hard),
            "audit_pack_repaired": len(out_rep),
        },
        "policy_audit_packs": policy.audit_packs.model_dump(),
        "paths": paths,
    }
    man_path = exp / "audit_packs_manifest.json"
    save_json(str(man_path), manifest)

    return {"manifest": str(man_path), **paths, "counts": manifest["counts"]}
