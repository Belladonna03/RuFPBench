"""Build ``final_lineage.jsonl`` rows from Stage 3 run artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..schemas import (
    FinalLineageRecord,
    HardSubsetRecord,
    QCLabelRecord,
    SplitAssignmentRecord,
    Stage3CandidateRecord,
)


def _pick(d: Any, *keys: str) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    return {k: d.get(k) for k in keys if k in d}


def condense_stage2_block(rec: Stage3CandidateRecord) -> Dict[str, Any]:
    """Shrink merger metadata into stable keys for lineage JSON."""
    meta = rec.metadata or {}
    out: Dict[str, Any] = {
        "source_stages": list(rec.source_stages or []),
        "source_statuses": list(rec.source_statuses or []),
    }
    raw_val = meta.get("stage2_validated_labels")
    if isinstance(raw_val, dict):
        inp = raw_val.get("input") or {}
        safety = raw_val.get("safety") or {}
        naturalness = raw_val.get("naturalness") or {}
        borderline = raw_val.get("borderline") or {}
        out["validated_semantic"] = {
            "prompt_id": inp.get("prompt_id"),
            "category": inp.get("category"),
            "family_id": inp.get("family_id"),
            "subtype": (inp.get("metadata") or {}).get("subtype"),
            "safety": _pick(safety, "safety_label", "confidence", "failure_modes"),
            "naturalness": _pick(naturalness, "naturalness_label", "confidence", "issue_tags"),
            "borderline": _pick(borderline, "borderline_label", "confidence", "issue_tags"),
        }
    raw_pp = meta.get("probe_positive_row")
    if isinstance(raw_pp, dict):
        sem = raw_pp.get("semantic_data") or {}
        inp = sem.get("input") or {}
        out["probe_positive"] = {
            "prompt_id": inp.get("prompt_id"),
            "category": inp.get("category"),
            "refusal_count_stage2": raw_pp.get("refusal_count"),
            "embedded_probe_count": len(raw_pp.get("probes") or []),
        }
    return out


def condense_stage25_block(rec: Stage3CandidateRecord) -> Optional[Dict[str, Any]]:
    if "stage25" not in set(rec.source_stages or []) and not rec.repair_metadata:
        return None
    rm = rec.repair_metadata or {}
    promo = rm.get("promotion") if isinstance(rm, dict) else None
    rp = rm.get("repaired_prompt") if isinstance(rm, dict) else None
    lineage_rl = (rec.lineage or {}).get("repair_lineage")
    block: Dict[str, Any] = {
        "present": True,
        "original_prompt_id": rec.original_prompt_id,
        "current_prompt_id": rec.prompt_id,
        "stage1_prompt_id": rec.stage1_prompt_id,
    }
    if isinstance(promo, dict):
        block["promotion"] = _pick(
            promo,
            "original_prompt_id",
            "repaired_prompt_id",
            "family_id",
            "category",
            "status",
        )
    if isinstance(rp, dict):
        block["repaired_prompt"] = _pick(
            rp,
            "repaired_prompt_id",
            "original_prompt_id",
            "stage1_prompt_id",
            "generation_route",
            "subtype",
        )
    if isinstance(lineage_rl, dict):
        block["repair_lineage"] = _pick(
            lineage_rl,
            "repair_id",
            "original_prompt_id",
            "repaired_prompt_id",
            "stage1_prompt_id",
        )
    return block


def _qc_to_dict(q: QCLabelRecord) -> Dict[str, Any]:
    return q.model_dump(mode="json")


def _split_to_dict(s: SplitAssignmentRecord) -> Dict[str, Any]:
    return s.model_dump(mode="json")


def build_final_lineage_records(
    *,
    stage3_run_id: str,
    balanced_pool: List[Stage3CandidateRecord],
    qc_by_item: Dict[str, QCLabelRecord],
    split_by_item: Dict[str, SplitAssignmentRecord],
    hard_ids: Set[str],
) -> List[FinalLineageRecord]:
    rows: List[FinalLineageRecord] = []
    for rec in sorted(balanced_pool, key=lambda r: r.item_id):
        iid = rec.item_id
        q = qc_by_item.get(iid)
        sp = split_by_item.get(iid)
        qc_payload: Dict[str, Any]
        if q is None:
            qc_payload = {
                "qc_label": "missing",
                "qc_pass": None,
                "qc_flags": ["no_qc_row_for_item"],
                "notes": "No matching row in stage3_qc_labels.jsonl",
            }
        else:
            qc_payload = _qc_to_dict(q)

        if sp is None:
            split_payload = {
                "split": "missing",
                "error": "no_split_assignment_for_item",
            }
        else:
            split_payload = _split_to_dict(sp)

        s1 = {
            "stage1_prompt_id": rec.stage1_prompt_id,
            "original_prompt_id": rec.original_prompt_id,
            "prompt_id": rec.prompt_id,
            "family_id": rec.family_id,
        }

        rows.append(
            FinalLineageRecord(
                stage3_run_id=stage3_run_id,
                item_id=iid,
                stage1_prompt=s1,
                stage2=condense_stage2_block(rec),
                stage25_repair=condense_stage25_block(rec),
                stage3_qc=qc_payload,
                final_split_assignment=split_payload,
                in_hard_subset=iid in hard_ids,
                lineage_refs=list(rec.lineage_refs or []),
            )
        )
    return rows
