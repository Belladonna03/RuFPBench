"""Merge final decisions + revalidation snapshots into repair_lineage.jsonl."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from ..io import load_jsonl, save_jsonl
from ..schemas import RepairLineageRecord, RepairRevalidationResult

logger = logging.getLogger(__name__)


def _load_jsonl_raw(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def finalize_repair_lineage(run_id: str, base_dir: str = "artifacts/stage25") -> None:
    """Set final_repair_decision + revalidation_snapshot on each lineage row."""
    root = os.path.join(base_dir, run_id)
    lineage_path = os.path.join(root, "repair_lineage.jsonl")
    if not os.path.isfile(lineage_path):
        logger.warning("No repair_lineage.jsonl at %s; skip finalize", lineage_path)
        return

    decisions: Dict[str, str] = {}
    for row in _load_jsonl_raw(os.path.join(root, "repair_promoted_set.jsonl")):
        decisions[row["repaired_prompt_id"]] = "promoted"
    for row in _load_jsonl_raw(os.path.join(root, "repair_failed_set.jsonl")):
        decisions[row["repaired_prompt_id"]] = "failed"
    for row in _load_jsonl_raw(os.path.join(root, "repair_review_queue.jsonl")):
        decisions[row["repaired_prompt_id"]] = "review"

    reval_by_id: Dict[str, Dict[str, Any]] = {}
    rev_path = os.path.join(root, "repair_revalidation_results.jsonl")
    if os.path.isfile(rev_path):
        for r in load_jsonl(rev_path, RepairRevalidationResult):
            reval_by_id[r.repaired_prompt_id] = r.model_dump(mode="json")

    updated: List[RepairLineageRecord] = []
    with open(lineage_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = RepairLineageRecord.model_validate_json(line)
            rid = rec.repaired_prompt_id
            if rid in decisions:
                rec = rec.model_copy(
                    update={
                        "final_repair_decision": decisions[rid],
                        "revalidation_snapshot": reval_by_id.get(rid, rec.revalidation_snapshot),
                    }
                )
            elif reval_by_id.get(rid):
                rec = rec.model_copy(
                    update={"revalidation_snapshot": reval_by_id[rid]}
                )
            updated.append(rec)

    save_jsonl(lineage_path, updated)
    logger.info("finalized repair_lineage.jsonl (%d rows)", len(updated))
