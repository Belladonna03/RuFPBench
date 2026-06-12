"""Human-readable audit output for a repair lineage record."""

from __future__ import annotations

import glob
import json
import os
from typing import Optional, Tuple

from ..schemas import RepairLineageRecord


def format_lineage_human(rec: RepairLineageRecord) -> str:
    lines = [
        "=" * 72,
        "RuFP Stage 2.5 — Repair lineage (audit)",
        "=" * 72,
        f"repair_id:              {rec.repair_id}",
        f"original_prompt_id:      {rec.original_prompt_id}",
        f"repaired_prompt_id:      {rec.repaired_prompt_id}",
        f"stage1_prompt_id:        {rec.stage1_prompt_id}",
        f"parent_stage2_run_id:    {rec.parent_stage2_run_id}",
        f"parent_stage25_run_id:   {rec.parent_stage25_run_id}",
        f"family_id / category:   {rec.family_id} / {rec.category}",
        f"repair_reason:           {rec.repair_reason}",
        f"repair_strategy:         {rec.repair_strategy.value}",
        f"repair_changed:          {rec.repair_changed}",
        f"original_hash (sha256): {rec.original_hash}",
        f"repaired_hash (sha256): {rec.repaired_hash}",
        f"final_repair_decision:   {rec.final_repair_decision}",
        "-" * 72,
        "Original text:",
        rec.original_prompt_text,
        "-" * 72,
        "Repaired text:",
        rec.repaired_text,
        "-" * 72,
        f"change_summary: {rec.change_summary}",
        "-" * 72,
        "Stage 2 labels snapshot:",
        _dump(rec.stage2_labels_snapshot),
        "Repair plan snapshot:",
        _dump(rec.repair_plan_snapshot),
        "Revalidation snapshot:",
        _dump(rec.revalidation_snapshot),
        "=" * 72,
    ]
    return "\n".join(lines)


def _dump(obj: object) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return repr(obj)


def find_lineage_by_repaired_id(
    repaired_prompt_id: str,
    *,
    stage25_root: str = "artifacts/stage25",
    run_id: Optional[str] = None,
) -> Optional[Tuple[str, RepairLineageRecord]]:
    """
    Search repair_lineage.jsonl under artifacts/stage25/<run_id>/ or all runs.
    Returns (file_path, record) or None.
    """
    if run_id:
        paths = [os.path.join(stage25_root, run_id, "repair_lineage.jsonl")]
    else:
        paths = sorted(glob.glob(os.path.join(stage25_root, "*", "repair_lineage.jsonl")))

    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("repaired_prompt_id") == repaired_prompt_id:
                    return p, RepairLineageRecord.model_validate(data)
    return None
