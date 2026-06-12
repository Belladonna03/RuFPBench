"""Build RepairLineageRecord from repaired prompt + snapshots."""

from __future__ import annotations

from typing import Any, Dict

from ..schemas import FinalRepairDecision, RepairedPrompt, RepairLineageRecord


def build_lineage_record(
    repaired: RepairedPrompt,
    *,
    stage2_labels_snapshot: Dict[str, Any],
    repair_plan_snapshot: Dict[str, Any],
    revalidation_snapshot: Dict[str, Any],
    final_repair_decision: str | FinalRepairDecision,
) -> RepairLineageRecord:
    decision = (
        final_repair_decision.value
        if isinstance(final_repair_decision, FinalRepairDecision)
        else str(final_repair_decision)
    )
    return RepairLineageRecord(
        repair_id=repaired.repair_id,
        original_prompt_id=repaired.original_prompt_id,
        repaired_prompt_id=repaired.repaired_prompt_id,
        stage1_prompt_id=repaired.stage1_prompt_id,
        parent_stage2_run_id=repaired.parent_stage2_run_id,
        parent_stage25_run_id=repaired.parent_stage25_run_id,
        repair_reason=repaired.repair_reason,
        repair_strategy=repaired.repair_strategy,
        repair_changed=repaired.rewrite_changed,
        original_hash=repaired.original_hash,
        repaired_hash=repaired.repaired_hash,
        stage2_labels_snapshot=dict(stage2_labels_snapshot),
        repair_plan_snapshot=dict(repair_plan_snapshot),
        revalidation_snapshot=dict(revalidation_snapshot),
        final_repair_decision=decision,
        family_id=repaired.family_id,
        category=repaired.category,
        subtype=repaired.subtype,
        generation_route=repaired.generation_route,
        original_prompt_text=repaired.original_prompt_text,
        repaired_text=repaired.repaired_text,
        change_summary=repaired.change_summary,
        metadata=dict(repaired.metadata),
    )


def infer_decision_from_rewrite(rewrite_changed: bool) -> FinalRepairDecision:
    if not rewrite_changed:
        return FinalRepairDecision.NO_IMPROVEMENT
    return FinalRepairDecision.PENDING_REVALIDATION
