"""Node 3: apply RepairPlan to produce RepairedPrompt + lineage fields."""

from __future__ import annotations

import logging
import uuid
from typing import Dict, Optional, Tuple

from jinja2 import Template

from ..schemas import (
    RepairCandidate,
    RepairPlan,
    RepairedPrompt,
    RepairLineageRecord,
    RepairStrategy,
)
from ..utils.hashing import text_sha256
from ..utils.diff_summary import simple_change_summary
from .lineage_builder import build_lineage_record, infer_decision_from_rewrite
from src.rufp_stage1.llm.base import LLMClient

logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return " ".join((s or "").split())


class PromptRepairerNode:
    def __init__(self, llm_client: LLMClient, template_path: str):
        self.llm_client = llm_client
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def repair_one(
        self,
        candidate: RepairCandidate,
        plan: RepairPlan,
        *,
        parent_stage2_run_id: str,
        parent_stage25_run_id: str,
        stage1_prompt_id: Optional[str] = None,
    ) -> Tuple[RepairedPrompt, RepairLineageRecord]:
        original_text = candidate.original_prompt_text
        repair_id = str(uuid.uuid4())
        repaired_prompt_id = f"rufp-repaired-{uuid.uuid4().hex[:12]}"

        prompt_text = self.template.render(
            text=original_text,
            strategy=plan.repair_strategy.value,
            instructions=plan.instructions,
            goals=plan.repair_goals,
            must_preserve=plan.must_preserve,
            must_avoid=plan.must_avoid,
        )

        repaired_raw = (await self.llm_client.generate(prompt_text)).strip()
        # If model returned quotes, strip outer quotes once
        if len(repaired_raw) >= 2 and repaired_raw[0] == repaired_raw[-1] == '"':
            repaired_raw = repaired_raw[1:-1].strip()

        orig_hash = text_sha256(original_text)
        rep_hash = text_sha256(repaired_raw)
        rewrite_changed = _norm(original_text) != _norm(repaired_raw)
        change_summary = simple_change_summary(original_text, repaired_raw)

        s1_id = stage1_prompt_id or candidate.metadata.get("stage1_prompt_id") or candidate.original_prompt_id

        repaired = RepairedPrompt(
            repair_id=repair_id,
            repaired_prompt_id=repaired_prompt_id,
            original_prompt_id=candidate.original_prompt_id,
            parent_stage2_run_id=parent_stage2_run_id,
            parent_stage25_run_id=parent_stage25_run_id,
            stage1_prompt_id=s1_id,
            family_id=candidate.family_id,
            category=candidate.category,
            subtype=candidate.subtype,
            generation_route=candidate.generation_route,
            repair_reason=candidate.repair_reason,
            repair_strategy=plan.repair_strategy,
            rewrite_changed=rewrite_changed,
            original_hash=orig_hash,
            repaired_hash=rep_hash,
            original_prompt_text=original_text,
            repaired_text=repaired_raw,
            change_summary=change_summary,
            metadata={
                **candidate.metadata,
                "source_bucket": candidate.source_bucket,
                "stage2_run_id": parent_stage2_run_id,
                "stage25_run_id": parent_stage25_run_id,
            },
        )

        plan_snapshot: Dict = {
            "repair_strategy": plan.repair_strategy.value,
            "repair_goals": plan.repair_goals,
            "must_preserve": plan.must_preserve,
            "must_avoid": plan.must_avoid,
            "expected_risk": plan.expected_risk,
            "reasoning": plan.reasoning,
            "instructions": plan.instructions,
            "planned_at": plan.planned_at.isoformat(),
        }

        revalidation_snapshot: Dict = {}  # filled after revalidation runner

        decision = infer_decision_from_rewrite(rewrite_changed)

        lineage = build_lineage_record(
            repaired,
            stage2_labels_snapshot=dict(candidate.stage2_labels_snapshot),
            repair_plan_snapshot=plan_snapshot,
            revalidation_snapshot=revalidation_snapshot,
            final_repair_decision=decision,
        )
        return repaired, lineage
