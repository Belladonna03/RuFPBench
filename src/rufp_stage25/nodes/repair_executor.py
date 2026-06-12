"""
Legacy executor (single LLM call). Prefer `PromptRepairerNode` for full lineage fields.
"""

import logging
import uuid
from jinja2 import Template

from ..schemas import RepairPlan, RepairedPrompt, RepairStrategy
from ..utils.hashing import text_sha256
from ..utils.diff_summary import simple_change_summary
from src.rufp_stage1.llm.base import LLMClient

logger = logging.getLogger(__name__)


class RepairExecutorNode:
    def __init__(self, llm_client: LLMClient, template_path: str):
        self.llm_client = llm_client
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def execute_repair(
        self,
        original_text: str,
        plan: RepairPlan,
        *,
        parent_stage2_run_id: str = "unknown",
        parent_stage25_run_id: str = "unknown",
    ) -> RepairedPrompt:
        prompt_text = self.template.render(
            text=original_text,
            strategy=plan.repair_strategy.value,
            instructions=plan.instructions,
        )

        try:
            repaired_text = (await self.llm_client.generate(prompt_text)).strip()
            repair_id = str(uuid.uuid4())
            rid = f"rufp-repaired-{uuid.uuid4().hex[:12]}"
            oh, rh = text_sha256(original_text), text_sha256(repaired_text)
            changed = oh != rh
            return RepairedPrompt(
                repair_id=repair_id,
                repaired_prompt_id=rid,
                original_prompt_id=plan.original_prompt_id,
                parent_stage2_run_id=parent_stage2_run_id,
                parent_stage25_run_id=parent_stage25_run_id,
                stage1_prompt_id=plan.original_prompt_id,
                family_id="unknown",
                category="unknown",
                subtype=None,
                generation_route="unknown",
                repair_reason="legacy_executor",
                repair_strategy=plan.repair_strategy,
                rewrite_changed=changed,
                original_hash=oh,
                repaired_hash=rh,
                original_prompt_text=original_text,
                repaired_text=repaired_text,
                change_summary=simple_change_summary(original_text, repaired_text),
                metadata={"legacy_repair_executor": True},
            )
        except Exception as e:
            logger.error(f"Error executing repair for {plan.original_prompt_id}: {e}")
            raise
