import json
import logging
from typing import List, Optional, Dict, Any, Set
from jinja2 import Template
from ..schemas import RepairPlan, RepairStrategy, RepairCandidate
from src.rufp_stage1.llm.base import LLMClient

logger = logging.getLogger(__name__)

class RepairPlannerNode:
    def __init__(self, llm_client: LLMClient, template_path: str):
        self.llm_client = llm_client
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def plan_repair(
        self,
        candidate: RepairCandidate,
        max_retries: int = 3,
        *,
        allowed_strategies: Optional[Set[str]] = None,
    ) -> Optional[RepairPlan]:
        prompt_text = self.template.render(
            text=candidate.original_prompt_text,
            category=candidate.category,
            repair_reason=candidate.repair_reason,
            stage2_labels=candidate.stage2_labels_snapshot
        )
        
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.generate(prompt_text)
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                data = json.loads(json_str)

                strat = RepairStrategy(data["repair_strategy"])
                if allowed_strategies is not None and strat.value not in allowed_strategies:
                    logger.warning(
                        "Planner returned disallowed strategy %s for %s (allowed=%s); retrying",
                        strat.value,
                        candidate.original_prompt_id,
                        sorted(allowed_strategies),
                    )
                    raise ValueError("repair_strategy_not_allowed")

                return RepairPlan(
                    original_prompt_id=candidate.original_prompt_id,
                    repair_strategy=strat,
                    repair_goals=data.get("repair_goals", []),
                    must_preserve=data.get("must_preserve", []),
                    must_avoid=data.get("must_avoid", []),
                    expected_risk=data.get("expected_risk", "unknown"),
                    reasoning=data.get("reasoning", ""),
                    instructions=data.get("instructions", ""),
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for candidate {candidate.original_prompt_id}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to plan repair for {candidate.original_prompt_id} after {max_retries} attempts")
        return None
