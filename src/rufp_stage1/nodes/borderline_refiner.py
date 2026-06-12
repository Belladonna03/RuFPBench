import json
import logging
from typing import List, Optional, Dict
from jinja2 import Template
from ..schemas import NaturalizedCandidate, RefinedCandidate, FamilyRouteDecision
from ..llm.base import LLMClient

logger = logging.getLogger(__name__)

class BorderlineRefinerNode:
    def __init__(self, llm_client: LLMClient, template_path: str):
        self.llm_client = llm_client
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def process_candidate(
        self, 
        candidate: NaturalizedCandidate, 
        route_decision: FamilyRouteDecision,
        max_retries: int = 3
    ) -> Optional[RefinedCandidate]:
        
        prompt = self.template.render(
            candidate_id=candidate.candidate_id,
            text=candidate.rewritten_text,
            safe_core_meaning=route_decision.safe_core_meaning,
            unsafe_neighbor=None, # Можно добавить в schemas позже если нужно
            red_flags=route_decision.red_flags
        )

        for attempt in range(max_retries):
            try:
                response = await self.llm_client.generate(prompt)
                
                # Extract JSON
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                data = json.loads(json_str)
                return RefinedCandidate(**data)

            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for candidate {candidate.candidate_id}: {e}")

        # Fallback to naturalized text if all attempts fail
        return RefinedCandidate(
            candidate_id=candidate.candidate_id,
            refined_text=candidate.rewritten_text,
            borderline_strategy="fallback_no_refinement",
            refinement_note="All refinement attempts failed"
        )
