import json
import logging
from typing import List, Optional
from jinja2 import Template
from ..schemas import FamilyBatch, FamilyRouteDecision, GenerationRoute
from ..llm.base import LLMClient

logger = logging.getLogger(__name__)

class FamilyRouterNode:
    def __init__(self, llm_client: LLMClient, template_path: str):
        self.llm_client = llm_client
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def process_batch(self, batch: FamilyBatch, max_retries: int = 3) -> Optional[FamilyRouteDecision]:
        prompt = self.template.render(
            family_name=batch.family_name,
            family_id=batch.family_id,
            items=batch.items
        )
        
        logger.debug(f"Prompt for family {batch.family_id}:\n{prompt}")

        for attempt in range(max_retries):
            try:
                response = await self.llm_client.generate(prompt)
                logger.debug(f"Response for family {batch.family_id} (attempt {attempt+1}):\n{response}")
                
                # Extract JSON from response (handling potential markdown blocks)
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                data = json.loads(json_str)
                return FamilyRouteDecision(**data)
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for family {batch.family_id}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for family {batch.family_id}")
                    return self._get_fallback_decision(batch)
        return None

    def _get_fallback_decision(self, batch: FamilyBatch) -> FamilyRouteDecision:
        return FamilyRouteDecision(
            family_id=batch.family_id,
            primary_route=GenerationRoute.DIRECT_EXPANSION,
            safe_core_meaning="Fallback safe meaning",
            must_keep=[item.canonical_form for item in batch.items],
            must_avoid=["harmful content"],
            red_flags=["violence", "death"]
        )
