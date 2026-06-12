import json
import logging
import uuid
from typing import List, Optional, Dict
from jinja2 import Template
from ..schemas import FamilyBatch, FamilyRouteDecision, Candidate, PromptLength, GenerationRoute
from ..llm.base import LLMClient

logger = logging.getLogger(__name__)

class CandidateGeneratorNode:
    def __init__(self, llm_client: LLMClient, template_path: str):
        self.llm_client = llm_client
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def process_batch(
        self, 
        batch: FamilyBatch, 
        route_decision: FamilyRouteDecision, 
        max_retries: int = 3
    ) -> List[Candidate]:
        prompt = self.template.render(
            family_name=batch.family_name,
            family_id=batch.family_id,
            category=batch.items[0].category if batch.items else "unknown",
            subtype=batch.items[0].subtype if batch.items else None,
            items=batch.items,
            route_decision=route_decision
        )

        candidates: List[Candidate] = []
        seen_texts = set()

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
                raw_candidates = data.get("candidates", [])

                for rc in raw_candidates:
                    text = rc["prompt_text"].strip()
                    if text in seen_texts:
                        continue
                    
                    seen_texts.add(text)
                    candidates.append(Candidate(
                        candidate_id=str(uuid.uuid4()),
                        family_id=batch.family_id,
                        category=batch.items[0].category if batch.items else "unknown",
                        subtype=batch.items[0].subtype if batch.items else None,
                        length_bucket=PromptLength(rc["length_bucket"]),
                        route=route_decision.primary_route,
                        prompt_text=text,
                        rationale=rc["rationale"]
                    ))

                if len(candidates) >= 6:
                    break
                else:
                    logger.warning(f"Attempt {attempt+1}: Only {len(candidates)} candidates generated. Retrying...")

            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for family {batch.family_id}: {e}")

        return candidates[:6] # Return at most 6
