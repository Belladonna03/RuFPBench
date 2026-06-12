import json
import logging
from typing import List, Optional, Dict
from jinja2 import Template
from ..schemas import Candidate, NaturalizedCandidate
from ..llm.base import LLMClient

logger = logging.getLogger(__name__)

class RUNaturalizerNode:
    def __init__(self, llm_client: LLMClient, template_path: str):
        self.llm_client = llm_client
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def process_batch(self, candidates: List[Candidate], max_retries: int = 3) -> List[NaturalizedCandidate]:
        if not candidates:
            return []

        prompt = self.template.render(candidates=candidates)
        
        results: List[NaturalizedCandidate] = []
        candidate_map = {c.candidate_id: c for c in candidates}

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
                raw_results = data.get("results", [])

                for res in raw_results:
                    cid = res["candidate_id"]
                    if cid not in candidate_map:
                        continue
                    
                    orig_text = candidate_map[cid].prompt_text
                    rewritten = res["rewritten_text"].strip()
                    
                    results.append(NaturalizedCandidate(
                        candidate_id=cid,
                        original_text=orig_text,
                        rewritten_text=rewritten,
                        change_type=res["change_type"],
                        rewrite_changed=(orig_text != rewritten)
                    ))
                
                if len(results) >= len(candidates):
                    break
                else:
                    logger.warning(f"Attempt {attempt+1}: Only {len(results)}/{len(candidates)} processed. Retrying...")
                    results = [] # Reset for retry

            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for naturalizer batch: {e}")

        # If failed after retries, return identity mapping
        if not results:
            for c in candidates:
                results.append(NaturalizedCandidate(
                    candidate_id=c.candidate_id,
                    original_text=c.prompt_text,
                    rewritten_text=c.prompt_text,
                    change_type="fallback_no_change",
                    rewrite_changed=False
                ))

        return results
