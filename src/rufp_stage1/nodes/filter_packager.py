import json
import logging
import uuid
import re
from typing import List, Dict, Any, Tuple
from jinja2 import Template
from ..schemas import RefinedCandidate, AcceptedPrompt, RejectedPrompt, Candidate
from ..llm.base import LLMClient

logger = logging.getLogger(__name__)

class FilterPackagerNode:
    def __init__(self, llm_client: LLMClient, template_path: str):
        self.llm_client = llm_client
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    def _normalize_text(self, text: str) -> str:
        # Убираем пунктуацию, лишние пробелы и приводим к нижнему регистру для дедупликации
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return " ".join(text.split())

    async def process_candidates(
        self, 
        refined_candidates: List[RefinedCandidate],
        raw_candidates: Dict[str, Candidate],
        max_retries: int = 2
    ) -> Tuple[List[AcceptedPrompt], List[RejectedPrompt]]:
        
        accepted = []
        rejected = []
        seen_normalized = set()

        for rc in refined_candidates:
            # 1. Near-duplicate check
            norm = self._normalize_text(rc.refined_text)
            if norm in seen_normalized:
                raw = raw_candidates.get(rc.candidate_id)
                rejected.append(RejectedPrompt(
                    candidate_id=rc.candidate_id,
                    family_id=raw.family_id if raw else "unknown",
                    text=rc.refined_text,
                    reason="duplicate_like"
                ))
                continue
            
            # 2. LLM Filter
            prompt = self.template.render(
                text=rc.refined_text,
                rationale=rc.metadata.get("rationale", ""),
                refinement_note=rc.refinement_note
            )

            decision_data = None
            for attempt in range(max_retries):
                try:
                    response = await self.llm_client.generate(prompt)
                    json_str = response.strip()
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    decision_data = json.loads(json_str)
                    break
                except Exception as e:
                    logger.warning(f"Filter attempt {attempt+1} failed for {rc.candidate_id}: {e}")

            if decision_data and decision_data.get("decision") == "accept_raw":
                raw = raw_candidates.get(rc.candidate_id)
                if raw:
                    accepted.append(AcceptedPrompt(
                        prompt_id=f"rufp-{str(uuid.uuid4())[:8]}",
                        candidate_id=rc.candidate_id,
                        family_id=raw.family_id,
                        category=raw.category,
                        subtype=raw.subtype,
                        length_bucket=raw.length_bucket,
                        route=raw.route,
                        text=rc.refined_text,
                        rationale=raw.rationale,
                        refinement_note=rc.refinement_note,
                        metadata=rc.metadata
                    ))
                    seen_normalized.add(norm)
                else:
                    logger.error(f"Missing raw candidate data for {rc.candidate_id}")
            else:
                raw = raw_candidates.get(rc.candidate_id)
                reason = decision_data.get("reason", "unknown_filter_error") if decision_data else "filter_failed"
                rejected.append(RejectedPrompt(
                    candidate_id=rc.candidate_id,
                    family_id=raw.family_id if raw else "unknown",
                    text=rc.refined_text,
                    reason=reason,
                    metadata={"comment": decision_data.get("comment")} if decision_data else {}
                ))

        return accepted, rejected

    def generate_summary(self, accepted: List[AcceptedPrompt], rejected: List[RejectedPrompt]) -> Dict[str, Any]:
        summary = {
            "total_processed": len(accepted) + len(rejected),
            "total_accepted": len(accepted),
            "total_rejected": len(rejected),
            "stats_by_category": {},
            "stats_by_route": {},
            "stats_by_length": {},
            "reject_reasons": {}
        }

        for a in accepted:
            summary["stats_by_category"][a.category] = summary["stats_by_category"].get(a.category, 0) + 1
            summary["stats_by_route"][a.route] = summary["stats_by_route"].get(a.route, 0) + 1
            summary["stats_by_length"][a.length_bucket] = summary["stats_by_length"].get(a.length_bucket, 0) + 1

        for r in rejected:
            summary["reject_reasons"][r.reason] = summary["reject_reasons"].get(r.reason, 0) + 1

        return summary
