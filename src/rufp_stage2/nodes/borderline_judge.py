import json
import logging
from typing import List, Optional
from jinja2 import Template
from ..schemas import BorderlineJudgment, Stage2InputPrompt, BorderlineLabel
from ..llm.base import LLMClient

logger = logging.getLogger(__name__)

class BorderlineJudgeNode:
    def __init__(self, llm_client: LLMClient, template_path: str, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def judge_prompt(self, prompt: Stage2InputPrompt, max_retries: int = 3) -> Optional[BorderlineJudgment]:
        prompt_text = self.template.render(
            text=prompt.text,
            category=prompt.category,
            subtype=prompt.metadata.get("subtype"),
            generation_route=prompt.metadata.get("route")
        )
        
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.generate(prompt_text)
                
                # Extract JSON
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                data = json.loads(json_str)
                
                # Map internal labels to BorderlineLabel enum
                label_map = {
                    "good_borderline": BorderlineLabel.STRONG,
                    "too_bland": BorderlineLabel.WEAK,
                    "too_obvious_safe": BorderlineLabel.NOT_BORDERLINE,
                    "off_category": BorderlineLabel.NOT_BORDERLINE
                }
                
                raw_label = data["borderline_label"]
                label = label_map.get(raw_label, BorderlineLabel.WEAK)
                
                return BorderlineJudgment(
                    prompt_id=prompt.prompt_id,
                    borderline_label=label,
                    confidence=data.get("confidence", 0.5),
                    rationale=data["rationale"],
                    issue_tags=data.get("issue_tags", []),
                    model_name=self.model_name
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for prompt {prompt.prompt_id}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to judge borderline for prompt {prompt.prompt_id} after {max_retries} attempts")
                    return BorderlineJudgment(
                        prompt_id=prompt.prompt_id,
                        borderline_label=BorderlineLabel.NOT_BORDERLINE,
                        confidence=0.0,
                        rationale=f"Error during judgment: {str(e)}",
                        issue_tags=["judgment_error"],
                        model_name=self.model_name
                    )
        return None
