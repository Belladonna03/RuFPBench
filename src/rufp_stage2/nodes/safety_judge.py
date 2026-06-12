import json
import logging
from typing import List, Optional
from jinja2 import Template
from ..schemas import SafetyJudgment, Stage2InputPrompt, SafetyLabel
from ..llm.base import LLMClient

logger = logging.getLogger(__name__)

class SafetyJudgeNode:
    def __init__(self, llm_client: LLMClient, template_path: str, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def judge_prompt(self, prompt: Stage2InputPrompt, max_retries: int = 3) -> Optional[SafetyJudgment]:
        prompt_text = self.template.render(text=prompt.text)
        
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.generate(prompt_text)
                
                # Extract JSON from potential markdown
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                data = json.loads(json_str)
                
                return SafetyJudgment(
                    prompt_id=prompt.prompt_id,
                    safety_label=SafetyLabel(data["safety_label"]),
                    confidence=data.get("confidence", 0.5),
                    rationale=data["rationale"],
                    failure_modes=data.get("failure_modes", []),
                    model_name=self.model_name
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for prompt {prompt.prompt_id}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to judge prompt {prompt.prompt_id} after {max_retries} attempts")
                    # Fallback to unclear
                    return SafetyJudgment(
                        prompt_id=prompt.prompt_id,
                        safety_label=SafetyLabel.UNCERTAIN,
                        confidence=0.0,
                        rationale=f"Error during judgment: {str(e)}",
                        failure_modes=["judgment_error"],
                        model_name=self.model_name
                    )
        return None
