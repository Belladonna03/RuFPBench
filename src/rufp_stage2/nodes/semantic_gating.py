import json
import logging
from typing import List, Dict, Any
from jinja2 import Template
from ..schemas import SemanticSafetyResult, NaturalnessResult, BorderlineResult, SafetyLabel, NaturalnessLabel, BorderlineLabel
from src.rufp_stage1.llm.base import LLMClient
from src.rufp_stage1.schemas import AcceptedPrompt

logger = logging.getLogger(__name__)

class SemanticGatingNode:
    def __init__(self, llm_client: LLMClient, template_path: str):
        self.llm_client = llm_client
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    async def process_prompt(self, prompt: AcceptedPrompt) -> Dict[str, Any]:
        prompt_text = self.template.render(text=prompt.text)
        
        try:
            response = await self.llm_client.generate(prompt_text)
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            safety = SemanticSafetyResult(
                prompt_id=prompt.prompt_id,
                label=SafetyLabel(data["safety"]["label"]),
                reasoning=data["safety"]["reasoning"]
            )
            
            naturalness = NaturalnessResult(
                prompt_id=prompt.prompt_id,
                label=NaturalnessLabel(data["naturalness"]["label"]),
                reasoning=data["naturalness"]["reasoning"]
            )
            
            borderline = BorderlineResult(
                prompt_id=prompt.prompt_id,
                label=BorderlineLabel(data["borderline"]["label"]),
                reasoning=data["borderline"]["reasoning"]
            )
            
            return {
                "safety": safety,
                "naturalness": naturalness,
                "borderline": borderline
            }
        except Exception as e:
            logger.error(f"Error gating prompt {prompt.prompt_id}: {e}")
            raise
