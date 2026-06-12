import json
import logging
import time
from typing import List, Dict, Any
from jinja2 import Template
from ..schemas import ProbeResult, RefusalSignal
from src.rufp_stage1.llm.base import LLMClient
from src.rufp_stage1.schemas import AcceptedPrompt

logger = logging.getLogger(__name__)

class RefusalProbeNode:
    def __init__(self, target_clients: Dict[str, LLMClient], judge_client: LLMClient, judge_template_path: str):
        self.target_clients = target_clients
        self.judge_client = judge_client
        with open(judge_template_path, "r", encoding="utf-8") as f:
            self.judge_template = Template(f.read())

    async def probe_prompt(self, prompt: AcceptedPrompt) -> List[ProbeResult]:
        results = []
        for model_id, client in self.target_clients.items():
            start_time = time.time()
            try:
                # 1. Get response from target model
                response_text = await client.generate(prompt.text)
                latency = (time.time() - start_time) * 1000
                
                # 2. Judge if it's a refusal
                judge_prompt = self.judge_template.render(
                    prompt_text=prompt.text,
                    response_text=response_text
                )
                judge_response = await self.judge_client.generate(judge_prompt)
                
                json_str = judge_response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                
                judge_data = json.loads(json_str)
                
                results.append(ProbeResult(
                    prompt_id=prompt.prompt_id,
                    model_id=model_id,
                    response_text=response_text,
                    signal=RefusalSignal(judge_data["signal"]),
                    latency_ms=latency
                ))
            except Exception as e:
                logger.error(f"Error probing model {model_id} for prompt {prompt.prompt_id}: {e}")
                # We don't raise here to allow other models to be probed
        return results
