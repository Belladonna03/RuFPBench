from .base import LLMClient
from typing import Any
import asyncio
import json

class MockLLMClient(LLMClient):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        await asyncio.sleep(0.01)
        
        # Mocking Repair Planner
        if "strategy" in prompt.lower() and "instructions" in prompt.lower():
            return json.dumps({
                "strategy": "strengthen_borderline",
                "reasoning": "Mock reasoning: prompt is too safe.",
                "instructions": "Add more aggressive verbs in a metaphorical context."
            })
            
        # Mocking Repair Executor
        if "исходный текст" in prompt.lower() or "original text" in prompt.lower():
            return "Repaired: " + prompt.split("\n")[1][:50] + "..."
            
        return "Mock Stage 2.5 response"
