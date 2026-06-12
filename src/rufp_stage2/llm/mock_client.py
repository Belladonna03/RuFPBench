from .base import LLMClient
from typing import Any
import asyncio
import json

class MockLLMClient(LLMClient):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        await asyncio.sleep(0.01)
        # Simple heuristic to return structured mock data if prompt looks like a judge request
        if "safety" in prompt.lower():
            return json.dumps({
                "label": "safe",
                "reasoning": "Mock judgment: text is safe."
            })
        return "Mock LLM response"

class MockProbeClient(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        await asyncio.sleep(0.05)
        # Simulate a refusal for specific keywords
        if any(word in prompt.lower() for word in ["убить", "смерть", "взрыв"]):
            return "Извините, я не могу выполнить этот запрос по соображениям безопасности."
        return f"Mock response from {self.model_name} for: {prompt[:20]}..."
