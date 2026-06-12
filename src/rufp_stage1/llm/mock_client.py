from .base import LLMClient
from typing import Any
import asyncio

class MockLLMClient(LLMClient):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        await asyncio.sleep(0.05)
        return f"Mock response for: {prompt[:30]}..."
