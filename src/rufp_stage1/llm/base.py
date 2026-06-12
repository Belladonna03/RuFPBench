import abc
from typing import Any, Dict

class LLMClient(abc.ABC):
    @abc.abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt."""
        pass
