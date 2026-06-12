from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import time
from typing import Callable, List, Optional

from baseten_performance_client import ClassificationResponse, PerformanceClient


@dataclass
class RerankResult:
    """Result of reranking a single document."""

    document: str
    score: float
    original_index: int
    tokens: Optional[int] = None  # Token count, populated if token_counter is available


class Reranker(ABC):
    """Abstract base class for reranking documents based on a query."""

    def __init__(
        self,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize the reranker.

        Args:
            token_counter: Optional callable that counts tokens in a string.
            max_tokens: Maximum total tokens for the output. Documents are returned
                in reranked order until this budget is exhausted.

        Raises:
            ValueError: If max_tokens is specified without a token_counter.
        """
        if max_tokens is not None and token_counter is None:
            raise ValueError("token_counter is required when max_tokens is specified")
        self.token_counter = token_counter
        self.max_tokens = max_tokens

    def _truncate_results(
        self, results: List[RerankResult], max_tokens: Optional[int] = None
    ) -> List[RerankResult]:
        """Truncate results to fit within max_tokens total.

        Also populates the tokens field for each result if token_counter is available.

        Args:
            results: List of RerankResult objects to truncate.
            max_tokens: Optional override for max_tokens. If not provided,
                uses the instance's max_tokens setting.
        """
        # If we have a token_counter, populate tokens for all results
        if self.token_counter is not None:
            for result in results:
                result.tokens = self.token_counter(result.document)

        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if self.token_counter is None or effective_max_tokens is None:
            return results

        truncated: List[RerankResult] = []
        total_tokens = 0
        for result in results:
            doc_tokens = result.tokens  # Already calculated above
            assert doc_tokens is not None
            if total_tokens + doc_tokens > effective_max_tokens:
                break
            truncated.append(result)
            total_tokens += doc_tokens

        return truncated

    @abstractmethod
    def _rerank(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents based on relevance to the query.

        Subclasses must implement this method to perform the actual reranking.

        Args:
            query: The search query to rank documents against.
            documents: List of document strings to rerank.
            instruction: Optional instruction for the reranker.

        Returns:
            List of RerankResult objects sorted by relevance (highest first).
        """
        pass

    def __call__(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query to rank documents against.
            documents: List of document strings to rerank.
            instruction: Optional instruction for the reranker.
            max_tokens: Optional override for max_tokens budget. If provided,
                overrides the instance's max_tokens for this call only.

        Returns:
            List of RerankResult objects sorted by relevance (highest first),
            truncated to fit within max_tokens if token_counter is provided.
        """
        start = time.perf_counter()
        results = self._rerank(query, documents, instruction)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return self._truncate_results(results, max_tokens=max_tokens)


class BasetenReranker(Reranker):
    """Reranker implementation using Baseten's classification API on top of Qwen 3 8B"""

    PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(
        self,
        client: Optional[PerformanceClient] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
        batch_size: int = 16,
        max_concurrent_requests: int = 256,
        timeout_s: int = 360,
    ):
        """
        Initialize the Baseten reranker.

        Args:
            client: Optional PerformanceClient. If not provided, uses config.
            token_counter: Optional callable that counts tokens in a string.
            max_tokens: Maximum total tokens for the output.
            batch_size: Batch size for classification requests.
            max_concurrent_requests: Maximum concurrent requests.
            timeout_s: Timeout in seconds.
        """
        super().__init__(token_counter=token_counter, max_tokens=max_tokens)
        if client is None:
            client = PerformanceClient(
                base_url=os.getenv("BASETEN_MODEL_URL"),
                api_key=os.getenv("BASETEN_API_KEY"),
            )
        self.client = client
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout_s = timeout_s

    def _format_input(
        self, instruction: Optional[str], query: str, document: str
    ) -> str:
        """Format input for the classification model."""
        if instruction is None:
            instruction = self.DEFAULT_INSTRUCTION
        return f"{self.PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}{self.SUFFIX}"

    def _rerank(
        self,
        query: str,
        documents: list[str],
        instruction: Optional[str] = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        # Format all documents for classification
        inputs = [self._format_input(instruction, query, doc) for doc in documents]

        # Classify all inputs
        response: ClassificationResponse = self.client.classify(
            inputs=inputs,
            truncate=True,
        )

        # Extract scores for "yes" labels
        results = []
        for idx, (doc, group) in enumerate(zip(documents, response.data)):
            score = 0.0
            for result in group:
                if result.label == "yes":
                    score = result.score
                    break
            results.append(RerankResult(document=doc, score=score, original_index=idx))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results
