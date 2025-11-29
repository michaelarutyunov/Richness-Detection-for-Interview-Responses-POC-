"""
Base LLM client with async interface and retry logic.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.llm.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM API."""

    content: str
    function_call: dict | None = None
    tokens_used: int = 0
    latency_ms: int = 0
    model_used: str = ""


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: API key for the provider
            model: Model identifier
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    async def generate_completion(
        self,
        messages: list[dict[str, str]],
        function_schema: dict | None = None,
    ) -> LLMResponse:
        """
        Generate completion from LLM.

        Args:
            messages: Chat messages in OpenAI format
            function_schema: Optional function calling schema

        Returns:
            LLMResponse: Response with content and metadata

        Raises:
            LLMProviderError: Provider API error
        """
        pass

    async def generate_with_retry(
        self,
        messages: list[dict[str, str]],
        function_schema: dict | None = None,
        max_retries: int = 2,
    ) -> LLMResponse:
        """
        Generate completion with exponential backoff retry.

        Args:
            messages: Chat messages
            function_schema: Optional function calling schema
            max_retries: Maximum number of retries (default: 2)

        Returns:
            LLMResponse: Response from LLM

        Raises:
            LLMProviderError: All retry attempts failed
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.generate_completion(messages, function_schema)

                logger.info(
                    f"LLM call succeeded (attempt {attempt + 1}/{max_retries + 1}) "
                    f"in {response.latency_ms}ms, {response.tokens_used} tokens"
                )

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")

                # If this was the last attempt, raise the error
                if attempt == max_retries:
                    raise LLMProviderError(
                        f"LLM call failed after {max_retries + 1} attempts: {last_error}"
                    ) from last_error

                # Exponential backoff: 1s, 2s, 4s
                wait_time = (2**attempt) * 1.0
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        # Should never reach here, but just in case
        raise LLMProviderError(
            f"LLM call failed after {max_retries + 1} attempts: {last_error}"
        ) from last_error
