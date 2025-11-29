"""
Kimi (Moonshot AI) LLM client implementation.
"""

import json
import logging
import time

from openai import AsyncOpenAI

from src.llm.base_client import BaseLLMClient, LLMResponse
from src.llm.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class KimiClient(BaseLLMClient):
    """LLM client for Kimi (Moonshot AI) using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        base_url: str = "https://api.moonshot.ai/v1",
    ):
        """
        Initialize Kimi client.

        Args:
            api_key: Moonshot API key
            model: Model identifier (e.g., "kimi-k2-turbo-preview")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            base_url: Moonshot API base URL
        """
        super().__init__(api_key, model, temperature, max_tokens, timeout)
        self.base_url = base_url
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate_completion(
        self,
        messages: list[dict[str, str]],
        function_schema: dict | None = None,
    ) -> LLMResponse:
        """
        Generate completion from Kimi API.

        Args:
            messages: Chat messages in OpenAI format
            function_schema: Optional function calling schema

        Returns:
            LLMResponse: Response with content and metadata

        Raises:
            LLMProviderError: Kimi API error
        """
        start_time = time.time()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

        # Add function calling if schema provided
        if function_schema:
            kwargs["tools"] = [{"type": "function", "function": function_schema}]
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": function_schema["name"]},
            }

        try:
            response = await self.client.chat.completions.create(**kwargs)
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract function call if present
            function_call = None
            content = ""

            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                function_call = json.loads(tool_call.function.arguments)
                logger.debug(f"Kimi function call: {function_call}")
            elif response.choices[0].message.content:
                content = response.choices[0].message.content

            tokens_used = response.usage.total_tokens if response.usage else 0

            return LLMResponse(
                content=content,
                function_call=function_call,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model_used=self.model,
            )

        except Exception as e:
            raise LLMProviderError(f"Kimi API error: {e}") from e
