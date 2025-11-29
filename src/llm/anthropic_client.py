"""
Anthropic (Claude) LLM client implementation.
"""

import logging
import time

from anthropic import AsyncAnthropic

from src.llm.base_client import BaseLLMClient, LLMResponse
from src.llm.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """LLM client for Anthropic Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model identifier (e.g., "claude-sonnet-4-5")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, model, temperature, max_tokens, timeout)
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate_completion(
        self,
        messages: list[dict[str, str]],
        function_schema: dict | None = None,
    ) -> LLMResponse:
        """
        Generate completion from Anthropic API.

        Args:
            messages: Chat messages in OpenAI format
            function_schema: Optional function calling schema

        Returns:
            LLMResponse: Response with content and metadata

        Raises:
            LLMProviderError: Anthropic API error
        """
        start_time = time.time()

        # Convert OpenAI format to Anthropic format
        # Extract system message if present
        system_msg = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

        if system_msg:
            kwargs["system"] = system_msg

        # Add function calling if schema provided
        if function_schema:
            kwargs["tools"] = [
                {
                    "name": function_schema["name"],
                    "description": function_schema["description"],
                    "input_schema": function_schema["parameters"],
                }
            ]

        try:
            response = await self.client.messages.create(**kwargs)
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract content and function call
            function_call = None
            content = ""

            for block in response.content:
                if block.type == "tool_use":
                    function_call = block.input
                    logger.debug(f"Anthropic tool use: {function_call}")
                elif block.type == "text":
                    content = block.text

            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return LLMResponse(
                content=content,
                function_call=function_call,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model_used=self.model,
            )

        except Exception as e:
            raise LLMProviderError(f"Anthropic API error: {e}") from e
