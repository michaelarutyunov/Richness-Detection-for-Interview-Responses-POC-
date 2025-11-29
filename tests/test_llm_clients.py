"""
Unit tests for LLM clients (mocked APIs).
"""

import json

import httpx
import pytest
import respx
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from src.llm.anthropic_client import AnthropicClient
from src.llm.base_client import LLMResponse
from src.llm.exceptions import LLMProviderError
from src.llm.kimi_client import KimiClient


@pytest.fixture
def mock_kimi_response():
    """Mock successful Kimi API response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "kimi-k2-turbo-preview",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "extract_graph_delta",
                                "arguments": json.dumps(
                                    {
                                        "nodes_added": [
                                            {
                                                "type": "attribute",
                                                "label": "affordable_price",
                                                "quote": "it's affordable",
                                            }
                                        ],
                                        "edges_added": [],
                                    }
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.fixture
def mock_kimi_error():
    """Mock Kimi API error response."""
    return httpx.Response(
        status_code=500,
        json={"error": {"message": "Internal server error", "type": "server_error"}},
    )


class TestKimiClient:
    """Tests for KimiClient."""

    @pytest.mark.asyncio
    async def test_generate_completion_success(self, mock_kimi_response):
        """Test successful Kimi API call with function calling."""
        client = KimiClient(
            api_key="test-key",
            model="kimi-k2-turbo-preview",
            temperature=0.3,
            max_tokens=1000,
            timeout=10,
        )

        with respx.mock:
            respx.post("https://api.moonshot.ai/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=mock_kimi_response)
            )

            messages = [
                {"role": "system", "content": "Extract nodes from responses."},
                {"role": "user", "content": "I like that it's affordable."},
            ]

            function_schema = {
                "name": "extract_graph_delta",
                "description": "Extract nodes and edges",
                "parameters": {"type": "object"},
            }

            response = await client.generate_completion(messages, function_schema)

            assert isinstance(response, LLMResponse)
            assert response.function_call is not None
            assert response.function_call["nodes_added"][0]["label"] == "affordable_price"
            assert response.tokens_used == 150
            assert response.model_used == "kimi-k2-turbo-preview"
            assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_generate_completion_without_function_calling(self, mock_kimi_response):
        """Test Kimi API call without function calling."""
        # Modify mock to return text content instead of function call
        text_response = mock_kimi_response.copy()
        text_response["choices"][0]["message"] = {
            "role": "assistant",
            "content": "This is a text response.",
            "tool_calls": None,
        }

        client = KimiClient(
            api_key="test-key",
            model="kimi-k2-turbo-preview",
            temperature=0.3,
            max_tokens=1000,
            timeout=10,
        )

        with respx.mock:
            respx.post("https://api.moonshot.ai/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=text_response)
            )

            messages = [{"role": "user", "content": "Hello"}]
            response = await client.generate_completion(messages)

            assert response.content == "This is a text response."
            assert response.function_call is None

    @pytest.mark.asyncio
    async def test_generate_with_retry_success_after_failure(self):
        """Test retry logic with transient failure."""
        client = KimiClient(
            api_key="test-key",
            model="kimi-k2-turbo-preview",
            temperature=0.3,
            max_tokens=1000,
            timeout=10,
        )

        call_count = 0

        def custom_response(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails
                return httpx.Response(500, json={"error": "temporary error"})
            else:
                # Second call succeeds
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-123",
                        "object": "chat.completion",
                        "created": 1677652288,
                        "model": "kimi-k2-turbo-preview",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "Success"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"total_tokens": 50},
                    },
                )

        with respx.mock:
            respx.post("https://api.moonshot.ai/v1/chat/completions").mock(
                side_effect=custom_response
            )

            messages = [{"role": "user", "content": "Test"}]
            response = await client.generate_with_retry(messages, max_retries=2)

            assert call_count == 2  # Failed once, succeeded on retry
            assert response.content == "Success"

    @pytest.mark.asyncio
    async def test_generate_with_retry_exhausted(self):
        """Test retry logic when all attempts fail."""
        client = KimiClient(
            api_key="test-key",
            model="kimi-k2-turbo-preview",
            temperature=0.3,
            max_tokens=1000,
            timeout=10,
        )

        with respx.mock:
            respx.post("https://api.moonshot.ai/v1/chat/completions").mock(
                return_value=httpx.Response(500, json={"error": "persistent error"})
            )

            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(LLMProviderError) as exc_info:
                await client.generate_with_retry(messages, max_retries=2)

            assert "failed after 3 attempts" in str(exc_info.value)


class TestAnthropicClient:
    """Tests for AnthropicClient."""

    @pytest.mark.asyncio
    async def test_generate_completion_with_tool_use(self, monkeypatch):
        """Test Anthropic API call with tool use."""

        # Mock the AsyncAnthropic client
        class MockAnthropicClient:
            class Messages:
                async def create(self, **kwargs):
                    return Message(
                        id="msg_123",
                        type="message",
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="tool_123",
                                type="tool_use",
                                name="extract_graph_delta",
                                input={
                                    "nodes_added": [
                                        {
                                            "type": "value",
                                            "label": "convenience",
                                            "quote": "it's convenient",
                                        }
                                    ],
                                    "edges_added": [],
                                },
                            )
                        ],
                        model="claude-sonnet-4-5",
                        stop_reason="tool_use",
                        usage=Usage(input_tokens=100, output_tokens=50),
                    )

            def __init__(self, **kwargs):
                self.messages = self.Messages()

        client = AnthropicClient(
            api_key="test-key",
            model="claude-sonnet-4-5",
            temperature=0.7,
            max_tokens=150,
            timeout=5,
        )

        # Replace the client's AsyncAnthropic instance
        client.client = MockAnthropicClient()

        messages = [
            {"role": "system", "content": "Extract nodes."},
            {"role": "user", "content": "It's convenient."},
        ]

        function_schema = {
            "name": "extract_graph_delta",
            "description": "Extract nodes",
            "parameters": {"type": "object"},
        }

        response = await client.generate_completion(messages, function_schema)

        assert response.function_call is not None
        assert response.function_call["nodes_added"][0]["label"] == "convenience"
        assert response.tokens_used == 150
        assert response.model_used == "claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_generate_completion_with_text(self, monkeypatch):
        """Test Anthropic API call with text response."""

        class MockAnthropicClient:
            class Messages:
                async def create(self, **kwargs):
                    return Message(
                        id="msg_123",
                        type="message",
                        role="assistant",
                        content=[TextBlock(type="text", text="This is a text response.")],
                        model="claude-sonnet-4-5",
                        stop_reason="end_turn",
                        usage=Usage(input_tokens=50, output_tokens=20),
                    )

            def __init__(self, **kwargs):
                self.messages = self.Messages()

        client = AnthropicClient(
            api_key="test-key",
            model="claude-sonnet-4-5",
            temperature=0.7,
            max_tokens=150,
            timeout=5,
        )

        client.client = MockAnthropicClient()

        messages = [{"role": "user", "content": "Hello"}]
        response = await client.generate_completion(messages)

        assert response.content == "This is a text response."
        assert response.function_call is None
        assert response.tokens_used == 70

    @pytest.mark.asyncio
    async def test_generate_with_retry_anthropic(self, monkeypatch):
        """Test retry logic for Anthropic client."""

        call_count = 0

        class MockAnthropicClient:
            class Messages:
                async def create(self, **kwargs):
                    nonlocal call_count
                    call_count += 1

                    if call_count == 1:
                        raise Exception("Temporary API error")

                    return Message(
                        id="msg_123",
                        type="message",
                        role="assistant",
                        content=[TextBlock(type="text", text="Success after retry")],
                        model="claude-sonnet-4-5",
                        stop_reason="end_turn",
                        usage=Usage(input_tokens=50, output_tokens=20),
                    )

            def __init__(self, **kwargs):
                self.messages = self.Messages()

        client = AnthropicClient(
            api_key="test-key",
            model="claude-sonnet-4-5",
            temperature=0.7,
            max_tokens=150,
            timeout=5,
        )

        client.client = MockAnthropicClient()

        messages = [{"role": "user", "content": "Test"}]
        response = await client.generate_with_retry(messages, max_retries=2)

        assert call_count == 2
        assert response.content == "Success after retry"
