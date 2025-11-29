"""
Integration tests with real APIs (requires API keys in .env).

Run with: uv run pytest tests/test_integration_real_api.py -m integration
Skip with: uv run pytest tests/ (integration tests not run by default)
"""

import os

import pytest
from dotenv import load_dotenv

from src.core.interview_graph import InterviewGraph
from src.core.schema_manager import SchemaManager
from src.interview.prompt_builder import PromptBuilder
from src.interview.response_processor import ResponseProcessor
from src.interview.validator import Validator
from src.llm.anthropic_client import AnthropicClient
from src.llm.kimi_client import KimiClient

# Load environment variables from .env
load_dotenv()


@pytest.fixture
def schema_manager():
    """Create schema manager."""
    schema = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    schema.load_schema()
    schema.validate_schema()
    return schema


@pytest.fixture
def empty_graph(schema_manager):
    """Create empty graph."""
    return InterviewGraph(schema_manager)


@pytest.fixture
def prompt_builder():
    """Create prompt builder."""
    return PromptBuilder("prompts/extraction_prompts.yaml")


@pytest.fixture
def validator(schema_manager):
    """Create validator."""
    return Validator(schema_manager)


class TestKimiRealAPI:
    """Tests for Kimi client with real API."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kimi_generate_completion(self):
        """Test Kimi client with real API call."""
        api_key = os.getenv("KIMI_API_KEY")

        if not api_key:
            pytest.skip("KIMI_API_KEY not found in .env")

        client = KimiClient(
            api_key=api_key,
            model="kimi-k2-turbo-preview",
            temperature=0.3,
            max_tokens=1000,
            timeout=10,
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' in exactly those words."},
        ]

        response = await client.generate_completion(messages)

        assert response.content is not None
        assert len(response.content) > 0
        assert response.tokens_used > 0
        assert response.latency_ms > 0
        assert response.latency_ms < 10000  # Should be < 10s
        assert response.model_used == "kimi-k2-turbo-preview"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kimi_function_calling(self):
        """Test Kimi function calling with real API."""
        api_key = os.getenv("KIMI_API_KEY")

        if not api_key:
            pytest.skip("KIMI_API_KEY not found in .env")

        client = KimiClient(
            api_key=api_key,
            model="kimi-k2-turbo-preview",
            temperature=0.3,
            max_tokens=1000,
            timeout=10,
        )

        messages = [
            {
                "role": "system",
                "content": "Extract product attributes from consumer responses.",
            },
            {"role": "user", "content": "I love that it's affordable and convenient."},
        ]

        function_schema = {
            "name": "extract_attributes",
            "description": "Extract product attributes",
            "parameters": {
                "type": "object",
                "required": ["attributes"],
                "properties": {
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product attributes mentioned",
                    }
                },
            },
        }

        response = await client.generate_completion(messages, function_schema)

        assert response.function_call is not None
        assert "attributes" in response.function_call
        assert len(response.function_call["attributes"]) > 0
        assert response.latency_ms < 10000


class TestAnthropicRealAPI:
    """Tests for Anthropic client with real API."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_anthropic_generate_completion(self):
        """Test Anthropic client with real API call."""
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not found in .env")

        client = AnthropicClient(
            api_key=api_key,
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=150,
            timeout=10,
        )

        messages = [{"role": "user", "content": "Say 'Hello, World!' in exactly those words."}]

        response = await client.generate_completion(messages)

        assert response.content is not None
        assert len(response.content) > 0
        assert response.tokens_used > 0
        assert response.latency_ms > 0
        assert response.latency_ms < 10000
        assert "claude-sonnet" in response.model_used.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_anthropic_tool_use(self):
        """Test Anthropic tool use with real API."""
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not found in .env")

        client = AnthropicClient(
            api_key=api_key,
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=300,
            timeout=10,
        )

        messages = [
            {
                "role": "user",
                "content": "Generate a follow-up question about product affordability.",
            }
        ]

        function_schema = {
            "name": "generate_question",
            "description": "Generate a follow-up question",
            "parameters": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The follow-up question to ask",
                    }
                },
            },
        }

        response = await client.generate_completion(messages, function_schema)

        # Claude may return tool use or text
        assert response.function_call is not None or response.content is not None
        assert response.latency_ms < 10000


class TestEndToEndRealAPI:
    """End-to-end tests with real APIs."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_extraction_pipeline_kimi(
        self, schema_manager, empty_graph, prompt_builder, validator
    ):
        """Test full extraction pipeline with real Kimi API."""
        api_key = os.getenv("KIMI_API_KEY")

        if not api_key:
            pytest.skip("KIMI_API_KEY not found in .env")

        kimi_client = KimiClient(
            api_key=api_key,
            model="kimi-k2-turbo-preview",
            temperature=0.3,
            max_tokens=1000,
            timeout=10,
        )

        processor = ResponseProcessor(
            llm_client=kimi_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        participant_response = (
            "I love that it's affordable, so I can buy it every week without "
            "worrying about my budget. That gives me peace of mind."
        )

        conversation_history = [
            {"role": "assistant", "content": "What do you like about this product?"},
            {"role": "user", "content": participant_response},
        ]

        delta = await processor.process_response(
            participant_response=participant_response,
            conversation_history=conversation_history,
            existing_graph=empty_graph,
            turn_number=1,
        )

        # Verify extraction occurred
        assert delta.extraction_metadata["model_used"] == "kimi-k2-turbo-preview"
        assert delta.extraction_metadata["latency_ms"] > 0
        assert delta.extraction_metadata["latency_ms"] < 10000
        assert delta.extraction_metadata["tokens_used"] > 0

        # May or may not extract nodes depending on LLM response
        # Just verify structure is correct
        assert "validation_errors" in delta.extraction_metadata
        assert "validation_warnings" in delta.extraction_metadata
        assert "nodes_extracted" in delta.extraction_metadata
        assert "edges_extracted" in delta.extraction_metadata

        # If nodes were extracted, verify they're valid
        if delta.nodes_added:
            for node in delta.nodes_added:
                assert node.type in [
                    "attribute",
                    "functional_consequence",
                    "psychosocial_consequence",
                    "value",
                    "setting",
                ]
                assert len(node.label) >= 3
                assert len(node.label) <= 40

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_retry_logic_real_api(
        self, schema_manager, empty_graph, prompt_builder, validator
    ):
        """Test retry logic with real API (should succeed on first try)."""
        api_key = os.getenv("KIMI_API_KEY")

        if not api_key:
            pytest.skip("KIMI_API_KEY not found in .env")

        kimi_client = KimiClient(
            api_key=api_key,
            model="kimi-k2-turbo-preview",
            temperature=0.3,
            max_tokens=1000,
            timeout=10,
        )

        processor = ResponseProcessor(
            llm_client=kimi_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await processor.process_response(
            participant_response="It's convenient.",
            conversation_history=[],
            existing_graph=empty_graph,
            turn_number=1,
        )

        # Should succeed without retries
        assert "error" not in delta.extraction_metadata
        assert delta.extraction_metadata["latency_ms"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_latency(
        self, schema_manager, empty_graph, prompt_builder, validator
    ):
        """Test that extraction latency is acceptable (<5s)."""
        api_key = os.getenv("KIMI_API_KEY")

        if not api_key:
            pytest.skip("KIMI_API_KEY not found in .env")

        kimi_client = KimiClient(
            api_key=api_key,
            model="kimi-k2-turbo-preview",
            temperature=0.3,
            max_tokens=1000,
            timeout=10,
        )

        processor = ResponseProcessor(
            llm_client=kimi_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await processor.process_response(
            participant_response="I like it because it's affordable.",
            conversation_history=[],
            existing_graph=empty_graph,
            turn_number=1,
        )

        # Extraction should complete in < 5 seconds
        assert delta.extraction_metadata["latency_ms"] < 5000
