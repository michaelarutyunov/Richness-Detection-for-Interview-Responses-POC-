"""
Integration tests for ResponseProcessor (mocked LLM).
"""

from unittest.mock import AsyncMock

import pytest

from src.core.interview_graph import InterviewGraph
from src.core.schema_manager import SchemaManager
from src.interview.prompt_builder import PromptBuilder
from src.interview.response_processor import ResponseProcessor
from src.interview.validator import Validator
from src.llm.base_client import LLMResponse


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
def mock_llm_client():
    """Create mock LLM client."""
    mock_client = AsyncMock()
    mock_client.model = "mock-model"
    return mock_client


@pytest.fixture
def prompt_builder():
    """Create prompt builder."""
    return PromptBuilder("prompts/extraction_prompts.yaml")


@pytest.fixture
def validator(schema_manager):
    """Create validator."""
    return Validator(schema_manager)


class TestResponseProcessor:
    """Tests for ResponseProcessor integration."""

    @pytest.mark.asyncio
    async def test_process_response_success(
        self, mock_llm_client, prompt_builder, validator, empty_graph
    ):
        """Test successful response processing."""
        # Mock LLM response with valid extraction
        mock_llm_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={
                    "nodes_added": [
                        {
                            "type": "attribute",
                            "label": "affordable_price",
                            "quote": "it's affordable",
                        },
                        {
                            "type": "functional_consequence",
                            "label": "regular_purchase",
                            "quote": "buy it every week",
                        },
                    ],
                    "edges_added": [
                        {
                            "type": "leads_to",
                            "source": "affordable_price",
                            "target": "regular_purchase",
                            "quote": "affordable, so I buy it every week",
                        }
                    ],
                },
                tokens_used=150,
                latency_ms=1200,
                model_used="mock-model",
            )
        )

        processor = ResponseProcessor(
            llm_client=mock_llm_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        participant_response = "I like that it's affordable, so I can buy it every week."
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

        # Verify delta structure
        assert len(delta.nodes_added) == 2
        assert len(delta.edges_added) == 1
        assert delta.richness_score > 0
        assert delta.extraction_metadata["model_used"] == "mock-model"
        assert delta.extraction_metadata["tokens_used"] == 150
        assert delta.extraction_metadata["latency_ms"] == 1200
        assert len(delta.extraction_metadata["validation_errors"]) == 0

        # Verify node details
        node_labels = {n.label for n in delta.nodes_added}
        assert "affordable_price" in node_labels
        assert "regular_purchase" in node_labels

        # Verify edge details
        assert delta.edges_added[0].source == "affordable_price"
        assert delta.edges_added[0].target == "regular_purchase"

    @pytest.mark.asyncio
    async def test_process_response_llm_failure(
        self, mock_llm_client, prompt_builder, validator, empty_graph
    ):
        """Test handling of LLM API failure."""
        from src.llm.exceptions import LLMProviderError

        # Mock LLM failure
        mock_llm_client.generate_with_retry = AsyncMock(side_effect=LLMProviderError("API error"))

        processor = ResponseProcessor(
            llm_client=mock_llm_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await processor.process_response(
            participant_response="Test response",
            conversation_history=[],
            existing_graph=empty_graph,
            turn_number=1,
        )

        # Should return empty delta with error in metadata
        assert len(delta.nodes_added) == 0
        assert len(delta.edges_added) == 0
        assert "error" in delta.extraction_metadata
        assert "API error" in delta.extraction_metadata["error"]

    @pytest.mark.asyncio
    async def test_process_response_no_function_call(
        self, mock_llm_client, prompt_builder, validator, empty_graph
    ):
        """Test handling when LLM doesn't return function call."""
        # Mock LLM response without function call
        mock_llm_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="I don't understand.",
                function_call=None,
                tokens_used=50,
                latency_ms=500,
                model_used="mock-model",
            )
        )

        processor = ResponseProcessor(
            llm_client=mock_llm_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await processor.process_response(
            participant_response="Test",
            conversation_history=[],
            existing_graph=empty_graph,
            turn_number=1,
        )

        # Should return empty delta with warning
        assert len(delta.nodes_added) == 0
        assert len(delta.edges_added) == 0
        assert "warning" in delta.extraction_metadata
        assert delta.extraction_metadata["warning"] == "No function call returned"

    @pytest.mark.asyncio
    async def test_process_response_validation_errors(
        self, mock_llm_client, prompt_builder, validator, empty_graph
    ):
        """Test handling of validation errors."""
        # Mock LLM response with invalid data
        mock_llm_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={
                    "nodes_added": [
                        {
                            "type": "invalid_type",  # Invalid
                            "label": "test_node",
                            "quote": "test",
                        }
                    ],
                    "edges_added": [],
                },
                tokens_used=100,
                latency_ms=800,
                model_used="mock-model",
            )
        )

        processor = ResponseProcessor(
            llm_client=mock_llm_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await processor.process_response(
            participant_response="Test response",
            conversation_history=[],
            existing_graph=empty_graph,
            turn_number=1,
        )

        # Should have validation errors in metadata
        assert len(delta.nodes_added) == 0  # Invalid node rejected
        assert len(delta.extraction_metadata["validation_errors"]) > 0
        assert any(
            "unknown node type" in err.lower()
            for err in delta.extraction_metadata["validation_errors"]
        )

    @pytest.mark.asyncio
    async def test_process_response_empty_extraction(
        self, mock_llm_client, prompt_builder, validator, empty_graph
    ):
        """Test handling of empty but valid extraction."""
        # Mock LLM response with empty extraction
        mock_llm_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={"nodes_added": [], "edges_added": []},
                tokens_used=50,
                latency_ms=600,
                model_used="mock-model",
            )
        )

        processor = ResponseProcessor(
            llm_client=mock_llm_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await processor.process_response(
            participant_response="Okay.",
            conversation_history=[],
            existing_graph=empty_graph,
            turn_number=1,
        )

        # Empty extraction is valid
        assert len(delta.nodes_added) == 0
        assert len(delta.edges_added) == 0
        assert delta.richness_score == 0.0
        assert len(delta.extraction_metadata["validation_errors"]) == 0

    @pytest.mark.asyncio
    async def test_process_response_richness_calculation(
        self, mock_llm_client, prompt_builder, validator, empty_graph
    ):
        """Test richness score calculation."""
        # Mock LLM response
        mock_llm_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={
                    "nodes_added": [
                        {
                            "type": "attribute",  # weight 0.5
                            "label": "test_attr",
                            "quote": "test",
                        },
                        {
                            "type": "value",  # weight 2.0
                            "label": "test_value",
                            "quote": "test",
                        },
                    ],
                    "edges_added": [
                        {
                            "type": "leads_to",  # boost 1.0
                            "source": "test_attr",
                            "target": "test_value",
                            "quote": "test",
                        }
                    ],
                },
                tokens_used=100,
                latency_ms=800,
                model_used="mock-model",
            )
        )

        processor = ResponseProcessor(
            llm_client=mock_llm_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await processor.process_response(
            participant_response="Test response",
            conversation_history=[],
            existing_graph=empty_graph,
            turn_number=1,
        )

        # Richness = 0.5 (attribute) + 2.0 (value) + 1.0 (leads_to) = 3.5
        assert delta.richness_score == pytest.approx(3.5)
