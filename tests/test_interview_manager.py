"""
Tests for InterviewManager (integration tests).
"""

from unittest.mock import AsyncMock

import pytest

from src.core.schema_manager import SchemaManager
from src.interview.interview_manager import InterviewManager
from src.llm.base_client import LLMResponse


@pytest.fixture
def schema_manager():
    """Create schema manager."""
    schema = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    schema.load_schema()
    schema.validate_schema()
    return schema


@pytest.fixture
def mock_extraction_client():
    """Create mock extraction client."""
    mock_client = AsyncMock()
    mock_client.model = "mock-extraction"
    return mock_client


@pytest.fixture
def mock_question_client():
    """Create mock question client."""
    mock_client = AsyncMock()
    mock_client.model = "mock-questions"
    return mock_client


class TestInterviewManager:
    """Tests for InterviewManager."""

    @pytest.mark.asyncio
    async def test_start_interview(
        self, schema_manager, mock_extraction_client, mock_question_client
    ):
        """Test starting an interview."""
        manager = InterviewManager(
            schema_manager=schema_manager,
            extraction_client=mock_extraction_client,
            question_client=mock_question_client,
        )

        question = await manager.start_interview()

        assert len(question) > 0
        assert question.endswith("?")
        assert manager._interview_started
        assert manager.turn_number == 0
        assert len(manager.conversation_history) == 1

    @pytest.mark.asyncio
    async def test_process_response(
        self, schema_manager, mock_extraction_client, mock_question_client
    ):
        """Test processing a participant response."""
        # Mock extraction response
        mock_extraction_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={
                    "nodes_added": [
                        {
                            "type": "attribute",
                            "label": "affordable_price",
                            "quote": "it's affordable",
                        }
                    ],
                    "edges_added": [],
                },
                tokens_used=100,
                latency_ms=1000,
                model_used="mock-extraction",
            )
        )

        # Mock question generation response
        mock_question_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="What makes that important to you?",
                function_call=None,
                tokens_used=20,
                latency_ms=500,
                model_used="mock-questions",
            )
        )

        manager = InterviewManager(
            schema_manager=schema_manager,
            extraction_client=mock_extraction_client,
            question_client=mock_question_client,
        )

        # Start interview
        await manager.start_interview()

        # Process response
        next_question = await manager.process_response("I like that it's affordable.")

        assert len(next_question) > 0
        assert next_question.endswith("?")
        assert manager.turn_number == 1
        assert manager.graph.node_count == 1

    @pytest.mark.asyncio
    async def test_should_continue(
        self, schema_manager, mock_extraction_client, mock_question_client
    ):
        """Test continuation logic."""
        manager = InterviewManager(
            schema_manager=schema_manager,
            extraction_client=mock_extraction_client,
            question_client=mock_question_client,
            min_richness=100.0,  # High threshold
            max_turns=2,  # Low max turns
        )

        # Initially should continue (no turns yet)
        assert manager.should_continue()

        # Mock extraction
        mock_extraction_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={"nodes_added": [], "edges_added": []},
                tokens_used=50,
                latency_ms=500,
                model_used="mock",
            )
        )

        await manager.start_interview()
        await manager.process_response("Test 1")
        await manager.process_response("Test 2")

        # Should stop after 2 turns
        assert not manager.should_continue()

    @pytest.mark.asyncio
    async def test_get_summary(self, schema_manager, mock_extraction_client, mock_question_client):
        """Test getting interview summary."""
        manager = InterviewManager(
            schema_manager=schema_manager,
            extraction_client=mock_extraction_client,
            question_client=mock_question_client,
        )

        await manager.start_interview()

        summary = manager.get_summary()

        assert "nodes" in summary
        assert "edges" in summary
        assert "richness" in summary
        assert "coverage" in summary
        assert "turns" in summary
        assert "is_complete" in summary

        assert summary["turns"] == 0
        assert summary["nodes"] == 0

    @pytest.mark.asyncio
    async def test_full_interview_flow(
        self, schema_manager, mock_extraction_client, mock_question_client
    ):
        """Test complete interview flow."""
        # Mock extraction with valid data
        mock_extraction_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={
                    "nodes_added": [
                        {
                            "type": "attribute",
                            "label": "affordable_price",
                            "quote": "affordable",
                        }
                    ],
                    "edges_added": [],
                },
                tokens_used=100,
                latency_ms=1000,
                model_used="mock",
            )
        )

        # Mock question generation (template-based fallback)
        mock_question_client.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="What else?", tokens_used=10, latency_ms=200, model_used="mock"
            )
        )

        manager = InterviewManager(
            schema_manager=schema_manager,
            extraction_client=mock_extraction_client,
            question_client=mock_question_client,
            min_richness=2.0,  # Low threshold for quick test
            max_turns=3,
        )

        # 1. Start
        q1 = await manager.start_interview()
        assert len(q1) > 0

        # 2. Response 1
        q2 = await manager.process_response("It's affordable.")
        assert len(q2) > 0
        assert manager.graph.node_count >= 1

        # 3. Response 2
        q3 = await manager.process_response("It's convenient.")
        assert len(q3) > 0

        # 4. Get transcript
        transcript = manager.get_conversation_transcript()
        assert len(transcript) >= 3

    def test_export_graph(
        self, schema_manager, mock_extraction_client, mock_question_client, tmp_path
    ):
        """Test graph export."""
        manager = InterviewManager(
            schema_manager=schema_manager,
            extraction_client=mock_extraction_client,
            question_client=mock_question_client,
        )

        output_path = tmp_path / "test_graph.graphml"
        manager.export_graph(str(output_path))

        # File should exist (even if empty graph)
        assert output_path.exists()
