"""
Tests for QuestionGenerator.
"""

from unittest.mock import AsyncMock

import pytest

from src.core.data_models import Node
from src.core.interview_graph import InterviewGraph
from src.core.schema_manager import SchemaManager
from src.interview.opportunity_ranker import QuestionStrategy, RankedOpportunity
from src.interview.question_generator import QuestionGenerator
from src.llm.base_client import LLMResponse


@pytest.fixture
def schema_manager():
    """Create schema manager."""
    schema = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    schema.load_schema()
    schema.validate_schema()
    return schema


@pytest.fixture
def simple_graph(schema_manager):
    """Create simple graph."""
    graph = InterviewGraph(schema_manager)

    node = Node(
        id="affordable_price",
        type="attribute",
        label="affordable_price",
        source_quotes=["it's affordable"],
        creation_turn=1,
        visit_count=1,
    )

    graph.add_node(node)
    return graph


@pytest.fixture
def mock_llm():
    """Create mock LLM client."""
    mock_client = AsyncMock()
    mock_client.model = "mock-model"
    return mock_client


class TestQuestionGenerator:
    """Tests for QuestionGenerator."""

    def test_init_without_llm(self):
        """Test initialization without LLM."""
        gen = QuestionGenerator(llm_client=None)

        assert gen.llm is None
        assert not gen.use_llm
        assert gen._templates is not None

    def test_init_with_llm(self, mock_llm):
        """Test initialization with LLM."""
        gen = QuestionGenerator(llm_client=mock_llm, use_llm=True)

        assert gen.llm == mock_llm
        assert gen.use_llm

    @pytest.mark.asyncio
    async def test_generate_dig_deeper_template(self, simple_graph):
        """Test template-based question for DIG_DEEPER."""
        gen = QuestionGenerator(llm_client=None, use_llm=False)

        opportunity = RankedOpportunity(
            node_id="affordable_price",
            node_label="affordable_price",
            node_type="attribute",
            strategy=QuestionStrategy.DIG_DEEPER,
            priority_score=10.0,
            rationale="test",
            metadata={},
        )

        question = await gen.generate_question(opportunity, simple_graph)

        assert "affordable_price" in question
        assert question.endswith("?")
        assert len(question) > 0

    @pytest.mark.asyncio
    async def test_generate_introduce_topic_template(self, simple_graph):
        """Test template-based question for INTRODUCE_TOPIC."""
        gen = QuestionGenerator(llm_client=None, use_llm=False)

        opportunity = RankedOpportunity(
            node_id="affordable_price",
            node_label="affordable_price",
            node_type="attribute",
            strategy=QuestionStrategy.INTRODUCE_TOPIC,
            priority_score=10.0,
            rationale="test",
            metadata={},
        )

        question = await gen.generate_question(opportunity, simple_graph)

        assert "affordable_price" in question
        assert question.endswith("?")

    @pytest.mark.asyncio
    async def test_generate_with_llm_success(self, simple_graph, mock_llm):
        """Test LLM-based question generation."""
        mock_llm.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="What makes affordable pricing important to you?",
                function_call=None,
                tokens_used=20,
                latency_ms=500,
                model_used="mock-model",
            )
        )

        gen = QuestionGenerator(llm_client=mock_llm, use_llm=True)

        opportunity = RankedOpportunity(
            node_id="affordable_price",
            node_label="affordable_price",
            node_type="attribute",
            strategy=QuestionStrategy.DIG_DEEPER,
            priority_score=10.0,
            rationale="test",
            metadata={},
        )

        conversation_history = [
            {"role": "assistant", "content": "What do you like?"},
            {"role": "user", "content": "I like that it's affordable."},
        ]

        question = await gen.generate_question(opportunity, simple_graph, conversation_history)

        assert len(question) > 0
        assert question.endswith("?")
        mock_llm.generate_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_llm_fallback_to_template(self, simple_graph, mock_llm):
        """Test fallback to templates when LLM fails."""
        # Mock LLM failure
        mock_llm.generate_with_retry = AsyncMock(side_effect=Exception("API error"))

        gen = QuestionGenerator(llm_client=mock_llm, use_llm=True)

        opportunity = RankedOpportunity(
            node_id="affordable_price",
            node_label="affordable_price",
            node_type="attribute",
            strategy=QuestionStrategy.DIG_DEEPER,
            priority_score=10.0,
            rationale="test",
            metadata={},
        )

        question = await gen.generate_question(opportunity, simple_graph)

        # Should still get a question from template
        assert len(question) > 0
        assert question.endswith("?")

    def test_get_opening_question(self):
        """Test opening question."""
        gen = QuestionGenerator(llm_client=None)

        question = gen.get_opening_question()

        assert len(question) > 0
        assert question.endswith("?")

    def test_get_closing_question(self):
        """Test closing question."""
        gen = QuestionGenerator(llm_client=None)

        question = gen.get_closing_question()

        assert len(question) > 0
        assert question.endswith("?")

    def test_post_process_question(self):
        """Test question post-processing."""
        gen = QuestionGenerator(llm_client=None)

        # Test adding question mark
        q1 = gen._post_process_question("Tell me more")
        assert q1.endswith("?")

        # Test capitalization
        q2 = gen._post_process_question("what do you think")
        assert q2[0].isupper()

        # Test already formatted
        q3 = gen._post_process_question("What do you think?")
        assert q3 == "What do you think?"
