"""
End-to-end integration tests for UI workflow (mocked LLMs).
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.llm.base_client import LLMResponse
from src.ui.gradio_app import InterviewSession


class TestUIIntegration:
    """End-to-end UI integration tests."""

    @pytest.mark.asyncio
    async def test_full_interview_flow(self):
        """Test complete interview flow from concept to completion."""
        # Mock LLM responses
        extraction_responses = [
            # Seed extraction
            LLMResponse(
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
            ),
            # Turn 1 extraction
            LLMResponse(
                content="",
                function_call={
                    "nodes_added": [
                        {
                            "type": "functional_consequence",
                            "label": "regular_purchase",
                            "quote": "buy monthly",
                        }
                    ],
                    "edges_added": [
                        {
                            "type": "leads_to",
                            "source": "affordable_price",
                            "target": "regular_purchase",
                            "quote": "affordable so I buy monthly",
                        }
                    ],
                },
                tokens_used=150,
                latency_ms=1200,
                model_used="mock",
            ),
        ]

        question_responses = [
            # Opening question
            LLMResponse(
                content="What do you like about this coffee service?",
                function_call=None,
                tokens_used=20,
                latency_ms=400,
                model_used="mock",
            ),
            # Follow-up question
            LLMResponse(
                content="What makes that important to you?",
                function_call=None,
                tokens_used=15,
                latency_ms=350,
                model_used="mock",
            ),
        ]

        # Create session with mocked clients
        with (
            patch("src.ui.gradio_app.KimiClient") as MockKimi,  # noqa: N806
            patch("src.ui.gradio_app.AnthropicClient") as MockAnthropic,  # noqa: N806
        ):

            # Setup mocks
            mock_kimi = MockKimi.return_value
            mock_kimi.model = "mock-extraction"
            mock_kimi.generate_with_retry = AsyncMock(side_effect=extraction_responses)

            mock_anthropic = MockAnthropic.return_value
            mock_anthropic.model = "mock-questions"
            mock_anthropic.generate_with_retry = AsyncMock(side_effect=question_responses)

            # Create session
            session = InterviewSession(
                schema_path="schemas/means_end_chain_v0.1.yaml",
                concept_description="A premium coffee subscription service that is affordable.",
            )

            # Start interview
            opening_question = await session.start()

            assert len(opening_question) > 0
            assert opening_question.endswith("?")
            assert session.seeds_extracted

            # Check initial stats
            stats = session.get_stats()
            assert stats["nodes"] >= 1  # At least seed node
            assert stats["turns"] == 0

            # Process first response
            next_question = await session.process_response(
                "I like that it's affordable, so I can buy it every month."
            )

            assert len(next_question) > 0
            assert next_question.endswith("?")

            # Check updated stats
            stats = session.get_stats()
            assert stats["nodes"] >= 2  # Seed + extracted node
            assert stats["turns"] == 1
            assert stats["richness"] > 0

    @pytest.mark.asyncio
    async def test_session_initialization(self):
        """Test session initialization and seed extraction."""
        with (
            patch("src.ui.gradio_app.KimiClient") as MockKimi,  # noqa: N806
            patch("src.ui.gradio_app.AnthropicClient"),
        ):

            mock_kimi = MockKimi.return_value
            mock_kimi.model = "mock"
            mock_kimi.generate_with_retry = AsyncMock(
                return_value=LLMResponse(
                    content="",
                    function_call={
                        "nodes_added": [
                            {
                                "type": "attribute",
                                "label": "test_attr",
                                "quote": "test",
                            }
                        ],
                        "edges_added": [],
                    },
                    tokens_used=100,
                    latency_ms=1000,
                    model_used="mock",
                )
            )

            session = InterviewSession(
                schema_path="schemas/means_end_chain_v0.1.yaml",
                concept_description="Test concept",
            )

            # Before initialization
            assert not session.seeds_extracted
            assert session.manager is None

            # Initialize
            await session.initialize()

            # After initialization
            assert session.seeds_extracted
            assert session.manager is not None
            assert session.manager.graph.node_count >= 0

    @pytest.mark.asyncio
    async def test_session_continuation_logic(self):
        """Test session knows when interview should continue."""
        with (
            patch("src.ui.gradio_app.KimiClient") as MockKimi,  # noqa: N806
            patch("src.ui.gradio_app.AnthropicClient"),
        ):

            # Mock seed extraction
            mock_kimi = MockKimi.return_value
            mock_kimi.model = "mock"
            mock_kimi.generate_with_retry = AsyncMock(
                return_value=LLMResponse(
                    content="",
                    function_call={"nodes_added": [], "edges_added": []},
                    tokens_used=50,
                    latency_ms=500,
                    model_used="mock",
                )
            )

            # Create session with low max_turns
            session = InterviewSession(
                schema_path="schemas/means_end_chain_v0.1.yaml",
                concept_description="Test",
            )

            await session.initialize()

            # Set low max turns for testing
            session.manager.max_turns = 2

            # Initially not complete
            assert not session.is_complete()

            # After 2 turns, should be complete
            # We need to manually set turn_number since we're not calling process_response
            session.manager.turn_number = 2

            assert session.is_complete()

    @pytest.mark.asyncio
    async def test_session_stats_update(self):
        """Test that session stats update correctly during interview."""
        with (
            patch("src.ui.gradio_app.KimiClient") as MockKimi,  # noqa: N806
            patch("src.ui.gradio_app.AnthropicClient") as MockAnthropic,  # noqa: N806
        ):

            # Mock responses
            mock_kimi = MockKimi.return_value
            mock_kimi.model = "mock"
            mock_kimi.generate_with_retry = AsyncMock(
                side_effect=[
                    # Seed extraction
                    LLMResponse(
                        content="",
                        function_call={
                            "nodes_added": [
                                {
                                    "type": "attribute",
                                    "label": "test_attr",
                                    "quote": "test",
                                }
                            ],
                            "edges_added": [],
                        },
                        tokens_used=100,
                        latency_ms=1000,
                        model_used="mock",
                    ),
                    # Turn 1 extraction
                    LLMResponse(
                        content="",
                        function_call={
                            "nodes_added": [
                                {
                                    "type": "value",
                                    "label": "test_value",
                                    "quote": "test",
                                }
                            ],
                            "edges_added": [],
                        },
                        tokens_used=100,
                        latency_ms=1000,
                        model_used="mock",
                    ),
                ]
            )

            mock_anthropic = MockAnthropic.return_value
            mock_anthropic.model = "mock"
            mock_anthropic.generate_with_retry = AsyncMock(
                return_value=LLMResponse(
                    content="What else?",
                    function_call=None,
                    tokens_used=10,
                    latency_ms=200,
                    model_used="mock",
                )
            )

            session = InterviewSession(
                schema_path="schemas/means_end_chain_v0.1.yaml",
                concept_description="Test",
            )

            # Initial stats
            stats_before = session.get_stats()
            assert stats_before["nodes"] == 0
            assert stats_before["turns"] == 0

            # Start interview (adds seed node)
            await session.start()

            stats_after_start = session.get_stats()
            assert stats_after_start["nodes"] >= 1  # Seed node added
            assert stats_after_start["turns"] == 0

            # Process response (adds another node)
            await session.process_response("Test response")

            stats_after_turn = session.get_stats()
            assert stats_after_turn["nodes"] >= 2  # Seed + new node
            assert stats_after_turn["turns"] == 1
            assert stats_after_turn["richness"] > stats_after_start["richness"]

    @pytest.mark.asyncio
    async def test_session_handles_llm_errors_gracefully(self):
        """Test that session handles LLM errors without crashing."""
        from src.llm.exceptions import LLMProviderError

        with (
            patch("src.ui.gradio_app.KimiClient") as MockKimi,  # noqa: N806
            patch("src.ui.gradio_app.AnthropicClient"),
        ):

            # Mock LLM failure
            mock_kimi = MockKimi.return_value
            mock_kimi.model = "mock"
            mock_kimi.generate_with_retry = AsyncMock(side_effect=LLMProviderError("API error"))

            session = InterviewSession(
                schema_path="schemas/means_end_chain_v0.1.yaml",
                concept_description="Test",
            )

            # Should not crash, even with LLM error
            await session.initialize()

            # Session should still be created (even if no seeds extracted)
            assert session.manager is not None
            assert session.seeds_extracted  # Marked as extracted even if failed
