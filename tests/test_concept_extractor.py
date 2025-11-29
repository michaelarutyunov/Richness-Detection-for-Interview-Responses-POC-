"""
Tests for ConceptExtractor (seed node extraction from concept description).
"""

from unittest.mock import AsyncMock

import pytest

from src.core.schema_manager import SchemaManager
from src.interview.concept_extractor import ConceptExtractor
from src.interview.prompt_builder import PromptBuilder
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
def mock_llm():
    """Create mock LLM client."""
    mock_client = AsyncMock()
    mock_client.model = "mock-extraction"
    return mock_client


@pytest.fixture
def prompt_builder():
    """Create prompt builder."""
    return PromptBuilder()


@pytest.fixture
def validator(schema_manager):
    """Create validator."""
    return Validator(schema_manager)


class TestConceptExtractor:
    """Tests for ConceptExtractor."""

    @pytest.mark.asyncio
    async def test_extract_seed_nodes_success(self, mock_llm, prompt_builder, validator):
        """Test successful seed node extraction."""
        # Mock LLM response with valid seed nodes
        mock_llm.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={
                    "nodes_added": [
                        {
                            "type": "attribute",
                            "label": "affordable_price",
                            "quote": "affordable",
                        },
                        {
                            "type": "attribute",
                            "label": "local_sourcing",
                            "quote": "local roasters",
                        },
                        {
                            "type": "functional_consequence",
                            "label": "monthly_delivery",
                            "quote": "delivers every month",
                        },
                    ],
                    "edges_added": [],
                },
                tokens_used=200,
                latency_ms=1500,
                model_used="mock-extraction",
            )
        )

        extractor = ConceptExtractor(
            llm_client=mock_llm,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        concept_description = (
            "A premium coffee subscription service that delivers freshly roasted beans "
            "from local roasters every month. It's affordable and convenient."
        )

        delta = await extractor.extract_seed_nodes(concept_description)

        # Verify delta structure
        assert len(delta.nodes_added) == 3
        assert len(delta.edges_added) == 0  # Seed extraction doesn't create edges
        assert delta.richness_score > 0
        assert delta.extraction_metadata["model_used"] == "mock-extraction"
        assert delta.extraction_metadata["nodes_extracted"] == 3
        assert len(delta.extraction_metadata["validation_errors"]) == 0

        # Verify nodes
        node_labels = {n.label for n in delta.nodes_added}
        assert "affordable_price" in node_labels
        assert "local_sourcing" in node_labels
        assert "monthly_delivery" in node_labels

        # All seed nodes should have turn=0 and visit_count=0
        for node in delta.nodes_added:
            assert node.creation_turn == 0
            assert node.visit_count == 0

    @pytest.mark.asyncio
    async def test_extract_seed_nodes_llm_failure(self, mock_llm, prompt_builder, validator):
        """Test handling of LLM failure."""
        from src.llm.exceptions import LLMProviderError

        # Mock LLM failure
        mock_llm.generate_with_retry = AsyncMock(side_effect=LLMProviderError("API error"))

        extractor = ConceptExtractor(
            llm_client=mock_llm,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await extractor.extract_seed_nodes("Test concept")

        # Should return empty delta with error
        assert len(delta.nodes_added) == 0
        assert "error" in delta.extraction_metadata
        assert "API error" in delta.extraction_metadata["error"]

    @pytest.mark.asyncio
    async def test_extract_seed_nodes_no_function_call(self, mock_llm, prompt_builder, validator):
        """Test handling when LLM doesn't return function call."""
        # Mock LLM response without function call
        mock_llm.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="I don't understand this concept.",
                function_call=None,
                tokens_used=50,
                latency_ms=500,
                model_used="mock-extraction",
            )
        )

        extractor = ConceptExtractor(
            llm_client=mock_llm,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await extractor.extract_seed_nodes("Test concept")

        # Should return empty delta with warning
        assert len(delta.nodes_added) == 0
        assert "warning" in delta.extraction_metadata
        assert delta.extraction_metadata["warning"] == "No function call returned"

    @pytest.mark.asyncio
    async def test_extract_seed_nodes_validation_errors(self, mock_llm, prompt_builder, validator):
        """Test handling of validation errors in extracted nodes."""
        # Mock LLM response with invalid node types
        mock_llm.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={
                    "nodes_added": [
                        {
                            "type": "invalid_type",  # Invalid
                            "label": "test_node",
                            "quote": "test",
                        },
                        {
                            "type": "attribute",
                            "label": "CamelCase",  # Invalid format
                            "quote": "test",
                        },
                    ],
                    "edges_added": [],
                },
                tokens_used=100,
                latency_ms=800,
                model_used="mock-extraction",
            )
        )

        extractor = ConceptExtractor(
            llm_client=mock_llm,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await extractor.extract_seed_nodes("Test concept")

        # Invalid nodes should be rejected
        assert len(delta.nodes_added) == 0
        assert len(delta.extraction_metadata["validation_errors"]) > 0

    @pytest.mark.asyncio
    async def test_extract_seed_nodes_empty_extraction(self, mock_llm, prompt_builder, validator):
        """Test handling of empty but valid extraction."""
        # Mock LLM response with no nodes
        mock_llm.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={"nodes_added": [], "edges_added": []},
                tokens_used=50,
                latency_ms=600,
                model_used="mock-extraction",
            )
        )

        extractor = ConceptExtractor(
            llm_client=mock_llm,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await extractor.extract_seed_nodes("Very vague concept")

        # Empty extraction is valid
        assert len(delta.nodes_added) == 0
        assert delta.richness_score == 0.0
        assert len(delta.extraction_metadata["validation_errors"]) == 0

    @pytest.mark.asyncio
    async def test_extract_seed_nodes_richness_calculation(
        self, mock_llm, prompt_builder, validator
    ):
        """Test richness score calculation for seed nodes."""
        # Mock LLM response with nodes of different weights
        mock_llm.generate_with_retry = AsyncMock(
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
                    "edges_added": [],
                },
                tokens_used=100,
                latency_ms=800,
                model_used="mock-extraction",
            )
        )

        extractor = ConceptExtractor(
            llm_client=mock_llm,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        delta = await extractor.extract_seed_nodes("Test concept")

        # Richness = 0.5 (attribute) + 2.0 (value) = 2.5
        assert delta.richness_score == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_concept_prompt_structure(self, mock_llm, prompt_builder, validator):
        """Test that concept extraction prompts are properly formatted."""
        mock_llm.generate_with_retry = AsyncMock(
            return_value=LLMResponse(
                content="",
                function_call={"nodes_added": [], "edges_added": []},
                tokens_used=50,
                latency_ms=500,
                model_used="mock-extraction",
            )
        )

        extractor = ConceptExtractor(
            llm_client=mock_llm,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        await extractor.extract_seed_nodes("A coffee maker")

        # Verify LLM was called with correct structure
        call_args = mock_llm.generate_with_retry.call_args
        messages = call_args[0][0]
        function_schema = call_args[0][1]

        # Should have system and user messages
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # System message should mention seed nodes
        assert (
            "seed" in messages[0]["content"].lower() or "initial" in messages[0]["content"].lower()
        )

        # Function schema should be for graph delta
        assert function_schema["name"] == "extract_graph_delta"
        assert "nodes_added" in function_schema["parameters"]["properties"]
