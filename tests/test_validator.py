"""
Tests for validator (4-stage validation).
"""

import pytest

from src.core.data_models import Node
from src.core.interview_graph import InterviewGraph
from src.core.schema_manager import SchemaManager
from src.interview.validator import Validator


@pytest.fixture
def schema_manager():
    """Create schema manager with means-end chain schema."""
    schema = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    schema.load_schema()
    schema.validate_schema()
    return schema


@pytest.fixture
def empty_graph(schema_manager):
    """Create empty interview graph."""
    return InterviewGraph(schema_manager)


@pytest.fixture
def graph_with_nodes(schema_manager):
    """Create graph with some existing nodes."""
    graph = InterviewGraph(schema_manager)

    # Add some nodes
    nodes = [
        Node(
            id="affordable_price",
            type="attribute",
            label="affordable_price",
            source_quotes=["it's affordable"],
            creation_turn=1,
            visit_count=1,
        ),
        Node(
            id="convenience",
            type="value",
            label="convenience",
            source_quotes=["convenient"],
            creation_turn=1,
            visit_count=1,
        ),
    ]

    for node in nodes:
        graph.add_node(node)

    return graph


class TestValidator:
    """Tests for Validator."""

    def test_validate_valid_extraction(self, schema_manager, empty_graph):
        """Test validation of a valid extraction."""
        validator = Validator(schema_manager)

        raw_extraction = {
            "nodes_added": [
                {
                    "type": "attribute",
                    "label": "affordable_price",
                    "quote": "it's affordable",
                }
            ],
            "edges_added": [],
        }

        participant_response = "I like that it's affordable and convenient."

        result = validator.validate(raw_extraction, empty_graph, participant_response)

        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.cleaned_nodes) == 1
        assert result.cleaned_nodes[0].label == "affordable_price"

    def test_validate_unknown_node_type(self, schema_manager, empty_graph):
        """Test rejection of unknown node type."""
        validator = Validator(schema_manager)

        raw_extraction = {
            "nodes_added": [{"type": "invalid_type", "label": "test_node", "quote": "test"}],
            "edges_added": [],
        }

        result = validator.validate(raw_extraction, empty_graph, "test response")

        assert not result.is_valid
        assert len(result.errors) > 0
        assert "unknown node type" in result.errors[0].lower()

    def test_validate_invalid_label_format(self, schema_manager, empty_graph):
        """Test rejection of invalid label format."""
        validator = Validator(schema_manager)

        invalid_labels = [
            "CamelCase",  # Uppercase
            "spaces in label",  # Spaces
            "ab",  # Too short
            "a" * 41,  # Too long
            "123label",  # Starts with number
        ]

        for label in invalid_labels:
            raw_extraction = {
                "nodes_added": [{"type": "attribute", "label": label, "quote": "test"}],
                "edges_added": [],
            }

            result = validator.validate(raw_extraction, empty_graph, "test")

            assert not result.is_valid
            assert any("invalid label format" in err.lower() for err in result.errors)

    def test_validate_missing_quote_warning(self, schema_manager, empty_graph):
        """Test warning for quote not in response."""
        validator = Validator(schema_manager)

        raw_extraction = {
            "nodes_added": [
                {
                    "type": "attribute",
                    "label": "affordable_price",
                    "quote": "this quote is not in the response",
                }
            ],
            "edges_added": [],
        }

        participant_response = "I like that it's affordable."

        result = validator.validate(raw_extraction, empty_graph, participant_response)

        # Still valid (warnings don't fail validation)
        assert result.is_valid
        assert len(result.warnings) > 0
        assert "not found in response" in result.warnings[0].lower()

    def test_validate_edge_with_nonexistent_nodes(self, schema_manager, empty_graph):
        """Test rejection of edge referencing nonexistent nodes."""
        validator = Validator(schema_manager)

        raw_extraction = {
            "nodes_added": [],
            "edges_added": [
                {
                    "type": "leads_to",
                    "source": "node_a",  # Doesn't exist
                    "target": "node_b",  # Doesn't exist
                    "quote": "test",
                }
            ],
        }

        result = validator.validate(raw_extraction, empty_graph, "test")

        assert not result.is_valid
        assert any("not found in nodes" in err for err in result.errors)

    def test_validate_edge_type_compatibility(self, schema_manager, graph_with_nodes):
        """Test edge type compatibility validation."""
        validator = Validator(schema_manager)

        # Try to create invalid edge: value -> attribute (not allowed)
        raw_extraction = {
            "nodes_added": [],
            "edges_added": [
                {
                    "type": "leads_to",
                    "source": "convenience",  # value type
                    "target": "affordable_price",  # attribute type
                    "quote": "test",
                }
            ],
        }

        result = validator.validate(raw_extraction, graph_with_nodes, "test")

        # Should fail because leads_to from value to attribute is invalid
        assert not result.is_valid
        assert any("invalid edge" in err.lower() for err in result.errors)

    def test_validate_valid_edge(self, schema_manager, graph_with_nodes):
        """Test validation of a valid edge."""
        validator = Validator(schema_manager)

        # Add a functional consequence node
        func_cons = Node(
            id="regular_purchase",
            type="functional_consequence",
            label="regular_purchase",
            source_quotes=["buy it regularly"],
            creation_turn=2,
            visit_count=1,
        )
        graph_with_nodes.add_node(func_cons)

        # Create valid edge: attribute -> functional_consequence
        raw_extraction = {
            "nodes_added": [],
            "edges_added": [
                {
                    "type": "leads_to",
                    "source": "affordable_price",
                    "target": "regular_purchase",
                    "quote": "affordable so I buy it regularly",
                }
            ],
        }

        participant_response = "It's affordable so I buy it regularly."

        result = validator.validate(raw_extraction, graph_with_nodes, participant_response)

        assert result.is_valid
        assert len(result.cleaned_edges) == 1
        assert result.cleaned_edges[0].source == "affordable_price"
        assert result.cleaned_edges[0].target == "regular_purchase"

    def test_validate_mixed_new_and_existing_nodes(self, schema_manager, graph_with_nodes):
        """Test edge validation with mix of new and existing nodes."""
        validator = Validator(schema_manager)

        # Add new node and create edge to existing node
        raw_extraction = {
            "nodes_added": [
                {
                    "type": "functional_consequence",
                    "label": "regular_purchase",
                    "quote": "buy it regularly",
                }
            ],
            "edges_added": [
                {
                    "type": "leads_to",
                    "source": "affordable_price",  # Existing
                    "target": "regular_purchase",  # New
                    "quote": "affordable so I buy it regularly",
                }
            ],
        }

        participant_response = "It's affordable so I buy it regularly."

        result = validator.validate(raw_extraction, graph_with_nodes, participant_response)

        assert result.is_valid
        assert len(result.cleaned_nodes) == 1
        assert len(result.cleaned_edges) == 1

    def test_validate_missing_required_fields(self, schema_manager, empty_graph):
        """Test validation of missing required fields."""
        validator = Validator(schema_manager)

        # Missing 'type' field in node
        raw_extraction = {
            "nodes_added": [{"label": "test_node", "quote": "test"}],
            "edges_added": [],
        }

        result = validator.validate(raw_extraction, empty_graph, "test")

        assert not result.is_valid
        assert any("missing 'type'" in err.lower() for err in result.errors)

    def test_validate_invalid_structure(self, schema_manager, empty_graph):
        """Test validation of invalid structure (not a dict)."""
        validator = Validator(schema_manager)

        raw_extraction = "not a dictionary"

        result = validator.validate(raw_extraction, empty_graph, "test")

        assert not result.is_valid
        assert "not a dictionary" in result.errors[0].lower()

    def test_validate_empty_extraction(self, schema_manager, empty_graph):
        """Test validation of empty but valid extraction."""
        validator = Validator(schema_manager)

        raw_extraction = {"nodes_added": [], "edges_added": []}

        result = validator.validate(raw_extraction, empty_graph, "test")

        assert result.is_valid
        assert len(result.cleaned_nodes) == 0
        assert len(result.cleaned_edges) == 0
