"""
Unit tests for InterviewGraph.
"""

import pytest

from src.core.data_models import Edge, GraphDelta, Node
from src.core.interview_graph import InterviewGraph
from src.core.schema_manager import SchemaManager


@pytest.fixture
def schema_manager():
    """Create schema manager for tests."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()
    manager.validate_schema()
    return manager


@pytest.fixture
def graph(schema_manager):
    """Create empty interview graph."""
    return InterviewGraph(schema_manager)


def test_add_valid_node(graph):
    """Test adding a valid node."""
    node = Node(
        id="n1",
        type="attribute",
        label="affordable_price",
        creation_turn=1,
        source_quotes=["it's affordable"],
    )

    result = graph.add_node(node)
    assert result is True
    assert graph.graph.has_node("n1")
    assert graph.node_count == 1


def test_add_invalid_node_type(graph):
    """Test adding node with unknown type raises error."""
    node = Node(id="n1", type="invalid_type", label="test", creation_turn=1)

    with pytest.raises(ValueError, match="Unknown node type"):
        graph.add_node(node)


def test_add_duplicate_node(graph):
    """Test adding duplicate node merges quotes."""
    node1 = Node(
        id="n1",
        type="attribute",
        label="affordable_price",
        creation_turn=1,
        source_quotes=["affordable"],
    )
    node2 = Node(
        id="n1",
        type="attribute",
        label="affordable_price",
        creation_turn=2,
        source_quotes=["cheap"],
    )

    result1 = graph.add_node(node1)
    result2 = graph.add_node(node2)

    assert result1 is True  # First is new
    assert result2 is False  # Second is merge

    existing = graph.get_node("n1")
    assert len(existing.source_quotes) == 2
    assert existing.visit_count == 1  # Incremented from 0
    assert graph.node_count == 1  # Still only 1 node


def test_add_valid_edge(graph):
    """Test adding a valid edge."""
    # Add nodes first
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1))
    graph.add_node(
        Node(id="n2", type="functional_consequence", label="regular_purchase", creation_turn=1)
    )

    edge = Edge(
        id="e1",
        type="leads_to",
        source="n1",
        target="n2",
        creation_turn=1,
        source_quote="affordable so I buy it weekly",
    )

    result = graph.add_edge(edge)
    assert result is True
    assert graph.graph.has_edge("n1", "n2")
    assert graph.edge_count == 1


def test_add_invalid_edge(graph):
    """Test adding invalid edge raises error."""
    graph.add_node(Node(id="n1", type="value", label="security", creation_turn=1))
    graph.add_node(Node(id="n2", type="attribute", label="price", creation_turn=1))

    # Can't go from value -> attribute per schema
    edge = Edge(
        id="e1", type="leads_to", source="n1", target="n2", creation_turn=1, source_quote="invalid"
    )

    with pytest.raises(ValueError, match="Invalid edge"):
        graph.add_edge(edge)


def test_add_edge_missing_source(graph):
    """Test adding edge with missing source node."""
    graph.add_node(Node(id="n2", type="attribute", label="price", creation_turn=1))

    edge = Edge(
        id="e1",
        type="leads_to",
        source="nonexistent",
        target="n2",
        creation_turn=1,
        source_quote="test",
    )

    with pytest.raises(ValueError, match="Source node not found"):
        graph.add_edge(edge)


def test_add_duplicate_edge(graph):
    """Test adding duplicate edge returns False."""
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1))
    graph.add_node(Node(id="n2", type="functional_consequence", label="regular", creation_turn=1))

    edge = Edge(
        id="e1", type="leads_to", source="n1", target="n2", creation_turn=1, source_quote="test"
    )

    result1 = graph.add_edge(edge)
    result2 = graph.add_edge(edge)

    assert result1 is True
    assert result2 is False
    assert graph.edge_count == 1


def test_apply_delta(graph):
    """Test applying a graph delta."""
    delta = GraphDelta(
        nodes_added=[
            Node(id="n1", type="attribute", label="price", creation_turn=1),
            Node(id="n2", type="functional_consequence", label="regular_purchase", creation_turn=1),
        ],
        edges_added=[
            Edge(
                id="e1",
                type="leads_to",
                source="n1",
                target="n2",
                creation_turn=1,
                source_quote="test",
            )
        ],
        richness_score=1.5,
    )

    nodes_added, edges_added = graph.apply_delta(delta, turn_number=1)
    assert nodes_added == 2
    assert edges_added == 1
    assert graph.node_count == 2
    assert graph.edge_count == 1


def test_get_node(graph):
    """Test retrieving node by ID."""
    node = Node(id="n1", type="attribute", label="price", creation_turn=1)
    graph.add_node(node)

    retrieved = graph.get_node("n1")
    assert retrieved is not None
    assert retrieved.id == "n1"
    assert retrieved.label == "price"

    # Non-existent node
    assert graph.get_node("nonexistent") is None


def test_get_neighbors_out(graph):
    """Test getting outgoing neighbors."""
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1))
    graph.add_node(Node(id="n2", type="functional_consequence", label="regular", creation_turn=1))
    graph.add_node(
        Node(id="n3", type="functional_consequence", label="convenient", creation_turn=1)
    )

    graph.add_edge(
        Edge(
            id="e1", type="leads_to", source="n1", target="n2", creation_turn=1, source_quote="test"
        )
    )
    graph.add_edge(
        Edge(
            id="e2", type="leads_to", source="n1", target="n3", creation_turn=1, source_quote="test"
        )
    )

    neighbors = graph.get_neighbors("n1", direction="out")
    assert len(neighbors) == 2
    assert all(isinstance(n, Node) for n in neighbors)


def test_get_neighbors_in(graph):
    """Test getting incoming neighbors."""
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1))
    graph.add_node(Node(id="n2", type="functional_consequence", label="regular", creation_turn=1))

    graph.add_edge(
        Edge(
            id="e1", type="leads_to", source="n1", target="n2", creation_turn=1, source_quote="test"
        )
    )

    incoming = graph.get_neighbors("n2", direction="in")
    assert len(incoming) == 1
    assert incoming[0].id == "n1"


def test_calculate_coverage(graph):
    """Test coverage calculation."""
    # Add one of each type
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1))
    graph.add_node(Node(id="n2", type="functional_consequence", label="regular", creation_turn=1))
    graph.add_node(Node(id="n3", type="psychosocial_consequence", label="peace", creation_turn=1))
    graph.add_node(Node(id="n4", type="value", label="security", creation_turn=1))

    coverage = graph.calculate_coverage()
    assert coverage["overall"] == 1.0  # All 4 types covered
    assert all(v == 1.0 for v in coverage["by_type"].values())
    assert coverage["node_counts"]["attribute"] == 1


def test_calculate_coverage_partial(graph):
    """Test partial coverage calculation."""
    # Only add 2 out of 4 types
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1))
    graph.add_node(Node(id="n2", type="value", label="security", creation_turn=1))

    coverage = graph.calculate_coverage()
    assert coverage["overall"] == 0.5  # 2/4 = 0.5
    assert coverage["by_type"]["attribute"] == 1.0
    assert coverage["by_type"]["value"] == 1.0
    assert coverage["by_type"]["functional_consequence"] == 0.0


def test_calculate_richness(graph):
    """Test richness calculation."""
    # Add nodes (0.5 + 1.0 + 1.5 + 2.0 = 5.0)
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1))
    graph.add_node(Node(id="n2", type="functional_consequence", label="regular", creation_turn=1))
    graph.add_node(Node(id="n3", type="psychosocial_consequence", label="peace", creation_turn=1))
    graph.add_node(Node(id="n4", type="value", label="security", creation_turn=1))

    # Add edge (boost = 1.0)
    graph.add_edge(
        Edge(
            id="e1", type="leads_to", source="n1", target="n2", creation_turn=1, source_quote="test"
        )
    )

    richness = graph.calculate_richness()
    assert richness == 6.0  # 5.0 from nodes + 1.0 from edge


def test_get_expansion_opportunities(graph):
    """Test opportunity identification."""
    # Use same type to test visit count effect
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1, visit_count=0))
    graph.add_node(Node(id="n2", type="attribute", label="quality", creation_turn=1, visit_count=3))

    opportunities = graph.get_expansion_opportunities()

    # n1 should rank higher (less visited, same type)
    assert len(opportunities) == 2
    assert opportunities[0].target_node_id == "n1"
    assert opportunities[0].priority > opportunities[1].priority


def test_to_dict_from_dict(graph):
    """Test graph serialization and deserialization."""
    # Create graph
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1))
    graph.add_node(Node(id="n2", type="functional_consequence", label="regular", creation_turn=1))
    graph.add_edge(
        Edge(
            id="e1", type="leads_to", source="n1", target="n2", creation_turn=1, source_quote="test"
        )
    )

    # Serialize
    data = graph.to_dict()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1

    # Deserialize to new graph
    new_graph = InterviewGraph(graph.schema)
    new_graph.from_dict(data)

    assert new_graph.node_count == 2
    assert new_graph.edge_count == 1
    assert new_graph.get_node("n1") is not None


def test_export_graphml(graph, tmp_path):
    """Test GraphML export."""
    graph.add_node(Node(id="n1", type="attribute", label="price", creation_turn=1))
    graph.add_node(Node(id="n2", type="functional_consequence", label="regular", creation_turn=1))
    graph.add_edge(
        Edge(
            id="e1", type="leads_to", source="n1", target="n2", creation_turn=1, source_quote="test"
        )
    )

    output_path = tmp_path / "test_graph.graphml"
    graph.export_graphml(str(output_path))

    assert output_path.exists()
    content = output_path.read_text()
    assert "graphml" in content.lower()
