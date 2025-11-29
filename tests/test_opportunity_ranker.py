"""
Tests for OpportunityRanker.
"""

import pytest

from src.core.data_models import Edge, Node
from src.core.interview_graph import InterviewGraph
from src.core.schema_manager import SchemaManager
from src.interview.opportunity_ranker import OpportunityRanker, QuestionStrategy


@pytest.fixture
def schema_manager():
    """Create schema manager."""
    schema = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    schema.load_schema()
    schema.validate_schema()
    return schema


@pytest.fixture
def populated_graph(schema_manager):
    """Create graph with some nodes."""
    graph = InterviewGraph(schema_manager)

    # Add nodes
    nodes = [
        Node(
            id="affordable_price",
            type="attribute",
            label="affordable_price",
            source_quotes=["it's affordable"],
            creation_turn=1,
            visit_count=2,  # Visited twice
        ),
        Node(
            id="regular_purchase",
            type="functional_consequence",
            label="regular_purchase",
            source_quotes=["buy it weekly"],
            creation_turn=1,
            visit_count=0,  # Not visited
        ),
        Node(
            id="peace_of_mind",
            type="psychosocial_consequence",
            label="peace_of_mind",
            source_quotes=["don't worry"],
            creation_turn=2,
            visit_count=1,
        ),
    ]

    for node in nodes:
        graph.add_node(node)

    # Add edges
    edges = [
        Edge(
            id="affordable_price-leads_to-regular_purchase",
            type="leads_to",
            source="affordable_price",
            target="regular_purchase",
            source_quote="affordable so I buy it weekly",
            creation_turn=1,
        )
    ]

    for edge in edges:
        graph.add_edge(edge)

    return graph


class TestOpportunityRanker:
    """Tests for OpportunityRanker."""

    def test_rank_empty_graph(self, schema_manager):
        """Test ranking with empty graph."""
        graph = InterviewGraph(schema_manager)
        ranker = OpportunityRanker(graph)

        opportunities = ranker.rank_opportunities()

        assert len(opportunities) == 0

    def test_rank_opportunities(self, populated_graph):
        """Test basic opportunity ranking."""
        ranker = OpportunityRanker(populated_graph)

        opportunities = ranker.rank_opportunities(max_opportunities=10)

        assert len(opportunities) == 3
        assert all(opp.priority_score > 0 for opp in opportunities)

        # Sorted by priority descending
        for i in range(len(opportunities) - 1):
            assert opportunities[i].priority_score >= opportunities[i + 1].priority_score

    def test_unvisited_nodes_ranked_higher(self, populated_graph):
        """Test that unvisited nodes get higher priority."""
        ranker = OpportunityRanker(populated_graph)

        opportunities = ranker.rank_opportunities()

        # Find regular_purchase (unvisited) and affordable_price (visited 2x)
        unvisited = next(o for o in opportunities if o.node_label == "regular_purchase")
        visited = next(o for o in opportunities if o.node_label == "affordable_price")

        # Unvisited should have higher recency score component
        assert unvisited.priority_score > visited.priority_score

    def test_shallow_branches_ranked_higher(self, populated_graph):
        """Test that nodes with fewer children rank higher."""
        ranker = OpportunityRanker(populated_graph)

        opportunities = ranker.rank_opportunities()

        # regular_purchase has no children (shallow)
        # affordable_price has 1 child
        shallow = next(o for o in opportunities if o.node_label == "regular_purchase")

        assert shallow.metadata["out_degree"] == 0

    def test_strategy_assignment(self, populated_graph):
        """Test that strategies are assigned correctly."""
        ranker = OpportunityRanker(populated_graph)

        opportunities = ranker.rank_opportunities()

        # Check strategies make sense
        for opp in opportunities:
            if opp.metadata["visit_count"] == 0:
                assert opp.strategy == QuestionStrategy.INTRODUCE_TOPIC
            elif opp.metadata["out_degree"] < 2:
                assert opp.strategy in [
                    QuestionStrategy.DIG_DEEPER,
                    QuestionStrategy.INTRODUCE_TOPIC,
                ]

    def test_update_focus(self, populated_graph):
        """Test focus tracking."""
        ranker = OpportunityRanker(populated_graph)

        ranker.update_focus("affordable_price")
        ranker.update_focus("regular_purchase")

        assert len(ranker._focus_stack) == 2
        assert ranker._focus_stack[-1] == "regular_purchase"

        # Focus stack limits size
        for i in range(10):
            ranker.update_focus(f"node_{i}")

        assert len(ranker._focus_stack) <= 5

    def test_should_continue_richness(self, populated_graph):
        """Test continuation based on richness."""
        ranker = OpportunityRanker(populated_graph)

        # Current richness is low
        current_richness = populated_graph.calculate_richness()
        assert current_richness < 5.0

        # Should continue if threshold not met
        assert ranker.should_continue(current_turn=1, min_richness=10.0, max_turns=20)

        # Should stop if threshold met
        assert not ranker.should_continue(current_turn=1, min_richness=1.0, max_turns=20)

    def test_should_continue_max_turns(self, populated_graph):
        """Test continuation based on max turns."""
        ranker = OpportunityRanker(populated_graph)

        # Max turn is 2 (from creation_turn in nodes)
        assert not ranker.should_continue(current_turn=2, min_richness=100.0, max_turns=2)
        assert ranker.should_continue(current_turn=2, min_richness=100.0, max_turns=10)

    def test_get_summary(self, populated_graph):
        """Test summary statistics."""
        ranker = OpportunityRanker(populated_graph)

        ranker.update_focus("affordable_price")
        ranker.update_focus("regular_purchase")

        summary = ranker.get_summary()

        assert summary["nodes"] == 3
        assert summary["edges"] == 1
        assert summary["richness"] > 0
        assert 0 <= summary["coverage"] <= 1.0
        assert len(summary["focus_path"]) == 2
