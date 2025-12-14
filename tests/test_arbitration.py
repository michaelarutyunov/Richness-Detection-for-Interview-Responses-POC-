"""
Unit tests for the arbitration module.

Tests all scorer implementations and the ArbitrationEngine.
"""

import pytest
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.graph import Graph, Node, Edge
from core.schema import Schema, NodeTypeDefinition
from core.state import (
    GraphState,
    CoverageState,
    CoverageGap,
    ReferenceElement,
    Momentum,
    NodeFocusTracker,
    EdgeFocusTracker
)
from core.history import History, Turn
from decision.strategy import Strategy, FocusTarget
from decision.arbitration import (
    ScoringContext,
    ArbitrationEngine,
    RedundancyScorer,
    KnowledgeCeilingScorer,
    MomentumAlignmentScorer,
    RecencyDiversityScorer,
    VerticalLadderingScorer,
    BranchHealthScorer,
    CoverageQualityScorer,
    SchemaTensionReadinessScorer,
    ReflectionModeScorer,
)


# --- Fixtures ---

@pytest.fixture
def sample_schema():
    """Create a simple test schema with MEC-like node types."""
    return Schema(
        name="test_mec",
        description="Test Means-End Chain schema",
        node_types={
            "attribute": NodeTypeDefinition(
                name="attribute",
                description="Concrete product features",
                is_terminal=False
            ),
            "feature": NodeTypeDefinition(
                name="feature",
                description="Product feature",
                is_terminal=False
            ),
            "functional_consequence": NodeTypeDefinition(
                name="functional_consequence",
                description="Direct outcomes",
                is_terminal=False
            ),
            "psychosocial_consequence": NodeTypeDefinition(
                name="psychosocial_consequence",
                description="Personal/social meanings",
                is_terminal=False
            ),
            "value": NodeTypeDefinition(
                name="value",
                description="End goals and core values",
                is_terminal=True  # Terminal type
            ),
        },
        edge_types={}
    )


@pytest.fixture
def sample_graph():
    """Create a simple test graph."""
    graph = Graph()

    # Add some nodes
    graph.add_node(Node(
        id="node1",
        label="wateriness",
        node_type="attribute",
        timestamp=datetime.now()
    ))
    graph.add_node(Node(
        id="node2",
        label="weak foam",
        node_type="attribute",
        timestamp=datetime.now()
    ))
    graph.add_node(Node(
        id="node3",
        label="enzymes",
        node_type="feature",
        timestamp=datetime.now()
    ))

    # Add an edge
    graph.add_edge(Edge(
        id="edge1",
        source_id="node1",
        target_id="node2",
        relation_type="causes"
    ))

    return graph


@pytest.fixture
def sample_history():
    """Create sample conversation history."""
    history = History()

    # Add a few turns
    history.add_turn(Turn(
        turn_number=1,
        question="What do you think about the product?",
        response="I noticed the foam is quite weak, probably because of the wateriness.",
        extracted_nodes=["node1", "node2"],
        extracted_edges=[("node1", "node2")],
        strategy_used="ensure_coverage",
        timestamp=datetime.now()
    ))

    history.add_turn(Turn(
        turn_number=2,
        question="How does wateriness lead to weak foam?",
        response="Well, the watery consistency doesn't support foam structure.",
        extracted_nodes=[],
        extracted_edges=[],
        strategy_used="resolve_schema_tension",
        timestamp=datetime.now()
    ))

    history.add_turn(Turn(
        turn_number=3,
        question="What about the enzymes?",
        response="I don't know much about enzymes honestly.",
        extracted_nodes=["node3"],
        extracted_edges=[],
        strategy_used="ensure_coverage",
        timestamp=datetime.now()
    ))

    return history


@pytest.fixture
def sample_graph_state(sample_graph):
    """Create sample graph state."""
    return GraphState(
        isolated_nodes=[sample_graph.get_node("node3")],
        ambiguous_nodes=[],
        invalid_edges=[],
        active_branch=[sample_graph.get_node("node1"), sample_graph.get_node("node2")],
        branch_depth=2,
        unexplored_nodes=[],
        terminal_nodes=[],
        total_nodes=3,
        total_edges=1,
        isolation_ratio=0.33
    )


@pytest.fixture
def sample_coverage_state():
    """Create sample coverage state."""
    state = CoverageState()
    state.reference_elements = {
        "wateriness": ReferenceElement(id="wateriness", content="The product is watery"),
        "enzymes": ReferenceElement(id="enzymes", content="Contains active enzymes"),
    }
    state.element_node_mappings = {"wateriness": ["node1"], "enzymes": []}
    state.element_reactions = {"wateriness": "negative", "enzymes": None}
    state.element_focus_counts = {"wateriness": 2, "enzymes": 1}
    state.gaps = [CoverageGap(element_id="enzymes", gap_type="unmentioned")]
    return state


@pytest.fixture
def sample_momentum():
    """Create sample momentum."""
    return Momentum(level="neutral", indicators=[])


@pytest.fixture
def sample_context(sample_graph, sample_graph_state, sample_coverage_state, sample_momentum, sample_history, sample_schema):
    """Create full scoring context."""
    return ScoringContext(
        graph=sample_graph,
        graph_state=sample_graph_state,
        coverage_state=sample_coverage_state,
        momentum=sample_momentum,
        history=sample_history,
        recent_questions=sample_history.get_recent_questions(6),
        schema=sample_schema
    )


@pytest.fixture
def sample_strategy():
    """Create sample strategy."""
    return Strategy(
        id="resolve_schema_tension",
        intent="Explore connection between concepts",
        applies_when="Invalid edge exists",
        suggested_tactics=["relationship_probe"]
    )


@pytest.fixture
def sample_focus(sample_graph):
    """Create sample focus target."""
    return FocusTarget.from_node(sample_graph.get_node("node1"))


# --- RedundancyScorer Tests ---

class TestRedundancyScorer:

    def test_no_penalty_for_unique_question(self, sample_context, sample_strategy, sample_focus):
        """Should return 1.0 for questions that don't match recent ones."""
        scorer = RedundancyScorer(threshold=0.85)

        # Focus on node3 (enzymes) - hasn't been asked about similarly
        focus = FocusTarget.from_node(sample_context.graph.get_node("node3"))

        score = scorer.score(sample_strategy, focus, sample_context)
        assert score == 1.0

    def test_penalty_for_similar_question(self, sample_context, sample_strategy):
        """Should penalize when question would be similar to recent."""
        scorer = RedundancyScorer(threshold=0.5)  # Lower threshold for test

        # Create focus that would generate similar question
        node1 = sample_context.graph.get_node("node1")
        node2 = sample_context.graph.get_node("node2")
        focus = FocusTarget.from_node_pair(node1, node2)

        # The recent question is "How does wateriness lead to weak foam?"
        # Focus on same pair should trigger similarity
        score = scorer.score(sample_strategy, focus, sample_context)

        # Should be penalized
        assert score < 1.0

    def test_jaccard_similarity_calculation(self):
        """Test the Jaccard similarity calculation."""
        scorer = RedundancyScorer()

        # Identical questions
        sim = scorer._jaccard_similarity("how does this work", "how does this work")
        assert sim == 1.0

        # Completely different
        sim = scorer._jaccard_similarity("apples oranges", "bananas grapes")
        assert sim == 0.0

        # Partial overlap
        sim = scorer._jaccard_similarity("how does wateriness lead to foam",
                                         "how does wateriness affect foam")
        assert 0.3 < sim < 0.9  # Should have some overlap


# --- KnowledgeCeilingScorer Tests ---

class TestKnowledgeCeilingScorer:

    def test_penalty_for_knowledge_ceiling(self, sample_context, sample_strategy):
        """Should penalize focus on topic where user said 'don't know'."""
        scorer = KnowledgeCeilingScorer(lookback=3)

        # Focus on enzymes - user said "I don't know much about enzymes"
        focus = FocusTarget.from_node(sample_context.graph.get_node("node3"))

        score = scorer.score(sample_strategy, focus, sample_context)
        assert score == 0.1  # Heavy penalty

    def test_no_penalty_without_ceiling(self, sample_context, sample_strategy):
        """Should not penalize focus on topic without knowledge ceiling signals."""
        scorer = KnowledgeCeilingScorer(lookback=3)

        # Focus on wateriness - user spoke about it confidently
        focus = FocusTarget.from_node(sample_context.graph.get_node("node1"))

        score = scorer.score(sample_strategy, focus, sample_context)
        assert score == 1.0

    def test_detects_ceiling_patterns(self):
        """Test detection of various knowledge ceiling patterns."""
        scorer = KnowledgeCeilingScorer()

        patterns = [
            "i don't know",
            "not sure",
            "i guess",
            "never thought about",
            "couldn't say"
        ]

        for pattern in patterns:
            assert any(p in pattern for p in scorer.ceiling_patterns)


# --- MomentumAlignmentScorer Tests ---

class TestMomentumAlignmentScorer:

    def test_boost_breadth_on_low_momentum(self, sample_context, sample_focus):
        """Should boost breadth strategies when momentum is low."""
        scorer = MomentumAlignmentScorer()

        sample_context.momentum.level = "low"

        breadth_strategy = Strategy(
            id="explore_breadth",
            intent="Shift to new area",
            applies_when="Branch stalling"
        )

        score = scorer.score(breadth_strategy, sample_focus, sample_context)
        assert score == 1.5  # Boost

    def test_penalize_depth_on_low_momentum(self, sample_context, sample_focus):
        """Should penalize depth strategies when momentum is low."""
        scorer = MomentumAlignmentScorer()

        sample_context.momentum.level = "low"

        depth_strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(depth_strategy, sample_focus, sample_context)
        assert score == 0.5  # Penalty

    def test_boost_depth_on_high_momentum(self, sample_context, sample_focus):
        """Should boost depth strategies when momentum is high."""
        scorer = MomentumAlignmentScorer()

        sample_context.momentum.level = "high"

        depth_strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(depth_strategy, sample_focus, sample_context)
        assert score == 1.3  # Boost

    def test_neutral_on_neutral_momentum(self, sample_context, sample_focus):
        """Should not affect scoring when momentum is neutral."""
        scorer = MomentumAlignmentScorer()

        sample_context.momentum.level = "neutral"

        strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 1.0  # Neutral


# --- RecencyDiversityScorer Tests ---

class TestRecencyDiversityScorer:

    def test_penalty_for_recently_used(self, sample_context, sample_focus):
        """Should penalize recently used strategies."""
        scorer = RecencyDiversityScorer(lookback=2, penalty=0.7)

        # ensure_coverage was used in turn 3
        strategy = Strategy(
            id="ensure_coverage",
            intent="Cover reference elements",
            applies_when="Gaps exist"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 0.7  # One recent use

    def test_cumulative_penalty(self, sample_context, sample_focus):
        """Should apply cumulative penalty for multiple recent uses."""
        scorer = RecencyDiversityScorer(lookback=3, penalty=0.7)

        # ensure_coverage was used in turns 1 and 3
        strategy = Strategy(
            id="ensure_coverage",
            intent="Cover reference elements",
            applies_when="Gaps exist"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == pytest.approx(0.49, rel=0.01)  # 0.7 ^ 2

    def test_no_penalty_for_unused(self, sample_context, sample_focus):
        """Should not penalize strategies not recently used."""
        scorer = RecencyDiversityScorer(lookback=2)

        # introduce_seed wasn't used
        strategy = Strategy(
            id="introduce_seed",
            intent="Open new territory",
            applies_when="All exhausted"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 1.0


# --- VerticalLadderingScorer Tests ---

class TestVerticalLadderingScorer:

    def test_boost_when_horizontally_saturated(self, sample_context, sample_focus):
        """Should boost vertical strategies when graph is horizontally saturated."""
        scorer = VerticalLadderingScorer()

        # Add more concrete nodes to simulate horizontal saturation
        for i in range(10):
            sample_context.graph.add_node(Node(
                id=f"concrete_{i}",
                label=f"feature_{i}",
                node_type="feature",
                timestamp=datetime.now()
            ))

        strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists",
            suggested_tactics=["upward_linking"]
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 1.5  # Boost due to horizontal saturation

    def test_no_boost_when_balanced(self, sample_context, sample_focus):
        """Should not boost when graph has good abstract/concrete balance."""
        scorer = VerticalLadderingScorer()

        # Add abstract nodes for balance
        for i in range(5):
            sample_context.graph.add_node(Node(
                id=f"value_{i}",
                label=f"meaning_{i}",
                node_type="value",
                timestamp=datetime.now()
            ))

        strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists",
            suggested_tactics=["upward_linking"]
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 1.0  # No boost needed


# --- BranchHealthScorer Tests ---

class TestBranchHealthScorer:

    def test_boost_breadth_on_stale_branch(self, sample_context, sample_focus):
        """Should boost explore_breadth when branch is stale."""
        scorer = BranchHealthScorer(stale_threshold=2, breadth_boost=1.8)

        # History shows no growth on active branch in last 2 turns
        # (turns 2 and 3 didn't add edges to node1 or node2)

        strategy = Strategy(
            id="explore_breadth",
            intent="Shift to new area",
            applies_when="Branch stalling"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 1.8  # Boost (updated default)

    def test_no_penalty_on_healthy_branch(self, sample_context, sample_focus):
        """Should not penalize when branch has recent growth."""
        scorer = BranchHealthScorer(stale_threshold=5)

        # Add recent edges to active branch
        sample_context.history.turns[-1].extracted_edges.append(("node1", "node2"))

        strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 1.0


# --- CoverageQualityScorer Tests ---

class TestCoverageQualityScorer:

    def test_penalty_for_knowledge_lack(self, sample_context, sample_focus):
        """Should penalize ensure_coverage for topics user doesn't know."""
        scorer = CoverageQualityScorer(penalty=0.4)

        strategy = Strategy(
            id="ensure_coverage",
            intent="Cover reference elements",
            applies_when="Gaps exist"
        )

        # Focus on enzymes gap - user said "don't know"
        focus = FocusTarget.from_coverage_gap(
            CoverageGap(element_id="enzymes", gap_type="unmentioned"),
            ReferenceElement(id="enzymes", content="Contains enzymes")
        )

        score = scorer.score(strategy, focus, sample_context)
        assert score == 0.4  # Penalty for knowledge lack

    def test_boost_for_unexplored(self, sample_context, sample_focus):
        """Should boost ensure_coverage for never-explored elements (first_touch_boost)."""
        scorer = CoverageQualityScorer(first_touch_boost=2.5)

        strategy = Strategy(
            id="ensure_coverage",
            intent="Cover reference elements",
            applies_when="Gaps exist"
        )

        # Add a new element that hasn't been explored
        sample_context.coverage_state.reference_elements["new_element"] = ReferenceElement(
            id="new_element",
            content="New element"
        )
        sample_context.coverage_state.element_focus_counts["new_element"] = 0  # Never explored
        sample_context.coverage_state.gaps.append(
            CoverageGap(element_id="new_element", gap_type="unmentioned")
        )

        focus = FocusTarget.from_coverage_gap(
            CoverageGap(element_id="new_element", gap_type="unmentioned"),
            ReferenceElement(id="new_element", content="New element")
        )

        score = scorer.score(strategy, focus, sample_context)
        assert score == 2.5  # Strong first_touch_boost for unexplored (updated behavior)

    def test_only_affects_ensure_coverage(self, sample_context, sample_focus):
        """Should only affect ensure_coverage strategy."""
        scorer = CoverageQualityScorer()

        other_strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(other_strategy, sample_focus, sample_context)
        assert score == 1.0


# --- ArbitrationEngine Tests ---

class TestArbitrationEngine:

    def test_selects_highest_utility(self, sample_context):
        """Should select strategy with highest utility score."""
        # Create engine with custom scorers
        scorers = [
            MomentumAlignmentScorer()
        ]
        engine = ArbitrationEngine(scorers)

        # Set low momentum
        sample_context.momentum.level = "low"

        # Create candidates
        breadth = Strategy(id="explore_breadth", intent="Shift", applies_when="stalling")
        depth = Strategy(id="deepen_branch", intent="Continue", applies_when="branch")

        candidates = [
            (depth, FocusTarget()),
            (breadth, FocusTarget())
        ]

        score, winner, focus = engine.select_best(candidates, sample_context)

        # Should select breadth due to low momentum boost
        assert winner.id == "explore_breadth"

    def test_create_default(self):
        """Should create engine with all default scorers."""
        engine = ArbitrationEngine.create_default()

        assert len(engine.scorers) == 9  # Updated from 7 (added 2 new scorers)
        scorer_names = [s.name for s in engine.scorers]
        assert "redundancy" in scorer_names
        assert "knowledge_ceiling" in scorer_names
        assert "momentum_alignment" in scorer_names
        assert "recency_diversity" in scorer_names
        assert "vertical_laddering" in scorer_names
        assert "branch_health" in scorer_names
        assert "coverage_quality" in scorer_names
        assert "schema_tension_readiness" in scorer_names  # New
        assert "reflection_mode" in scorer_names  # New

    def test_from_config(self):
        """Should create engine from config dict."""
        config = {
            "scorers": {
                "redundancy": {"weight": 1.0, "threshold": 0.9},
                "momentum_alignment": {"weight": 0.5}
            }
        }

        engine = ArbitrationEngine.from_config(config)

        assert len(engine.scorers) == 2
        assert engine.weights.get("redundancy") == 1.0
        assert engine.weights.get("momentum_alignment") == 0.5


# --- Integration Tests ---

class TestArbitrationIntegration:

    def test_full_scoring_pipeline(self, sample_context):
        """Test full scoring pipeline with all scorers."""
        engine = ArbitrationEngine.create_default()

        # Create realistic candidates
        strategies = [
            Strategy(id="ensure_coverage", intent="Cover elements", applies_when="gaps"),
            Strategy(id="resolve_schema_tension", intent="Explore connections", applies_when="invalid"),
            Strategy(id="explore_breadth", intent="Shift focus", applies_when="stalling"),
        ]

        candidates = [(s, FocusTarget()) for s in strategies]

        score, winner, focus = engine.select_best(candidates, sample_context)

        # Should complete without errors and return valid result
        assert winner is not None
        assert 0.0 <= score <= 100.0  # Reasonable score range

    def test_scoring_context_build(self, sample_graph, sample_graph_state,
                                   sample_coverage_state, sample_momentum, sample_history):
        """Test ScoringContext.build() factory method."""
        context = ScoringContext.build(
            graph=sample_graph,
            graph_state=sample_graph_state,
            coverage_state=sample_coverage_state,
            momentum=sample_momentum,
            history=sample_history
        )

        assert context.graph is sample_graph
        assert len(context.recent_questions) == 3  # 3 turns in history


# --- New Scorer Tests (Coverage Priority, Branch Saturation, Value Ladder) ---

class TestCoverageQualityScorerEnhancements:
    """Tests for CoverageQualityScorer new behaviors: first_touch_boost and exhaustion detection."""

    def test_first_touch_boost(self, sample_context, sample_focus):
        """Should give strong boost when focus_count == 0 (first-time coverage)."""
        scorer = CoverageQualityScorer(first_touch_boost=2.5)

        strategy = Strategy(
            id="ensure_coverage",
            intent="Cover reference elements",
            applies_when="Gaps exist"
        )

        # Add a new element with focus_count = 0
        sample_context.coverage_state.reference_elements["rtb"] = ReferenceElement(
            id="rtb",
            content="Reason to believe"
        )
        sample_context.coverage_state.element_focus_counts["rtb"] = 0  # Never explored
        sample_context.coverage_state.gaps.append(
            CoverageGap(element_id="rtb", gap_type="unmentioned")
        )

        focus = FocusTarget.from_coverage_gap(
            CoverageGap(element_id="rtb", gap_type="unmentioned"),
            ReferenceElement(id="rtb", content="Reason to believe")
        )

        score = scorer.score(strategy, focus, sample_context)
        assert score == 2.5  # Strong first-touch boost

    def test_exhaustion_penalty(self, sample_context, sample_focus):
        """Should penalize when probed multiple times without new edges."""
        scorer = CoverageQualityScorer(
            exhaustion_threshold=2,
            exhaustion_penalty=0.15
        )

        strategy = Strategy(
            id="ensure_coverage",
            intent="Cover reference elements",
            applies_when="Gaps exist"
        )

        # Simulate element probed multiple times without producing edges
        sample_context.coverage_state.element_focus_counts["enzymes"] = 3  # Above threshold
        # History shows no edges produced for this element

        focus = FocusTarget.from_coverage_gap(
            CoverageGap(element_id="enzymes", gap_type="no_reaction"),
            ReferenceElement(id="enzymes", content="Contains enzymes")
        )

        score = scorer.score(strategy, focus, sample_context)
        # Should get exhaustion penalty since probed 3 times with no edges
        assert score == 0.15


class TestBranchHealthScorerEnhancements:
    """Tests for BranchHealthScorer new behaviors: severe penalties and connect_isolate handling."""

    def test_severe_penalty_for_very_stale_branch(self, sample_context, sample_focus):
        """Should apply severe penalty when branch very stale (turns >= severe_threshold)."""
        scorer = BranchHealthScorer(
            stale_threshold=2,
            severe_stale_threshold=4,
            severe_depth_penalty=0.1
        )

        # Add more turns with no growth on active branch
        for i in range(3):
            sample_context.history.add_turn(Turn(
                turn_number=4 + i,
                question=f"Question {4 + i}",
                response=f"Response {4 + i}",
                extracted_nodes=[],
                extracted_edges=[],  # No edges to branch nodes
                strategy_used="deepen_branch",
                timestamp=datetime.now()
            ))

        strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 0.1  # Severe penalty

    def test_connect_isolate_penalty_on_stale_branch(self, sample_context, sample_focus):
        """Should penalize connect_isolate on stale branches."""
        scorer = BranchHealthScorer(
            stale_threshold=2,
            connect_isolate_penalty=0.5
        )

        strategy = Strategy(
            id="connect_isolate",
            intent="Integrate orphan concept",
            applies_when="Isolated node exists"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 0.5  # Penalty for connect_isolate on stale branch

    def test_introduce_seed_boost_when_very_stale(self, sample_context, sample_focus):
        """Should boost introduce_seed when branch is very stale."""
        scorer = BranchHealthScorer(
            stale_threshold=2,
            severe_stale_threshold=4,
            breadth_boost=1.8
        )

        # Make branch very stale
        for i in range(4):
            sample_context.history.add_turn(Turn(
                turn_number=4 + i,
                question=f"Question {4 + i}",
                response=f"Response {4 + i}",
                extracted_nodes=[],
                extracted_edges=[],
                strategy_used="deepen_branch",
                timestamp=datetime.now()
            ))

        strategy = Strategy(
            id="introduce_seed",
            intent="Open new territory",
            applies_when="All exhausted"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 1.8  # Boost for introduce_seed when very stale


class TestVerticalLadderingScorerEnhancements:
    """Tests for VerticalLadderingScorer new behaviors: value proximity detection."""

    def test_value_closure_boost_for_terminal_type(self, sample_context):
        """Should boost deepen_branch when focus node is a terminal type (schema-agnostic)."""
        scorer = VerticalLadderingScorer(terminal_closure_boost=2.0)

        # Add a terminal-type node (value is terminal in MEC schema)
        value_node = Node(
            id="value_node",
            label="feeling accomplished",
            node_type="value",  # Terminal type
            timestamp=datetime.now()
        )
        sample_context.graph.add_node(value_node)

        strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        focus = FocusTarget.from_node(value_node)

        score = scorer.score(strategy, focus, sample_context)
        assert score == 2.0  # Terminal closure boost

    def test_value_proximity_boost_near_value_node(self, sample_context):
        """Should boost when focus node is within N steps of value node."""
        scorer = VerticalLadderingScorer(
            terminal_proximity_boost=1.8,
            near_terminal_depth=2
        )

        # Create a chain: concrete -> intermediate -> value
        concrete = Node(
            id="concrete",
            label="thick foam",
            node_type="attribute",
            timestamp=datetime.now()
        )
        intermediate = Node(
            id="intermediate",
            label="pleasant sensation",
            node_type="functional_consequence",
            timestamp=datetime.now()
        )
        value = Node(
            id="value",
            label="feeling good",
            node_type="value",
            timestamp=datetime.now()
        )

        sample_context.graph.add_node(concrete)
        sample_context.graph.add_node(intermediate)
        sample_context.graph.add_node(value)
        sample_context.graph.add_edge(Edge(
            id="e1", source_id="concrete", target_id="intermediate", relation_type="leads_to"
        ))
        sample_context.graph.add_edge(Edge(
            id="e2", source_id="intermediate", target_id="value", relation_type="leads_to"
        ))

        strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        # Focus on concrete node (2 steps from value)
        focus = FocusTarget.from_node(concrete)

        score = scorer.score(strategy, focus, sample_context)
        assert score == 1.8  # Proximity boost


class TestSchemaTensionReadinessScorer:
    """Tests for SchemaTensionReadinessScorer (new scorer)."""

    def test_boost_when_both_nodes_explored(self, sample_context):
        """Should boost resolve_schema_tension when both nodes have been explored."""
        scorer = SchemaTensionReadinessScorer(readiness_boost=1.6)

        node1 = sample_context.graph.get_node("node1")
        node2 = sample_context.graph.get_node("node2")

        # Add turns that explored both nodes
        sample_context.history.add_turn(Turn(
            turn_number=4,
            question="What about wateriness?",
            response="It affects the texture",
            extracted_nodes=[], extracted_edges=[],
            strategy_used="deepen_branch",
            timestamp=datetime.now(),
            metadata={"focus_node_id": "node1"}
        ))
        sample_context.history.add_turn(Turn(
            turn_number=5,
            question="What about weak foam?",
            response="It dissolves quickly",
            extracted_nodes=[], extracted_edges=[],
            strategy_used="deepen_branch",
            timestamp=datetime.now(),
            metadata={"focus_node_id": "node2"}
        ))

        strategy = Strategy(
            id="resolve_schema_tension",
            intent="Explore connection",
            applies_when="Invalid edge exists"
        )

        focus = FocusTarget.from_node_pair(node1, node2)

        score = scorer.score(strategy, focus, sample_context)
        assert score == 1.6  # Readiness boost

    def test_penalty_for_premature_exploration(self, sample_context):
        """Should penalize when one or both nodes haven't been explored."""
        scorer = SchemaTensionReadinessScorer(premature_penalty=0.4)

        node1 = sample_context.graph.get_node("node1")
        node2 = sample_context.graph.get_node("node2")

        # No turns with focus_node_id metadata for these nodes

        strategy = Strategy(
            id="resolve_schema_tension",
            intent="Explore connection",
            applies_when="Invalid edge exists"
        )

        focus = FocusTarget.from_node_pair(node1, node2)

        score = scorer.score(strategy, focus, sample_context)
        assert score == 0.4  # Premature penalty

    def test_only_affects_resolve_schema_tension(self, sample_context, sample_focus):
        """Should only affect resolve_schema_tension strategy."""
        scorer = SchemaTensionReadinessScorer()

        other_strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(other_strategy, sample_focus, sample_context)
        assert score == 1.0  # Neutral for other strategies


class TestReflectionModeScorer:
    """Tests for ReflectionModeScorer (new scorer)."""

    def test_triggers_reflection_mode(self, sample_context, sample_focus):
        """Should trigger reflection mode when all conditions met."""
        scorer = ReflectionModeScorer(
            no_new_nodes_threshold=3,
            depth_penalty=0.2,
            reflection_boost=2.0
        )

        # Setup conditions for reflection mode:
        # 1. No coverage gaps
        sample_context.coverage_state.gaps = []

        # 2. No new nodes in recent turns
        for i in range(3):
            sample_context.history.add_turn(Turn(
                turn_number=4 + i,
                question=f"Question {4 + i}",
                response=f"Response {4 + i}",
                extracted_nodes=[],  # No new nodes
                extracted_edges=[],
                strategy_used="explore_breadth",
                timestamp=datetime.now()
            ))

        # 3. Value nodes exist
        sample_context.graph.add_node(Node(
            id="value1",
            label="feeling good",
            node_type="value",
            timestamp=datetime.now()
        ))

        # Test depth strategy gets penalty
        depth_strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(depth_strategy, sample_focus, sample_context)
        assert score == 0.2  # Heavy depth penalty

        # Test reflection strategy gets boost
        reflection_strategy = Strategy(
            id="introduce_seed",
            intent="Open new territory",
            applies_when="All exhausted"
        )

        score = scorer.score(reflection_strategy, sample_focus, sample_context)
        assert score == 2.0  # Reflection boost

    def test_no_trigger_with_coverage_gaps(self, sample_context, sample_focus):
        """Should not trigger reflection mode when coverage gaps exist."""
        scorer = ReflectionModeScorer()

        # Coverage gaps exist (default fixture has gaps)
        assert len(sample_context.coverage_state.gaps) > 0

        strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 1.0  # Neutral (no reflection mode)

    def test_no_trigger_with_new_nodes(self, sample_context, sample_focus):
        """Should not trigger when nodes were extracted recently."""
        scorer = ReflectionModeScorer(no_new_nodes_threshold=3)

        # Clear coverage gaps
        sample_context.coverage_state.gaps = []

        # Add value node
        sample_context.graph.add_node(Node(
            id="value1",
            label="feeling good",
            node_type="value",
            timestamp=datetime.now()
        ))

        # Recent turn has extracted nodes
        sample_context.history.add_turn(Turn(
            turn_number=4,
            question="What does that mean?",
            response="It means feeling accomplished",
            extracted_nodes=["new_node"],  # Has new nodes
            extracted_edges=[],
            strategy_used="deepen_branch",
            timestamp=datetime.now()
        ))

        strategy = Strategy(
            id="deepen_branch",
            intent="Continue exploration",
            applies_when="Active branch exists"
        )

        score = scorer.score(strategy, sample_focus, sample_context)
        assert score == 1.0  # Neutral (no reflection mode)


# --- Updated ArbitrationEngine Tests ---

class TestArbitrationEngineUpdated:
    """Updated tests for ArbitrationEngine with new scorers."""

    def test_create_default_includes_new_scorers(self):
        """Should create engine with all 9 scorers including new ones."""
        engine = ArbitrationEngine.create_default()

        assert len(engine.scorers) == 9  # Updated from 7
        scorer_names = [s.name for s in engine.scorers]
        assert "redundancy" in scorer_names
        assert "knowledge_ceiling" in scorer_names
        assert "momentum_alignment" in scorer_names
        assert "recency_diversity" in scorer_names
        assert "vertical_laddering" in scorer_names
        assert "branch_health" in scorer_names
        assert "coverage_quality" in scorer_names
        assert "schema_tension_readiness" in scorer_names  # New
        assert "reflection_mode" in scorer_names  # New

    def test_from_config_with_new_scorers(self):
        """Should create engine from config with new scorers."""
        config = {
            "scorers": {
                "schema_tension_readiness": {
                    "weight": 1.0,
                    "readiness_boost": 1.6,
                    "premature_penalty": 0.4
                },
                "reflection_mode": {
                    "weight": 1.0,
                    "no_new_nodes_threshold": 3,
                    "depth_penalty": 0.2
                }
            }
        }

        engine = ArbitrationEngine.from_config(config)

        assert len(engine.scorers) == 2
        scorer_names = [s.name for s in engine.scorers]
        assert "schema_tension_readiness" in scorer_names
        assert "reflection_mode" in scorer_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
