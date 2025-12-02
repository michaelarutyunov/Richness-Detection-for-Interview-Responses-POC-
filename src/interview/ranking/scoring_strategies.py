"""
Scoring strategies for opportunity ranking.

Implements Strategy pattern for multi-dimensional node scoring.
Each strategy calculates scores independently and provides phase-adaptive weights.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict

import networkx as nx

from src.core.data_models import InterviewPhase
from src.core.interview_graph import InterviewGraph

logger = logging.getLogger(__name__)


class ScoringStrategy(ABC):
    """Abstract base for scoring strategies."""

    def __init__(self, weights: Dict[InterviewPhase, float]):
        """
        Initialize strategy with phase-specific weights.

        Args:
            weights: Mapping of phase to weight value
        """
        self.weights = weights

    @abstractmethod
    def calculate_score(
        self, node_id: str, graph: InterviewGraph, context: Dict
    ) -> float:
        """
        Calculate score for a node (0.0 to 1.0).

        Args:
            node_id: Node to score
            graph: Interview graph
            context: Additional context (turn_number, phase, coverage_metrics, etc.)

        Returns:
            Score value (0.0 to 1.0)
        """
        pass

    def get_weight(self, phase: InterviewPhase) -> float:
        """Get strategy weight for current phase."""
        return self.weights.get(phase, 0.0)


class CoverageScorer(ScoringStrategy):
    """Scores nodes for breadth coverage."""

    def __init__(self):
        # Phase-adaptive weights (extracted from legacy lines 255-291)
        weights = {
            InterviewPhase.COVERAGE: 4.0,
            InterviewPhase.DEPTH: 2.0,
            InterviewPhase.CONNECTION: 1.0,
            InterviewPhase.WRAP_UP: 1.0,
        }
        super().__init__(weights)

    def calculate_score(
        self, node_id: str, graph: InterviewGraph, context: Dict
    ) -> float:
        """
        Score based on node type coverage.

        Favors underexplored node types.
        Extracted from legacy lines 293-306.
        """
        node = graph.get_node(node_id)
        if not node:
            return 0.0

        coverage_metrics = context.get("coverage_metrics")
        if not coverage_metrics:
            coverage_metrics = graph.calculate_coverage()

        node_count = coverage_metrics["node_counts"].get(node.type, 0)

        # Lower coverage = higher score
        if node_count == 0:
            return 1.0  # New type is most valuable

        # Inverse of coverage (more nodes of this type = lower score)
        return 1.0 / (node_count + 1)


class DepthScorer(ScoringStrategy):
    """Scores nodes for depth exploration (shallow branches)."""

    def __init__(self):
        # Phase-adaptive weights
        weights = {
            InterviewPhase.COVERAGE: 1.0,
            InterviewPhase.DEPTH: 2.5,
            InterviewPhase.CONNECTION: 2.0,
            InterviewPhase.WRAP_UP: 1.0,
        }
        super().__init__(weights)

    def calculate_score(
        self, node_id: str, graph: InterviewGraph, context: Dict
    ) -> float:
        """
        Score based on exploration depth.

        Favors nodes with few outgoing edges (shallow branches).
        Extracted from legacy lines 308-317.
        """
        out_degree = graph.graph.out_degree(node_id)

        # Fewer children = more opportunity
        return 1.0 / (out_degree + 1)


class RecencyScorer(ScoringStrategy):
    """Scores nodes based on visit recency."""

    def __init__(
        self,
        decay_function: str = "exponential",
        enable_time_aware_recency: bool = True,
    ):
        # Phase-adaptive weights
        weights = {
            InterviewPhase.COVERAGE: 1.5,
            InterviewPhase.DEPTH: 2.0,
            InterviewPhase.CONNECTION: 1.5,
            InterviewPhase.WRAP_UP: 2.0,
        }
        super().__init__(weights)
        self.decay_function = decay_function
        self.enable_time_aware_recency = enable_time_aware_recency

    def calculate_score(
        self, node_id: str, graph: InterviewGraph, context: Dict
    ) -> float:
        """
        Enhanced recency scoring with exponential decay and time-awareness.

        Penalizes:
        - Recently visited nodes
        - Frequently visited nodes

        Extracted from legacy lines 319-351.
        """
        node = graph.get_node(node_id)
        if not node:
            return 0.0

        visit_count = node.visit_count
        last_visit_turn = node.last_visit_turn
        current_turn = context.get("turn_number", 0)

        # Calculate base score with decay
        if self.decay_function == "exponential":
            # Exponential decay: 1.0, 0.5, 0.25, 0.125, ...
            base_score = 1.0 / (2**visit_count)
        else:
            # Linear decay (original): 1.0, 0.5, 0.33, 0.25, ...
            base_score = 1.0 / (visit_count + 1)

        # Time-aware penalty (lines 341-349)
        if self.enable_time_aware_recency and last_visit_turn is not None:
            turns_since_visit = current_turn - last_visit_turn

            if turns_since_visit <= 2:
                # Very recent visit: strong penalty
                return base_score * 0.5
            elif turns_since_visit > 5:
                # Old visit: allow revisiting
                return min(base_score * 1.5, 1.0)

        return base_score


class FocusScorer(ScoringStrategy):
    """Scores nodes for focused exploration around recent topics."""

    def __init__(self):
        # Phase-adaptive weights
        weights = {
            InterviewPhase.COVERAGE: 1.0,
            InterviewPhase.DEPTH: 1.5,
            InterviewPhase.CONNECTION: 2.5,
            InterviewPhase.WRAP_UP: 1.0,
        }
        super().__init__(weights)

    def calculate_score(
        self, node_id: str, graph: InterviewGraph, context: Dict
    ) -> float:
        """
        Score based on proximity to recently visited nodes.

        Rewards staying near recently explored nodes.
        Extracted from legacy lines 353-375.
        """
        recent_nodes = context.get("recent_nodes", [])
        if not recent_nodes:
            return 0.5  # Neutral if no focus yet

        # Check if node is neighbor of recent focus
        recent_focus = recent_nodes[-3:]  # Last 3 nodes

        for focus_node in recent_focus:
            if focus_node not in graph.graph:
                continue

            # Check if node is successor or predecessor
            if node_id in graph.graph.successors(focus_node):
                return 1.0  # Direct child of recent focus
            if node_id in graph.graph.predecessors(focus_node):
                return 0.8  # Parent of recent focus

        return 0.3  # Unrelated to focus


class DiversityScorer(ScoringStrategy):
    """Scores nodes for diversity (exploring distant concepts)."""

    def __init__(self, diversity_weight: float = 1.0):
        # Phase-adaptive weights
        weights = {
            InterviewPhase.COVERAGE: 0.0,  # Not used in coverage phase
            InterviewPhase.DEPTH: 0.0,  # Not used in depth phase
            InterviewPhase.CONNECTION: 0.0,  # Not used in connection phase
            InterviewPhase.WRAP_UP: 0.0,
        }
        super().__init__(weights)
        self.diversity_weight = diversity_weight

    def calculate_score(
        self, node_id: str, graph: InterviewGraph, context: Dict
    ) -> float:
        """
        Bonus for exploring nodes far from recent focus.

        Encourages topic switching when focus gets stale.
        Extracted from legacy lines 377-413.

        Note: In the legacy implementation, diversity is added as a bonus
        when enable_diversity_bonus=True, not as a weighted dimension.
        """
        recent_nodes = context.get("recent_nodes", [])
        if not recent_nodes:
            return 0.5

        recent_focus = recent_nodes[-3:]

        # Calculate graph distance to recent focus
        min_distance = float("inf")
        for focus_node in recent_focus:
            if focus_node not in graph.graph:
                continue

            try:
                distance = nx.shortest_path_length(
                    graph.graph.to_undirected(), source=node_id, target=focus_node
                )
                min_distance = min(min_distance, distance)
            except nx.NetworkXNoPath:
                # No path = very distant
                distance = 10

        # Normalize: distance 0 = 0.0, distance 3+ = 1.0
        if min_distance == float("inf"):
            return 1.0  # Isolated node, maximum diversity
        else:
            return min(min_distance / 3.0, 1.0)
