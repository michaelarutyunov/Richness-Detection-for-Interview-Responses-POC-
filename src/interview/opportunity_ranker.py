"""
Opportunity Ranker for intelligent question selection.

Ranks graph nodes for exploration based on multiple criteria.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from src.core.interview_graph import InterviewGraph

logger = logging.getLogger(__name__)


class QuestionStrategy(str, Enum):
    """Question generation strategies."""

    DIG_DEEPER = "dig_deeper"  # Probe existing concept
    CONNECT_CONCEPTS = "connect_concepts"  # Explore relationship
    INTRODUCE_TOPIC = "introduce_topic"  # New concept
    CLOSING = "closing"  # End interview


@dataclass
class RankedOpportunity:
    """Ranked opportunity for exploration."""

    node_id: str
    node_label: str
    node_type: str
    strategy: QuestionStrategy
    priority_score: float
    rationale: str
    metadata: dict


class OpportunityRanker:
    """Ranks graph exploration opportunities."""

    def __init__(
        self,
        graph: InterviewGraph,
        focus_weight: float = 2.0,
        coverage_weight: float = 3.0,
        depth_weight: float = 1.5,
        recency_weight: float = 1.0,
    ):
        """
        Initialize opportunity ranker.

        Args:
            graph: Interview graph to analyze
            focus_weight: Weight for staying on current topic
            coverage_weight: Weight for covering underexplored types
            depth_weight: Weight for exploring shallow branches
            recency_weight: Weight for avoiding recently visited nodes
        """
        self.graph = graph
        self.focus_weight = focus_weight
        self.coverage_weight = coverage_weight
        self.depth_weight = depth_weight
        self.recency_weight = recency_weight
        self._focus_stack = []  # Track recent exploration path

    def rank_opportunities(self, max_opportunities: int = 10) -> list[RankedOpportunity]:
        """
        Rank all exploration opportunities in graph.

        Args:
            max_opportunities: Maximum number of opportunities to return

        Returns:
            List of ranked opportunities sorted by priority (descending)
        """
        if self.graph.node_count == 0:
            return []

        opportunities = []
        coverage_metrics = self.graph.calculate_coverage()

        for node_id in self.graph.graph.nodes():
            node_data = self.graph.graph.nodes[node_id]["data"]

            # Calculate multi-dimensional scores
            coverage_score = self._calculate_coverage_score(node_data.type, coverage_metrics)
            depth_score = self._calculate_depth_score(node_id)
            recency_score = self._calculate_recency_score(node_data.visit_count)
            focus_score = self._calculate_focus_score(node_id)

            # Weighted priority
            priority = (
                (coverage_score * self.coverage_weight)
                + (depth_score * self.depth_weight)
                + (recency_score * self.recency_weight)
                + (focus_score * self.focus_weight)
            )

            # Determine strategy
            strategy = self._determine_strategy(node_id, node_data)

            opportunities.append(
                RankedOpportunity(
                    node_id=node_id,
                    node_label=node_data.label,
                    node_type=node_data.type,
                    strategy=strategy,
                    priority_score=priority,
                    rationale=self._build_rationale(
                        node_data, coverage_score, depth_score, recency_score
                    ),
                    metadata={
                        "visit_count": node_data.visit_count,
                        "out_degree": self.graph.graph.out_degree(node_id),
                        "richness_weight": self.graph.schema.get_richness_weight(node_data.type),
                    },
                )
            )

        # Sort by priority (descending)
        opportunities.sort(key=lambda o: o.priority_score, reverse=True)

        return opportunities[:max_opportunities]

    def _calculate_coverage_score(self, node_type: str, coverage_metrics: dict) -> float:
        """
        Score based on type coverage.

        Favor underexplored node types.
        """
        node_count = coverage_metrics["node_counts"].get(node_type, 0)

        # Lower coverage = higher score
        if node_count == 0:
            return 1.0  # New type is most valuable

        # Inverse of coverage (more nodes of this type = lower score)
        return 1.0 / (node_count + 1)

    def _calculate_depth_score(self, node_id: str) -> float:
        """
        Score based on exploration depth.

        Favor nodes with few outgoing edges (shallow branches).
        """
        out_degree = self.graph.graph.out_degree(node_id)

        # Fewer children = more opportunity
        return 1.0 / (out_degree + 1)

    def _calculate_recency_score(self, visit_count: int) -> float:
        """
        Score based on visit frequency.

        Penalize frequently visited nodes.
        """
        # Lower visit count = higher score
        return 1.0 / (visit_count + 1)

    def _calculate_focus_score(self, node_id: str) -> float:
        """
        Score based on current focus.

        Reward staying near recently explored nodes.
        """
        if not self._focus_stack:
            return 0.5  # Neutral if no focus yet

        # Check if node is neighbor of recent focus
        recent_focus = self._focus_stack[-3:]  # Last 3 nodes

        for focus_node in recent_focus:
            if focus_node not in self.graph.graph:
                continue

            # Check if node is successor or predecessor
            if node_id in self.graph.graph.successors(focus_node):
                return 1.0  # Direct child of recent focus
            if node_id in self.graph.graph.predecessors(focus_node):
                return 0.8  # Parent of recent focus

        return 0.3  # Unrelated to focus

    def _determine_strategy(self, node_id: str, node_data) -> QuestionStrategy:
        """Determine best question strategy for this node."""
        out_degree = self.graph.graph.out_degree(node_id)
        visit_count = node_data.visit_count

        # Never visited: introduce
        if visit_count == 0:
            return QuestionStrategy.INTRODUCE_TOPIC

        # Visited but shallow: dig deeper
        if out_degree < 2:
            return QuestionStrategy.DIG_DEEPER

        # Well explored: connect to other concepts
        return QuestionStrategy.CONNECT_CONCEPTS

    def _build_rationale(
        self, node_data, coverage_score: float, depth_score: float, recency_score: float
    ) -> str:
        """Build human-readable rationale for ranking."""
        reasons = []

        if coverage_score > 0.5:
            reasons.append(f"underexplored type ({node_data.type})")

        if depth_score > 0.5:
            reasons.append("shallow branch")

        if recency_score > 0.5:
            reasons.append("not recently visited")

        if not reasons:
            reasons.append("available for exploration")

        return ", ".join(reasons)

    def update_focus(self, node_id: str):
        """
        Update focus stack after exploring a node.

        Args:
            node_id: Node that was just explored
        """
        self._focus_stack.append(node_id)

        # Keep only recent history
        if len(self._focus_stack) > 5:
            self._focus_stack.pop(0)

    def should_continue(
        self, current_turn: int, min_richness: float = 5.0, max_turns: int = 20
    ) -> bool:
        """
        Determine if interview should continue.

        Args:
            current_turn: Current interview turn number
            min_richness: Minimum richness threshold
            max_turns: Maximum turns before stopping

        Returns:
            bool: True if should continue interviewing
        """
        current_richness = self.graph.calculate_richness()

        # Stop if rich enough
        if current_richness >= min_richness:
            logger.info(f"Richness threshold reached: {current_richness:.2f} >= {min_richness}")
            return False

        # Stop if too long
        if current_turn >= max_turns:
            logger.info(f"Max turns reached: {current_turn} >= {max_turns}")
            return False

        return True

    def get_summary(self) -> dict:
        """Get interview summary statistics."""
        coverage = self.graph.calculate_coverage()
        richness = self.graph.calculate_richness()

        return {
            "nodes": self.graph.node_count,
            "edges": self.graph.edge_count,
            "richness": richness,
            "coverage": coverage["overall"],
            "type_coverage": coverage["by_type"],
            "focus_path": self._focus_stack[-3:] if self._focus_stack else [],
        }
