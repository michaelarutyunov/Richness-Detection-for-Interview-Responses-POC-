"""
Topic exhaustion detection for opportunity ranking.

Identifies over-explored nodes to avoid repetitive questioning.
"""

import logging
from typing import Set

from src.core.interview_graph import InterviewGraph

logger = logging.getLogger(__name__)


class ExhaustionDetector:
    """Detects exhausted topics in interview graph."""

    def __init__(
        self, visit_threshold: int = 3, enable_detection: bool = True
    ):
        """
        Initialize exhaustion detector.

        Args:
            visit_threshold: Visits before marking as exhausted
            enable_detection: Enable/disable detection
        """
        self.visit_threshold = visit_threshold
        self.enable_detection = enable_detection

    def is_exhausted(
        self, node_id: str, graph: InterviewGraph
    ) -> bool:
        """
        Check if node is exhausted using 3-criteria check.

        Criteria (ALL must be true):
        1. visit_count >= threshold
        2. Has been elaborated (out_degree > 0)
        3. All successors also visited

        Extracted from legacy lines 415-455.

        Args:
            node_id: Node to check
            graph: Interview graph

        Returns:
            True if node is exhausted
        """
        if not self.enable_detection:
            return False

        node = graph.get_node(node_id)
        if not node:
            return False

        # Criterion 1: Check visit threshold
        if node.visit_count < self.visit_threshold:
            return False

        # Criterion 2: Must have some elaboration
        out_degree = graph.graph.out_degree(node_id)
        if out_degree == 0:
            # No children yet, not exhausted
            return False

        # Criterion 3: Check if all successors visited
        successors = list(graph.graph.successors(node_id))
        all_visited = all(
            graph.graph.nodes[s]["data"].visit_count > 0 for s in successors
        )

        if all_visited:
            logger.debug(
                f"Topic exhausted: {node.label} "
                f"(visits={node.visit_count}, children={out_degree}, all visited)"
            )
            return True

        return False

    def get_exhausted_nodes(self, graph: InterviewGraph) -> Set[str]:
        """
        Get all exhausted nodes in graph.

        Returns:
            Set of exhausted node IDs
        """
        exhausted = set()

        for node in graph.nodes:
            if self.is_exhausted(node.id, graph):
                exhausted.add(node.id)

        return exhausted

    def mark_visited(
        self, node_id: str, graph: InterviewGraph, turn_number: int
    ):
        """
        Mark node as visited (increment count, update timestamp).

        Note: This is a utility method. In practice, the graph itself
        handles visit tracking via graph.visit_node().

        Args:
            node_id: Node that was visited
            graph: Interview graph
            turn_number: Current turn number
        """
        node = graph.get_node(node_id)
        if not node:
            return

        # Update visit metadata
        node.visit_count = node.visit_count + 1
        node.last_visit_turn = turn_number

        logger.debug(
            f"Node {node_id} visited: "
            f"count={node.visit_count}, "
            f"turn={turn_number}"
        )
