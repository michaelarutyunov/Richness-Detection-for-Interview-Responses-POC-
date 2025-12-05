"""
GraphNeedsDetector - Analyzes graph state to identify structural needs.
"""

import logging
from typing import List, Dict, Any
from src.core.models import GraphState, Need, NeedName, InterviewState


logger = logging.getLogger(__name__)


class GraphNeedsDetector:
    """
    Analyzes GraphState and produces prioritized list of structural needs.
    
    The detector identifies opportunities for graph improvement:
    - Bridge isolation: Connect isolated nodes to main graph
    - Depth completion: Deepen shallow branches  
    - Seed expansion: Add more conceptual nodes when graph is too small
    """
    
    def __init__(self, config=None):
        """Initialize the detector with configuration."""
        if config is None:
            # Use sensible defaults when no config provided
            from src.core.models import GraphNeedsConfig
            self.config = GraphNeedsConfig()
        else:
            self.config = config
        logger.info("GraphNeedsDetector initialized with config: %s", self.config)
    
    def detect_productive_needs(self, graph_state: GraphState) -> List[Need]:
        """
        Detect only productive needs that can lead to graph growth.
        
        Args:
            graph_state: Current state of the knowledge graph
            
        Returns:
            List of productive needs (bridge, depth, seed) sorted by priority
        """
        logger.info("Detecting productive needs for turn %s", graph_state.turn_number)
        logger.debug("Graph state: %s nodes, %s edges", 
                    graph_state.get_node_count(), graph_state.get_edge_count())
        
        needs = []
        
        # Detect each type of productive need
        bridge_need = self._detect_bridge_isolation(graph_state)
        if bridge_need and bridge_need.score > 0:
            needs.append(bridge_need)
            logger.debug("Detected bridge isolation need: %s", bridge_need)
        
        depth_need = self._detect_depth_completion(graph_state)
        if depth_need and depth_need.score > 0:
            needs.append(depth_need)
            logger.debug("Detected depth completion need: %s", depth_need)
        
        seed_need = self._detect_seed_expansion(graph_state)
        if seed_need and seed_need.score > 0:
            needs.append(seed_need)
            logger.debug("Detected seed expansion need: %s", seed_need)
        
        # Sort by score (highest first)
        needs.sort(key=lambda x: x.score, reverse=True)
        
        logger.info("Detected %s productive needs: %s", len(needs), [str(need) for need in needs])
        return needs
    
    def calculate_dead_end_score(self, graph_state: GraphState, interview_state: InterviewState) -> float:
        """
        Calculate dead-end score (productivity detector) with false positive protection.
        
        Returns:
            float: Dead-end score (0.0-1.0), higher means more likely stalling
        """
        logger.debug("Calculating dead-end score for turn %s", interview_state.turn_number)
        
        # Protection 1: Recent depth activity gets grace period
        if self._recent_depth_activity(interview_state):
            if interview_state.turn_number <= interview_state.last_depth_turn + 2:
                logger.debug("Depth protection active - giving depth more chances")
                return 0.0  # No dead-end during depth grace period
        
        # Protection 2: Single repetition doesn't trigger dead-end
        repetition_count = self._count_recent_repetitions(interview_state)
        if repetition_count <= 1:
            repetition_factor = 0.0
        else:
            repetition_factor = min(1.0, (repetition_count - 1) / 3)
        
        # Calculate components
        depth_stuck = self._is_depth_stuck(graph_state, interview_state)
        no_new_edges = self._no_recent_graph_growth(graph_state, interview_state)
        low_growth = self._is_graph_growth_stalled(graph_state, interview_state)
        
        # Weighted score (your formula)
        dead_end_score = (
            0.35 * depth_stuck +
            0.30 * no_new_edges +
            0.20 * repetition_factor +
            0.15 * low_growth
        )
        
        logger.debug("Dead-end score components: depth_stuck=%.2f, no_new_edges=%.2f, repetition=%.2f, low_growth=%.2f, total=%.2f",
                    depth_stuck, no_new_edges, repetition_factor, low_growth, dead_end_score)
        
        return dead_end_score
    
    def _recent_depth_activity(self, interview_state: InterviewState) -> bool:
        """Check if recent turns involved depth exploration."""
        # Check last 3 turns for depth tactics
        recent_tactics = self._get_recent_tactics(interview_state, 3)
        depth_tactics = ["value_ladder", "emotional_probe", "causal_value_link"]
        return any(tactic in depth_tactics for tactic in recent_tactics)
    
    def _get_recent_tactics(self, interview_state: InterviewState, count: int) -> List[str]:
        """Get tactic IDs from recent turns."""
        # For now, check if last tactic usage suggests recent depth activity
        # This is a simplified implementation - can be enhanced with proper tactic history
        if interview_state.turn_number == 0:
            return []
        
        # Check if any depth tactics were used recently by looking at last_depth_turn
        if interview_state.last_depth_turn >= interview_state.turn_number - count:
            return ["value_ladder"]  # Assume depth activity if last_depth_turn is recent
        
        return []
    
    def _count_recent_repetitions(self, interview_state: InterviewState) -> int:
        """Count repetition patterns in recent questions."""
        if len(interview_state.question_history) < 3:
            return 0
        
        recent_questions = interview_state.question_history[-3:]
        repetition_count = 0
        
        # Simple repetition detection: same question appears multiple times
        for i, question in enumerate(recent_questions):
            if question in recent_questions[:i]:  # Appeared before in recent list
                repetition_count += 1
        
        return repetition_count
    
    def _is_depth_stuck(self, graph_state: GraphState, interview_state: InterviewState) -> float:
        """Detect if depth exploration is stuck (sustained shallow state)."""
        # Check if average depth has been stuck at low levels for multiple turns
        # This is a simplified implementation - can be enhanced with historical tracking
        avg_depth = graph_state.get_average_depth()
        
        if avg_depth <= 1.5:  # Shallow depth for multiple turns
            return 1.0
        elif avg_depth <= 2.0:
            return 0.5
        else:
            return 0.0
    
    def _no_recent_graph_growth(self, graph_state: GraphState, interview_state: InterviewState) -> float:
        """Detect lack of new edges from recent activity."""
        # Simplified: check if graph has grown in recent turns
        # Enhanced version would track growth rate over time
        node_count = graph_state.get_node_count()
        edge_count = graph_state.get_edge_count()
        
        # If graph is very small or not growing, consider it stalled
        if node_count <= 2 and edge_count <= 1:
            return 1.0
        elif edge_count <= 2:
            return 0.7
        else:
            return 0.0
    
    def _is_graph_growth_stalled(self, graph_state: GraphState, interview_state: InterviewState) -> float:
        """Detect overall graph growth stagnation."""
        # Check growth rate - simplified implementation
        total_nodes = graph_state.get_node_count()
        total_edges = graph_state.get_edge_count()
        
        # Low growth indicators
        if total_nodes <= 3 and total_edges <= 2:
            return 1.0
        elif total_nodes <= 5 and total_edges <= 3:
            return 0.6
        else:
            return 0.0
    
    def _detect_bridge_isolation(self, graph_state: GraphState) -> Need:
        """
        Detect isolated nodes that should be connected to the main graph.
        
        Scoring: number of isolated nodes / total nodes
        """
        isolated_nodes = graph_state.get_isolated_nodes()
        total_nodes = graph_state.get_node_count()
        
        if total_nodes == 0:
            score = 0.0
        else:
            score = len(isolated_nodes) / total_nodes
        
        context = {
            "isolated_nodes": [node.id for node in isolated_nodes],
            "isolated_count": len(isolated_nodes),
            "total_nodes": total_nodes
        }
        
        need = Need(
            name=NeedName.BRIDGE_ISOLATION,
            score=min(score, 1.0),  # Cap at 1.0
            context=context
        )
        
        logger.debug("Bridge isolation detection: %s isolated out of %s nodes (score: %.2f)",
                    len(isolated_nodes), total_nodes, need.score)
        
        return need
    
    def _detect_depth_completion(self, graph_state: GraphState) -> Need:
        """
        Detect shallow branches that need deeper exploration.
        
        Scoring based on average graph depth and shallow node count.
        """
        if graph_state.get_node_count() == 0:
            score = 0.0
            shallow_nodes = []
        else:
            # Calculate average depth
            avg_depth = graph_state.get_average_depth()
            
            # Find shallow nodes (nodes with no children or very few connections)
            shallow_nodes = self._find_shallow_nodes(graph_state)
            
            # Score: combination of low average depth and many shallow nodes
            depth_score = max(0.0, 1.0 - avg_depth / self.config.target_depth)
            shallow_score = len(shallow_nodes) / graph_state.get_node_count()
            
            score = (depth_score + shallow_score) / 2.0
        
        context = {
            "average_depth": graph_state.get_average_depth(),
            "shallow_nodes": [node.id for node in shallow_nodes],
            "shallow_count": len(shallow_nodes),
            "total_nodes": graph_state.get_node_count()
        }
        
        need = Need(
            name=NeedName.DEPTH_COMPLETION,
            score=min(score, 1.0),
            context=context
        )
        
        logger.debug("Depth completion detection: avg_depth=%.2f, shallow_nodes=%s (score: %.2f)",
                    graph_state.get_average_depth(), len(shallow_nodes), need.score)
        
        return need
    
    def _detect_seed_expansion(self, graph_state: GraphState) -> Need:
        """
        Detect when graph is too small and needs more seed concepts.
        
        Scoring: 1.0 if below minimum threshold, decreasing as graph grows.
        """
        node_count = graph_state.get_node_count()
        min_threshold = self.config.min_nodes_for_seed_expansion
        
        if node_count < min_threshold:
            # Higher score when significantly below threshold
            ratio = node_count / min_threshold
            if ratio <= 0.5:  # Less than half threshold
                score = 1.0
            else:
                # Linear decrease from 1.0 to 0.1 as we approach threshold
                score = 1.0 - (ratio - 0.5) * 1.8  # Maps 0.5->1.0, 1.0->0.1
        else:
            score = 0.0
        
        context = {
            "node_count": node_count,
            "min_threshold": min_threshold,
            "ratio": node_count / min_threshold if min_threshold > 0 else 0.0
        }
        
        need = Need(
            name=NeedName.SEED_EXPANSION,
            score=score,
            context=context
        )
        
        logger.debug("Seed expansion detection: %s nodes, threshold=%s (score: %.2f)",
                    node_count, min_threshold, need.score)
        
        return need
    
    def _find_shallow_nodes(self, graph_state: GraphState) -> List:
        """
        Find nodes that are likely shallow (need deeper exploration).
        
        A node is considered shallow if:
        - It has no outgoing edges (leaf node)
        - It has very few connections overall
        - It hasn't been visited much (indicating unexplored potential)
        """
        shallow_nodes = []
        
        for node in graph_state.nodes.values():
            # Count connections (both incoming and outgoing)
            connections = 0
            for edge in graph_state.edges.values():
                if edge.source == node.id or edge.target == node.id:
                    connections += 1
            
            # Consider shallow if:
            # - Leaf node (no outgoing edges) AND
            # - (Few connections OR low visit count indicating unexplored)
            has_outgoing = any(edge.source == node.id for edge in graph_state.edges.values())

            if not has_outgoing and (connections <= 2 or node.visit_count <= 1):
                shallow_nodes.append(node)
        
        return shallow_nodes
    
    def get_need_description(self, need: Need) -> str:
        """Get a human-readable description of a need."""
        if need.name == NeedName.BRIDGE_ISOLATION:
            isolated_count = need.context.get("isolated_count", 0)
            return f"Connect {isolated_count} isolated concept(s) to the main graph"
        
        elif need.name == NeedName.DEPTH_COMPLETION:
            avg_depth = need.context.get("average_depth", 0)
            shallow_count = need.context.get("shallow_count", 0)
            return f"Deepen shallow branches (avg depth: {avg_depth:.1f}, {shallow_count} shallow nodes)"
        
        elif need.name == NeedName.SEED_EXPANSION:
            node_count = need.context.get("node_count", 0)
            return f"Expand conceptual coverage (current: {node_count} nodes)"
        
        return f"Address {need.name.value} (score: {need.score:.2f})"