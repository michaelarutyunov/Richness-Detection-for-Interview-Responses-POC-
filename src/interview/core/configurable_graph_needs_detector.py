"""
Configurable GraphNeedsDetector - Uses interview configuration from YAML.
Replaces hardcoded values with configuration-driven behavior.
"""

import logging
from typing import List, Dict, Any
from src.core.models import GraphState, Need, NeedName, InterviewState
from src.config.interview_config_loader import InterviewConfig

logger = logging.getLogger(__name__)


class ConfigurableGraphNeedsDetector:
    """
    Configurable GraphNeedsDetector that uses interview settings from YAML.
    
    This detector replaces hardcoded values with configuration-driven behavior,
    making graph needs detection truly configurable.
    """
    
    def __init__(self, config: InterviewConfig):
        """Initialize the detector with interview configuration.
        
        Args:
            config: Interview configuration from YAML
        """
        self.config = config
        logger.info("ConfigurableGraphNeedsDetector initialized with interview config")
    
    def detect_productive_needs(self, graph_state: GraphState) -> List[Need]:
        """Detect productive needs using configuration values.
        
        Args:
            graph_state: Current state of the knowledge graph
            
        Returns:
            List of productive needs (bridge, depth, seed) sorted by priority
        """
        logger.info("Detecting productive needs with configuration (turn %s)", graph_state.turn_number)
        logger.debug("Graph state: %s nodes, %s edges", 
                    graph_state.get_node_count(), graph_state.get_edge_count())
        
        needs = []
        
        # Detect each type of productive need using configuration values
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
        
        logger.info("Detected %s productive needs with configuration", len(needs))
        return needs
    
    def _detect_bridge_isolation(self, graph_state: GraphState) -> Optional[Need]:
        """Detect bridge isolation using configuration threshold."""
        isolated_nodes = graph_state.get_isolated_nodes()
        total_nodes = graph_state.get_node_count()
        
        if total_nodes == 0:
            return None
        
        isolation_ratio = len(isolated_nodes) / total_nodes
        score = isolation_ratio
        
        if score >= self.config.graph_needs.isolation_threshold:
            return Need(
                name=NeedName.BRIDGE_ISOLATION,
                score=score,
                context={
                    "isolated_count": len(isolated_nodes),
                    "total_nodes": total_nodes,
                    "isolation_ratio": isolation_ratio,
                    "threshold": self.config.graph_needs.isolation_threshold
                }
            )
        
        return None
    
    def _detect_depth_completion(self, graph_state: GraphState) -> Optional[Need]:
        """Detect depth completion using configuration threshold."""
        average_depth = graph_state.get_average_depth()
        shallow_nodes = len([node for node in graph_state.nodes.values() 
                           if graph_state.get_node_depth(node.id) < self.config.graph_needs.target_depth])
        
        if average_depth <= self.config.graph_needs.depth_completion_threshold:
            score = min(1.0, shallow_nodes / max(1, len(graph_state.nodes)))
            
            return Need(
                name=NeedName.DEPTH_COMPLETION,
                score=score,
                context={
                    "average_depth": average_depth,
                    "shallow_nodes": shallow_nodes,
                    "target_depth": self.config.graph_needs.target_depth,
                    "threshold": self.config.graph_needs.depth_completion_threshold
                }
            )
        
        return None
    
    def _detect_seed_expansion(self, graph_state: GraphState) -> Optional[Need]:
        """Detect seed expansion using configuration threshold."""
        node_count = graph_state.get_node_count()
        
        if node_count < self.config.graph_needs.min_nodes_for_seed_expansion:
            score = 1.0 - (node_count / self.config.graph_needs.min_nodes_for_seed_expansion)
            
            return Need(
                name=NeedName.SEED_EXPANSION,
                score=score,
                context={
                    "node_count": node_count,
                    "min_threshold": self.config.graph_needs.min_nodes_for_seed_expansion,
                    "expansion_needed": self.config.graph_needs.min_nodes_for_seed_expansion - node_count
                }
            )
        
        return None
    
    def calculate_dead_end_score(self, graph_state: GraphState, interview_state: InterviewState) -> float:
        """Calculate dead-end score using configuration weights."""
        logger.debug("Calculating dead-end score with configuration (turn %s)", interview_state.turn_number)
        
        # Protection 1: Recent depth activity gets grace period
        if self._recent_depth_activity(interview_state):
            if interview_state.turn_number <= interview_state.last_depth_turn + self.config.graph_needs.dead_end_probe_count:
                logger.debug("Depth protection active - giving depth more chances (probe_count=%s)", 
                           self.config.graph_needs.dead_end_probe_count)
                return 0.0
        
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
        
        # Weighted score using configuration weights
        strategy_weights = self.config.graph_needs.strategy_weights
        dead_end_score = (
            strategy_weights.get('depth_stuck', 0.35) * depth_stuck +
            strategy_weights.get('no_new_edges', 0.30) * no_new_edges +
            strategy_weights.get('repetition', 0.20) * repetition_factor +
            strategy_weights.get('low_growth', 0.15) * low_growth
        )
        
        logger.debug("Dead-end score components with config: depth_stuck=%.2f, no_new_edges=%.2f, repetition=%.2f, low_growth=%.2f, total=%.2f",
                    depth_stuck, no_new_edges, repetition_factor, low_growth, dead_end_score)
        
        return dead_end_score
    
    def _recent_depth_activity(self, interview_state: InterviewState) -> bool:
        """Check if recent turns involved depth exploration using configured count."""
        recent_tactics = self._get_recent_tactics(interview_state, self.config.graph_needs.recent_tactics_count)
        depth_tactics = ["value_ladder", "emotional_probe", "causal_value_link"]
        return any(tactic in depth_tactics for tactic in recent_tactics)
    
    def _get_recent_tactics(self, interview_state: InterviewState, count: int) -> List[str]:
        """Get tactic IDs from recent turns using configured count."""
        # Simplified implementation - can be enhanced with proper tactic history
        if interview_state.turn_number == 0:
            return []
        
        # For now, check recent question patterns as proxy for tactics
        recent_questions = interview_state.question_history[-count:]
        # This is a simplified proxy - in real implementation, track actual tactic usage
        return ["emotional_probe"] if any("feel" in q.lower() for q in recent_questions) else []
    
    def _count_recent_repetitions(self, interview_state: InterviewState) -> int:
        """Count recent question repetitions using configured count."""
        recent_questions = interview_state.question_history[-self.config.tactic_selection.recent_questions_count:]
        repetition_count = 0
        
        for i, question in enumerate(recent_questions):
            if i > 0 and question == recent_questions[i-1]:
                repetition_count += 1
        
        return repetition_count
    
    def _is_depth_stuck(self, graph_state: GraphState, interview_state: InterviewState) -> float:
        """Check if depth exploration is stuck using target depth from configuration."""
        average_depth = graph_state.get_average_depth()
        if average_depth >= self.config.graph_needs.target_depth:
            return 0.0
        
        # Scale based on how far below target we are
        depth_ratio = average_depth / self.config.graph_needs.target_depth
        return max(0.0, 1.0 - depth_ratio)
    
    def _no_recent_graph_growth(self, graph_state: GraphState, interview_state: InterviewState) -> float:
        """Check if graph growth has stalled recently."""
        # Simplified implementation - can be enhanced with proper growth tracking
        recent_turns = max(3, interview_state.turn_number // 3)
        if interview_state.turn_number <= recent_turns:
            return 0.0
        
        # Check if new nodes/edges were added in recent turns
        # This is a simplified proxy - in real implementation, track actual growth
        return 0.3  # Placeholder implementation
    
    def _is_graph_growth_stalled(self, graph_state: GraphState, interview_state: InterviewState) -> float:
        """Check if overall graph growth is stalled."""
        # Simplified implementation - can be enhanced with proper growth analysis
        return 0.2  # Placeholder implementation - would track growth rate over time