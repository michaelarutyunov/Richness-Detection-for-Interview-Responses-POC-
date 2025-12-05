"""
StrategySelector - Maps detected needs to appropriate strategies.
"""

import logging
from typing import List, Optional
from src.core.models import Need, StrategyName, NeedName, GraphState, InterviewState


logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Maps the top detected need to a strategic approach.
    
    The selector takes the highest-priority need and determines which
    strategy should be used to address it. For MVP, this is a simple
    hardcoded mapping.
    """
    
    # Hardcoded mapping for MVP - can be made configurable later
    NEED_TO_STRATEGY_MAP = {
        NeedName.BRIDGE_ISOLATION: StrategyName.BRIDGE_BUILDING,
        NeedName.DEPTH_COMPLETION: StrategyName.DEPTH_COMPLETION,
        NeedName.SEED_EXPANSION: StrategyName.SEED_EXPANSION,
    }
    
    def __init__(self, dead_end_threshold: float = 0.6):
        """
        Initialize the strategy selector with configuration.

        Args:
            dead_end_threshold: Score threshold above which dead-end is detected (default: 0.6)
        """
        self.dead_end_threshold = dead_end_threshold
        logger.info("StrategySelector initialized with dead_end_threshold=%.2f", dead_end_threshold)
        logger.debug("Need to strategy mapping: %s", self.NEED_TO_STRATEGY_MAP)
    
    def select(self, needs: List[Need], graph_state: GraphState, interview_state: InterviewState) -> StrategyName:
        """
        Select strategy with dead-end fallback protection.
        
        Args:
            needs: List of productive needs (bridge, depth, seed only)
            graph_state: Current graph state for dead-end calculation
            interview_state: Current interview state for tracking
            
        Returns:
            Selected strategy name
        """
        logger.info("Selecting strategy with dead-end fallback protection")
        
        # Step 1: Try productive strategies first
        if needs:
            sorted_needs = sorted(needs, key=lambda x: x.score, reverse=True)
            top_need = sorted_needs[0]
            strategy = self.NEED_TO_STRATEGY_MAP.get(top_need.name)
            
            if strategy is None:
                logger.error("No strategy mapping found for need: %s", top_need.name)
                raise ValueError(f"No strategy mapping found for need: {top_need.name}")
            
            logger.info("Selected productive strategy: %s for need: %s (score: %.2f)", 
                       strategy.value, top_need.name.value, top_need.score)
            
            self._log_strategy_decision(top_need, strategy, sorted_needs)
            return strategy
        
        # Step 2: No productive moves - check for dead-end stalling
        from src.interview.core.graph_needs_detector import GraphNeedsDetector
        # Create detector without config - will use default behavior
        needs_detector = GraphNeedsDetector()
        dead_end_score = needs_detector.calculate_dead_end_score(
            graph_state, interview_state
        )
        
        if dead_end_score >= self.dead_end_threshold:  # Use configured threshold
            logger.info("Dead-end detected (score: %.2f >= threshold: %.2f), using resolution strategy",
                       dead_end_score, self.dead_end_threshold)
            return StrategyName.DEAD_END_RESOLUTION
        
        # Step 3: Safety fallback - no needs, no dead-end
        logger.info("No productive needs and no dead-end detected, using seed expansion fallback")
        return StrategyName.SEED_EXPANSION
    
    def _log_strategy_decision(self, top_need: Need, selected_strategy: StrategyName, all_needs: List[Need]) -> None:
        """Log the strategy selection decision with rationale."""
        logger.debug("Strategy selection rationale:")
        logger.debug("  Top need: %s (score: %.2f)", top_need.name.value, top_need.score)
        logger.debug("  Selected strategy: %s", selected_strategy.value)
        
        if len(all_needs) > 1:
            logger.debug("  Other needs considered:")
            for i, need in enumerate(all_needs[1:], 1):
                logger.debug("    %d. %s (score: %.2f)", i, need.name.value, need.score)
        
        # Add context-specific rationale
        if top_need.name == NeedName.BRIDGE_ISOLATION:
            isolated_count = top_need.context.get("isolated_count", 0)
            logger.debug("  Rationale: %s isolated nodes need connection", isolated_count)
            
        elif top_need.name == NeedName.DEPTH_COMPLETION:
            avg_depth = top_need.context.get("average_depth", 0)
            shallow_count = top_need.context.get("shallow_count", 0)
            logger.debug("  Rationale: Average depth %.1f, %s shallow nodes need exploration", 
                        avg_depth, shallow_count)
            
        elif top_need.name == NeedName.SEED_EXPANSION:
            node_count = top_need.context.get("node_count", 0)
            min_threshold = top_need.context.get("min_threshold", 4)
            logger.debug("  Rationale: Only %s nodes, need at least %s for good coverage", 
                        node_count, min_threshold)
    
    def get_strategy_description(self, strategy: StrategyName) -> str:
        """Get a human-readable description of a strategy."""
        descriptions = {
            StrategyName.BRIDGE_BUILDING: "Connect isolated concepts to build a more cohesive understanding",
            StrategyName.DEPTH_COMPLETION: "Deepen exploration by probing for underlying reasons and meanings",
            StrategyName.SEED_EXPANSION: "Expand conceptual coverage by surfacing new topics and ideas"
        }
        
        return descriptions.get(strategy, f"Execute {strategy.value} strategy")
    
    def get_all_strategies(self) -> List[StrategyName]:
        """Get list of all available strategies."""
        return list(StrategyName)
    
    def validate_strategy(self, strategy: StrategyName) -> bool:
        """Validate that a strategy is supported."""
        return strategy in self.NEED_TO_STRATEGY_MAP.values()