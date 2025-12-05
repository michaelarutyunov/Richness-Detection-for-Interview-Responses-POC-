"""
Schema-driven TacticSelector - Updated to work with the new v0.2 schema format.
Uses schema-defined strategies and tactics instead of separate configuration files.
"""

import logging
from typing import List, Optional, Dict, Any
from src.core.models import StrategyName, InterviewState, SchemaTactic
from src.core.schema_loader import SchemaLoader
from src.interview.tactics.loader import SchemaDrivenTacticLoader

logger = logging.getLogger(__name__)


class SchemaDrivenTacticSelector:
    """
    Selects tactics based on schema-defined strategies and constraints.
    Works directly with the unified schema format.
    """
    
    def __init__(
        self,
        schema_loader: Optional[SchemaLoader] = None,
        usage_penalty_weight: float = 0.7,
        recency_penalty_weight: float = 0.15,
        recency_penalty_cap: float = 0.5,
        recent_tactics_count: int = 3,
        recent_questions_count: int = 6
    ):
        """
        Initialize with configuration parameters.

        Args:
            schema_loader: Schema loader instance (optional)
            usage_penalty_weight: Weight for usage frequency penalty (default: 0.7)
            recency_penalty_weight: Weight for recency penalty (default: 0.15)
            recency_penalty_cap: Maximum recency penalty (default: 0.5)
            recent_tactics_count: Number of recent tactics to track (default: 3)
            recent_questions_count: Number of recent questions to track (default: 6)
        """
        self.schema_loader = schema_loader or SchemaLoader()
        self.tactic_loader = SchemaDrivenTacticLoader(self.schema_loader)
        self.usage_penalty_weight = usage_penalty_weight
        self.recency_penalty_weight = recency_penalty_weight
        self.recency_penalty_cap = recency_penalty_cap
        self.recent_tactics_count = recent_tactics_count
        self.recent_questions_count = recent_questions_count
        logger.info("SchemaDrivenTacticSelector initialized with usage_penalty_weight=%.2f, recency_penalty_weight=%.2f",
                   usage_penalty_weight, recency_penalty_weight)
    
    def select(self, strategy: StrategyName, interview_state: InterviewState, 
               available_tactics: List[SchemaTactic]) -> Optional[SchemaTactic]:
        """
        Select the best tactic for the given strategy and state.
        
        Args:
            strategy: The selected strategy
            interview_state: Current interview state
            available_tactics: List of all available tactics
            
        Returns:
            Selected tactic or None if no valid tactics found
        """
        logger.info(f"Selecting tactic for strategy: {strategy.value} (turn {interview_state.turn_number})")
        
        # Get tactics that support this strategy from schema
        strategy_tactics = self._get_strategy_tactics(strategy)
        logger.debug(f"Strategy {strategy.value} supports tactics: {strategy_tactics}")
        
        if not strategy_tactics:
            logger.warning(f"No tactics found for strategy: {strategy.value}")
            return None
        
        # Filter available tactics by strategy
        compatible_tactics = [
            tactic for tactic in available_tactics 
            if tactic.id in strategy_tactics
        ]
        
        logger.debug(f"Found {len(compatible_tactics)} compatible tactics for strategy {strategy.value}")
        
        if not compatible_tactics:
            logger.warning(f"No compatible tactics available for strategy: {strategy.value}")
            return None
        
        # Apply safety constraints
        valid_tactics = self._apply_constraints(compatible_tactics, interview_state)
        logger.debug(f"After constraints: {len(valid_tactics)} valid tactics")
        
        if not valid_tactics:
            logger.warning(f"No tactics passed constraint filters for strategy: {strategy.value}")
            return None
        
        # Score tactics for variety
        scored_tactics = self._score_tactics(valid_tactics, interview_state)
        
        # Select highest-scoring tactic
        best_tactic = max(scored_tactics, key=lambda x: x[1])[0]
        
        logger.info(f"Selected tactic: {best_tactic.id} for strategy: {strategy.value} (score: {max(scored_tactics, key=lambda x: x[1])[1]:.2f})")
        
        self._log_tactic_selection(strategy, best_tactic, scored_tactics, interview_state)
        
        return best_tactic
    
    def _get_strategy_tactics(self, strategy: StrategyName) -> List[str]:
        """Get list of tactic IDs that support the given strategy from schema."""
        strategy_def = self.schema_loader.get_strategy(strategy.value.lower())
        if not strategy_def:
            logger.warning(f"Strategy '{strategy.value}' not found in schema")
            return []
        
        return strategy_def.tactics
    
    def _apply_constraints(self, tactics: List[SchemaTactic], interview_state: InterviewState) -> List[SchemaTactic]:
        """Apply safety constraint filters to tactics."""
        valid_tactics = []
        
        for tactic in tactics:
            # Check minimum turn constraint
            if interview_state.turn_number < tactic.min_turn:
                logger.debug(f"Filtered out {tactic.id}: turn {interview_state.turn_number} < min_turn {tactic.min_turn}")
                continue
            
            # Check maximum visit count
            usage_count = interview_state.get_tactic_usage_count(tactic.id)
            if usage_count >= tactic.max_visit_count:
                logger.debug(f"Filtered out {tactic.id}: usage count {usage_count} >= max {tactic.max_visit_count}")
                continue
            

            

            
            valid_tactics.append(tactic)
        
        return valid_tactics
    
    def _score_tactics(self, tactics: List[SchemaTactic], interview_state: InterviewState) -> List[tuple]:
        """Score tactics based on variety and other factors."""
        scored_tactics = []
        
        for tactic in tactics:
            score = 0.0
            
            # Variety scoring: penalize recently used tactics
            usage_count = interview_state.get_tactic_usage_count(tactic.id)
            recency_penalty = self._calculate_recency_penalty(tactic.id, interview_state)

            # Base score uses exponential decay to prevent score collapse
            # Heavily-used tactics remain distinguishable (not all at 0.0)
            score = 1.0 * (0.7 ** usage_count) - recency_penalty
            
            # Ensure score doesn't go negative
            score = max(0.0, score)
            
            scored_tactics.append((tactic, score))
            logger.debug(f"SchemaTactic {tactic.id} scored: {score:.2f} (usage: {usage_count}, recency penalty: {recency_penalty:.2f})")
        
        return scored_tactics
    
    def _calculate_recency_penalty(self, tactic_id: str, interview_state: InterviewState) -> float:
        """Calculate penalty for recently used tactics based on turn tracking."""
        # Get usage count from interview state tracking
        usage_count = interview_state.get_tactic_usage_count(tactic_id)

        # Calculate recency weight based on actual usage, not string matching
        recency_weight = 0.15
        penalty = usage_count * recency_weight

        # Cap penalty to avoid completely eliminating tactics
        return min(penalty, 0.5)

    def _log_tactic_selection(self, strategy: StrategyName, selected_tactic: SchemaTactic, 
                             scored_tactics: List[tuple], interview_state: InterviewState) -> None:
        """Log the tactic selection decision with rationale."""
        logger.debug("SchemaTactic selection details:")
        logger.debug(f"  Strategy: {strategy.value}")
        logger.debug(f"  Selected: {selected_tactic.id} (score: {max(scored_tactics, key=lambda x: x[1])[1]:.2f})")
        logger.debug("  All scored tactics:")
        
        for tactic, score in sorted(scored_tactics, key=lambda x: x[1], reverse=True):
            usage_count = interview_state.get_tactic_usage_count(tactic.id)
            marker = " ← SELECTED" if tactic.id == selected_tactic.id else ""
            logger.debug(f"    {tactic.id}: {score:.2f} (used {usage_count} times){marker}")
        
        logger.debug(f"  Interview state: turn {interview_state.turn_number}")
    
    def get_strategy_description(self, strategy: StrategyName) -> str:
        """Get description of a strategy from the schema."""
        strategy_def = self.schema_loader.get_strategy(strategy.value.lower())
        if strategy_def:
            return strategy_def.description
        return f"Strategy {strategy.value}"
    
    def get_strategy_priority(self, strategy: StrategyName) -> float:
        """Get priority score for a strategy."""
        # This could be enhanced to read from schema if needed
        # For now, use default priorities
        default_priorities = {
            StrategyName.SEED_EXPANSION: 0.9,
            StrategyName.BRIDGE_BUILDING: 0.7,
            StrategyName.DEPTH_COMPLETION: 0.6
        }
        return default_priorities.get(strategy, 0.5)
    
    def get_tactic_constraints(self, tactic: SchemaTactic) -> Dict[str, Any]:
        """Get constraint information for a tactic."""
        return {
            "min_turn": tactic.min_turn,
            "max_visit_count": tactic.max_visit_count,
            "schema_metadata": self.tactic_loader.get_tactic_metadata(tactic.id)
        }