"""
Public API for opportunity ranking.

Facade that simplifies access to ranking engine while maintaining
backward compatibility with the legacy implementation.
"""

import logging

from src.core.data_models import InterviewPhase
from src.core.interview_graph import InterviewGraph
from src.interview.ranking.exhaustion_detector import ExhaustionDetector
from src.interview.ranking.ranking_engine import (
    QuestionStrategy,
    RankedOpportunity,
    RankingEngine,
)
from src.interview.ranking.scoring_strategies import (
    CoverageScorer,
    DepthScorer,
    FocusScorer,
    RecencyScorer,
)

logger = logging.getLogger(__name__)


class OpportunityRanker:
    """
    Public API for ranking graph exploration opportunities.

    Simplified facade over RankingEngine with sensible defaults.
    Maintains backward compatibility with legacy implementation.
    """

    def __init__(
        self,
        graph: InterviewGraph,
        # Adaptive weight configuration
        use_adaptive_weights: bool = True,
        # Recency configuration
        recency_decay_function: str = "exponential",
        enable_time_aware_recency: bool = True,
        # Exhaustion detection
        enable_exhaustion_detection: bool = True,
        exhaustion_visit_threshold: int = 3,
        # Exploration configuration
        enable_epsilon_greedy: bool = True,
        exploration_rate_coverage: float = 0.3,
        exploration_rate_depth: float = 0.2,
        exploration_rate_connection: float = 0.1,
        # Diversity configuration
        enable_diversity_bonus: bool = True,
        diversity_weight: float = 1.0,
    ):
        """
        Initialize opportunity ranker with enhanced features.

        Args:
            graph: Interview graph to analyze
            use_adaptive_weights: Use phase-adaptive weights
            recency_decay_function: "linear" or "exponential"
            enable_time_aware_recency: Penalize recent visits more heavily
            enable_exhaustion_detection: Skip over-explored topics
            exhaustion_visit_threshold: Visit count to mark as exhausted
            enable_epsilon_greedy: Random exploration to escape local maxima
            exploration_rate_coverage: Exploration rate in coverage phase (0-1)
            exploration_rate_depth: Exploration rate in depth phase (0-1)
            exploration_rate_connection: Exploration rate in connection phase (0-1)
            enable_diversity_bonus: Bonus for exploring distant nodes
            diversity_weight: Weight for diversity bonus
        """
        self.graph = graph
        self._focus_stack = []  # Track recent exploration path

        # Create exhaustion detector
        self.exhaustion_detector = ExhaustionDetector(
            visit_threshold=exhaustion_visit_threshold,
            enable_detection=enable_exhaustion_detection,
        )

        # Create scoring strategies with configuration
        strategies = [
            CoverageScorer(),
            DepthScorer(),
            RecencyScorer(
                decay_function=recency_decay_function,
                enable_time_aware_recency=enable_time_aware_recency,
            ),
            FocusScorer(),
        ]

        # Create ranking engine
        self.engine = RankingEngine(
            graph=graph,
            strategies=strategies,
            exhaustion_detector=self.exhaustion_detector,
            enable_epsilon_greedy=enable_epsilon_greedy,
            exploration_rates={
                InterviewPhase.COVERAGE: exploration_rate_coverage,
                InterviewPhase.DEPTH: exploration_rate_depth,
                InterviewPhase.CONNECTION: exploration_rate_connection,
                InterviewPhase.WRAP_UP: 0.0,
            },
            use_adaptive_weights=use_adaptive_weights,
            enable_diversity_bonus=enable_diversity_bonus,
            diversity_weight=diversity_weight,
        )

    def rank_opportunities(
        self,
        max_opportunities: int = 10,
        current_turn: int = 0,
        interview_phase: InterviewPhase = InterviewPhase.COVERAGE,
    ) -> list[RankedOpportunity]:
        """
        Rank all exploration opportunities in graph.

        Args:
            max_opportunities: Maximum number of opportunities to return
            current_turn: Current turn number (for time-aware recency)
            interview_phase: Current interview phase (for adaptive weights)

        Returns:
            List of ranked opportunities sorted by priority (descending)
        """
        # Build recent nodes from focus stack
        recent_nodes = self._focus_stack[-5:] if self._focus_stack else []

        opportunities = self.engine.rank_opportunities(
            phase=interview_phase,
            turn_number=current_turn,
            recent_nodes=recent_nodes,
            max_opportunities=max_opportunities,
        )

        logger.info(
            f"Ranked {len(opportunities)} opportunities for phase {interview_phase.value}"
        )

        return opportunities

    def select_opportunity_with_exploration(
        self,
        opportunities: list[RankedOpportunity],
        interview_phase: InterviewPhase = InterviewPhase.COVERAGE,
        current_turn: int = 0,
    ) -> RankedOpportunity:
        """
        Select opportunity using epsilon-greedy strategy.

        Args:
            opportunities: Ranked opportunities list
            interview_phase: Current interview phase
            current_turn: Current turn number

        Returns:
            Selected opportunity
        """
        return self.engine.select_opportunity_with_exploration(
            opportunities=opportunities,
            interview_phase=interview_phase,
            current_turn=current_turn,
        )

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
            logger.info(
                f"Richness threshold reached: {current_richness:.2f} >= {min_richness}"
            )
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
