"""
Core ranking engine for opportunity scoring.

Orchestrates multiple scoring strategies to rank graph nodes.
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from src.core.data_models import InterviewPhase
from src.core.interview_graph import InterviewGraph
from src.interview.ranking.exhaustion_detector import ExhaustionDetector
from src.interview.ranking.scoring_strategies import (
    CoverageScorer,
    DepthScorer,
    DiversityScorer,
    FocusScorer,
    RecencyScorer,
    ScoringStrategy,
)

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


class RankingEngine:
    """Orchestrates multi-dimensional node scoring."""

    def __init__(
        self,
        graph: InterviewGraph,
        strategies: List[ScoringStrategy] = None,
        exhaustion_detector: ExhaustionDetector = None,
        enable_epsilon_greedy: bool = True,
        exploration_rates: Dict[InterviewPhase, float] = None,
        use_adaptive_weights: bool = True,
        enable_diversity_bonus: bool = True,
        diversity_weight: float = 1.0,
    ):
        """
        Initialize ranking engine.

        Args:
            graph: Interview graph to rank
            strategies: List of scoring strategies (defaults to all 5)
            exhaustion_detector: Exhaustion detector instance
            enable_epsilon_greedy: Enable random exploration
            exploration_rates: Exploration probability per phase
            use_adaptive_weights: Use phase-adaptive weights
            enable_diversity_bonus: Add diversity as bonus (legacy behavior)
            diversity_weight: Weight for diversity bonus
        """
        self.graph = graph
        self.use_adaptive_weights = use_adaptive_weights
        self.enable_diversity_bonus = enable_diversity_bonus
        self.diversity_weight = diversity_weight

        # Default strategies (4 main + diversity as optional bonus)
        self.strategies = strategies or [
            CoverageScorer(),
            DepthScorer(),
            RecencyScorer(),
            FocusScorer(),
        ]

        # Diversity scorer (used conditionally as bonus)
        self.diversity_scorer = DiversityScorer(diversity_weight=diversity_weight)

        self.exhaustion_detector = exhaustion_detector or ExhaustionDetector()
        self.enable_epsilon_greedy = enable_epsilon_greedy
        self.exploration_rates = exploration_rates or {
            InterviewPhase.COVERAGE: 0.3,
            InterviewPhase.DEPTH: 0.2,
            InterviewPhase.CONNECTION: 0.1,
            InterviewPhase.WRAP_UP: 0.0,
        }

    def rank_opportunities(
        self,
        phase: InterviewPhase,
        turn_number: int,
        recent_nodes: List[str] = None,
        max_opportunities: int = 10,
    ) -> List[RankedOpportunity]:
        """
        Rank all graph nodes for exploration.

        Extracted from legacy lines 110-216.

        Args:
            phase: Current interview phase
            turn_number: Current turn number
            recent_nodes: Recently visited node IDs
            max_opportunities: Maximum opportunities to return

        Returns:
            List of ranked opportunities (best first)
        """
        if self.graph.node_count == 0:
            return []

        opportunities = []

        # Pre-calculate coverage metrics (shared across all nodes)
        coverage_metrics = self.graph.calculate_coverage()

        # Build context for scoring
        context = {
            "phase": phase,
            "turn_number": turn_number,
            "recent_nodes": recent_nodes or [],
            "coverage_metrics": coverage_metrics,
        }

        # Get exhausted nodes to filter
        exhausted = self.exhaustion_detector.get_exhausted_nodes(self.graph)

        # Score each node
        for node_id in self.graph.graph.nodes():
            node_data = self.graph.graph.nodes[node_id]["data"]

            # Skip exhausted nodes
            if node_id in exhausted:
                logger.debug(f"Skipping exhausted topic: {node_data.label}")
                continue

            # Calculate multi-dimensional scores
            total_score = 0.0
            score_breakdown = {}

            # Calculate scores for each strategy
            for strategy in self.strategies:
                score = strategy.calculate_score(node_id, self.graph, context)
                weight = (
                    strategy.get_weight(phase) if self.use_adaptive_weights else 1.0
                )
                weighted_score = score * weight

                total_score += weighted_score
                score_breakdown[strategy.__class__.__name__] = {
                    "score": score,
                    "weight": weight,
                    "weighted": weighted_score,
                }

            # Optional diversity bonus (legacy behavior: lines 154-156, 164)
            diversity_score = 0.0
            if self.enable_diversity_bonus:
                diversity_score = self.diversity_scorer.calculate_score(
                    node_id, self.graph, context
                )
                diversity_bonus = diversity_score * self.diversity_weight
                total_score += diversity_bonus
                score_breakdown["DiversityScorer"] = {
                    "score": diversity_score,
                    "weight": self.diversity_weight,
                    "weighted": diversity_bonus,
                }

            # Determine strategy (extracted from lines 457-471)
            strategy = self._determine_strategy(node_id, node_data)

            # Build rationale (extracted from lines 473-499)
            rationale = self._build_rationale(score_breakdown)

            # Debug logging (lines 171-176)
            logger.debug(
                f"Node {node_id} ({node_data.label}): "
                f"priority={total_score:.3f}, strategy={strategy.value}"
            )

            # Create opportunity
            opportunity = RankedOpportunity(
                node_id=node_id,
                node_label=node_data.label,
                node_type=node_data.type,
                strategy=strategy,
                priority_score=total_score,
                rationale=rationale,
                metadata={
                    "visit_count": node_data.visit_count,
                    "last_visit_turn": node_data.last_visit_turn,
                    "creation_turn": node_data.creation_turn,  # For tie-breaking
                    "out_degree": self.graph.graph.out_degree(node_id),
                    "richness_weight": self.graph.schema.get_richness_weight(
                        node_data.type
                    ),
                    "scores": score_breakdown,
                },
            )

            opportunities.append(opportunity)

        # Sort by priority with 3-level tie-breaking (CRITICAL - lines 207-214)
        opportunities.sort(
            key=lambda o: (
                o.priority_score,  # Primary: priority score
                -o.metadata["visit_count"],  # Tie-break 1: fewer visits
                -o.metadata["creation_turn"],  # Tie-break 2: older nodes
            ),
            reverse=True,
        )

        return opportunities[:max_opportunities]

    def select_opportunity_with_exploration(
        self,
        opportunities: List[RankedOpportunity],
        interview_phase: InterviewPhase = InterviewPhase.COVERAGE,
        current_turn: int = 0,
    ) -> RankedOpportunity:
        """
        Select opportunity using epsilon-greedy strategy.

        Extracted from legacy lines 218-253.

        Args:
            opportunities: Ranked opportunities list
            interview_phase: Current interview phase
            current_turn: Current turn number

        Returns:
            Selected opportunity
        """
        if not opportunities:
            raise ValueError("No opportunities available")

        # Get exploration rate for current phase
        exploration_rate = self.exploration_rates.get(interview_phase, 0.2)

        # Epsilon-greedy selection
        if self.enable_epsilon_greedy and random.random() < exploration_rate:
            # Exploration: random from top 5
            candidates = opportunities[: min(5, len(opportunities))]
            selected = random.choice(candidates)
            logger.info(
                f"Turn {current_turn}: Exploring randomly (rate={exploration_rate:.1%}) "
                f"-> {selected.node_label}"
            )
            return selected
        else:
            # Exploitation: best opportunity
            return opportunities[0]

    def _determine_strategy(
        self, node_id: str, node_data
    ) -> QuestionStrategy:
        """
        Determine best question strategy for this node.

        Extracted from legacy lines 457-471.
        """
        out_degree = self.graph.graph.out_degree(node_id)
        visit_count = node_data.visit_count

        # Never visited: introduce
        if visit_count == 0:
            return QuestionStrategy.INTRODUCE_TOPIC

        # Visited but shallow: dig deeper
        # Note: Changed from < 2 to < 1 (line 467 comment)
        if out_degree < 1:
            return QuestionStrategy.DIG_DEEPER

        # Well explored: connect to other concepts
        return QuestionStrategy.CONNECT_CONCEPTS

    def _build_rationale(self, score_breakdown: Dict) -> str:
        """
        Build human-readable rationale for ranking.

        Extracted from legacy lines 473-499.
        """
        if not score_breakdown:
            return "available for exploration"

        reasons = []

        # Extract individual scores
        for strategy_name, scores in score_breakdown.items():
            score = scores["score"]

            if strategy_name == "CoverageScorer" and score > 0.5:
                reasons.append("underexplored type")
            elif strategy_name == "DepthScorer" and score > 0.5:
                reasons.append("shallow branch")
            elif strategy_name == "RecencyScorer" and score > 0.5:
                reasons.append("not recently visited")
            elif strategy_name == "DiversityScorer" and score > 0.7:
                reasons.append("diverse topic")

        if not reasons:
            reasons.append("available for exploration")

        return ", ".join(reasons)
