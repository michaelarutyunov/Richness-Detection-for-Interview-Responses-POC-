"""
Opportunity Ranker for intelligent question selection.

Ranks graph nodes for exploration based on multiple criteria with enhancements:
- Exponential recency penalty
- Time-aware recency scoring
- Topic exhaustion detection
- Phase-adaptive weights
- Epsilon-greedy exploration
- Diversity bonus
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum

import networkx as nx

from src.core.data_models import InterviewPhase
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
    """Ranks graph exploration opportunities with adaptive strategies."""

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
        self.use_adaptive_weights = use_adaptive_weights
        self.recency_decay_function = recency_decay_function
        self.enable_time_aware_recency = enable_time_aware_recency
        self.enable_exhaustion_detection = enable_exhaustion_detection
        self.exhaustion_visit_threshold = exhaustion_visit_threshold
        self.enable_epsilon_greedy = enable_epsilon_greedy
        self.exploration_rates = {
            InterviewPhase.COVERAGE: exploration_rate_coverage,
            InterviewPhase.DEPTH: exploration_rate_depth,
            InterviewPhase.CONNECTION: exploration_rate_connection,
        }
        self.enable_diversity_bonus = enable_diversity_bonus
        self.diversity_weight = diversity_weight
        self._focus_stack = []  # Track recent exploration path

        # Default weights (will be overridden if adaptive)
        self.focus_weight = 2.0
        self.coverage_weight = 3.0
        self.depth_weight = 1.5
        self.recency_weight = 1.0

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
        if self.graph.node_count == 0:
            return []

        # Update weights based on phase
        if self.use_adaptive_weights:
            self._set_phase_adaptive_weights(interview_phase)

        opportunities = []
        coverage_metrics = self.graph.calculate_coverage()

        for node_id in self.graph.graph.nodes():
            node_data = self.graph.graph.nodes[node_id]["data"]

            # Skip exhausted topics
            if self.enable_exhaustion_detection and self._is_topic_exhausted(node_id):
                logger.debug(f"Skipping exhausted topic: {node_data.label}")
                continue

            # Calculate multi-dimensional scores
            coverage_score = self._calculate_coverage_score(node_data.type, coverage_metrics)
            depth_score = self._calculate_depth_score(node_id)
            recency_score = self._calculate_recency_score(
                node_data.visit_count, node_data.last_visit_turn, current_turn
            )
            focus_score = self._calculate_focus_score(node_id)

            # Optional diversity bonus
            diversity_score = 0.0
            if self.enable_diversity_bonus:
                diversity_score = self._calculate_diversity_score(node_id)

            # Weighted priority
            priority = (
                (coverage_score * self.coverage_weight)
                + (depth_score * self.depth_weight)
                + (recency_score * self.recency_weight)
                + (focus_score * self.focus_weight)
                + (diversity_score * self.diversity_weight)
            )

            # Determine strategy
            strategy = self._determine_strategy(node_id, node_data)

            # Debug logging: component scores for troubleshooting identical priorities
            logger.debug(
                f"Node {node_id} ({node_data.label}): coverage={coverage_score:.3f}, "
                f"depth={depth_score:.3f}, recency={recency_score:.3f}, "
                f"focus={focus_score:.3f}, diversity={diversity_score:.3f}, "
                f"priority={priority:.3f}, strategy={strategy.value}"
            )

            opportunities.append(
                RankedOpportunity(
                    node_id=node_id,
                    node_label=node_data.label,
                    node_type=node_data.type,
                    strategy=strategy,
                    priority_score=priority,
                    rationale=self._build_rationale(
                        node_data, coverage_score, depth_score, recency_score, diversity_score
                    ),
                    metadata={
                        "visit_count": node_data.visit_count,
                        "last_visit_turn": node_data.last_visit_turn,
                        "out_degree": self.graph.graph.out_degree(node_id),
                        "richness_weight": self.graph.schema.get_richness_weight(node_data.type),
                        "scores": {
                            "coverage": coverage_score,
                            "depth": depth_score,
                            "recency": recency_score,
                            "focus": focus_score,
                            "diversity": diversity_score,
                        },
                    },
                )
            )

        # Sort by priority (descending) with tie-breaking
        # If scores are identical, prioritize less-explored (fewer visits) and older nodes
        opportunities.sort(
            key=lambda o: (
                o.priority_score,                    # Primary: priority score
                -o.metadata["visit_count"],          # Tie-break 1: fewer visits (negative for descending)
                -o.metadata["creation_turn"]         # Tie-break 2: older nodes (negative for descending)
            ),
            reverse=True
        )

        return opportunities[:max_opportunities]

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
        if not opportunities:
            raise ValueError("No opportunities available")

        # Get exploration rate for current phase
        exploration_rate = self.exploration_rates.get(interview_phase, 0.2)

        # Epsilon-greedy selection
        if self.enable_epsilon_greedy and random.random() < exploration_rate:
            # Exploration: random from top 5
            candidates = opportunities[:min(5, len(opportunities))]
            selected = random.choice(candidates)
            logger.info(
                f"Turn {current_turn}: Exploring randomly (rate={exploration_rate:.1%}) "
                f"-> {selected.node_label}"
            )
            return selected
        else:
            # Exploitation: best opportunity
            return opportunities[0]

    def _set_phase_adaptive_weights(self, phase: InterviewPhase):
        """
        Set weights based on interview phase.

        Args:
            phase: Current interview phase
        """
        if phase == InterviewPhase.COVERAGE:
            # Early phase: explore broadly
            self.coverage_weight = 4.0
            self.depth_weight = 1.0
            self.recency_weight = 1.5
            self.focus_weight = 1.0
        elif phase == InterviewPhase.DEPTH:
            # Middle phase: balanced
            self.coverage_weight = 2.0
            self.depth_weight = 2.5
            self.recency_weight = 2.0
            self.focus_weight = 1.5
        elif phase == InterviewPhase.CONNECTION:
            # Later phase: connect concepts
            self.coverage_weight = 1.0
            self.depth_weight = 2.0
            self.recency_weight = 1.5
            self.focus_weight = 2.5
        else:
            # Wrap-up: default weights
            self.coverage_weight = 1.0
            self.depth_weight = 1.0
            self.recency_weight = 2.0
            self.focus_weight = 1.0

        logger.debug(
            f"Phase {phase.value}: weights=(coverage={self.coverage_weight}, "
            f"depth={self.depth_weight}, recency={self.recency_weight}, "
            f"focus={self.focus_weight})"
        )

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

    def _calculate_recency_score(
        self, visit_count: int, last_visit_turn: int | None, current_turn: int
    ) -> float:
        """
        Enhanced recency scoring with exponential decay and time-awareness.

        Args:
            visit_count: Number of times node visited
            last_visit_turn: Turn when last visited (None if never)
            current_turn: Current turn number

        Returns:
            Recency score (higher = less recently visited)
        """
        if self.recency_decay_function == "exponential":
            # Exponential decay: 1.0, 0.5, 0.25, 0.125, ...
            base_score = 1.0 / (2**visit_count)
        else:
            # Linear decay (original): 1.0, 0.5, 0.33, 0.25, ...
            base_score = 1.0 / (visit_count + 1)

        # Time-aware penalty
        if self.enable_time_aware_recency and last_visit_turn is not None:
            turns_since_visit = current_turn - last_visit_turn

            if turns_since_visit <= 2:
                # Very recent visit: strong penalty
                return base_score * 0.5
            elif turns_since_visit > 5:
                # Old visit: allow revisiting
                return min(base_score * 1.5, 1.0)

        return base_score

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

    def _calculate_diversity_score(self, node_id: str) -> float:
        """
        Bonus for exploring nodes far from recent focus.

        Encourages topic switching when focus gets stale.

        Args:
            node_id: Node to score

        Returns:
            Diversity score (higher = farther from focus)
        """
        if not self._focus_stack:
            return 0.5

        recent_focus = self._focus_stack[-3:]

        # Calculate graph distance to recent focus
        min_distance = float("inf")
        for focus_node in recent_focus:
            if focus_node not in self.graph.graph:
                continue

            try:
                distance = nx.shortest_path_length(
                    self.graph.graph.to_undirected(), source=node_id, target=focus_node
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

    def _is_topic_exhausted(self, node_id: str) -> bool:
        """
        Detect if a topic has been over-explored.

        Criteria:
        - visit_count >= threshold
        - Has been elaborated (out_degree > 0)
        - All successors also visited

        Args:
            node_id: Node to check

        Returns:
            True if topic is exhausted
        """
        node_data = self.graph.graph.nodes[node_id]["data"]

        # Check visit threshold
        if node_data.visit_count < self.exhaustion_visit_threshold:
            return False

        # Must have some elaboration
        out_degree = self.graph.graph.out_degree(node_id)
        if out_degree == 0:
            # No children yet, not exhausted
            return False

        # Check if all successors visited
        successors = list(self.graph.graph.successors(node_id))
        all_visited = all(
            self.graph.graph.nodes[s]["data"].visit_count > 0 for s in successors
        )

        if all_visited:
            logger.debug(
                f"Topic exhausted: {node_data.label} "
                f"(visits={node_data.visit_count}, children={out_degree}, all visited)"
            )
            return True

        return False

    def _determine_strategy(self, node_id: str, node_data) -> QuestionStrategy:
        """Determine best question strategy for this node."""
        out_degree = self.graph.graph.out_degree(node_id)
        visit_count = node_data.visit_count

        # Never visited: introduce
        if visit_count == 0:
            return QuestionStrategy.INTRODUCE_TOPIC

        # Visited but shallow: dig deeper
        if out_degree < 1:  # Changed from < 2 to trigger CONNECT_CONCEPTS earlier
            return QuestionStrategy.DIG_DEEPER

        # Well explored: connect to other concepts
        return QuestionStrategy.CONNECT_CONCEPTS

    def _build_rationale(
        self,
        node_data,
        coverage_score: float,
        depth_score: float,
        recency_score: float,
        diversity_score: float = 0.0,
    ) -> str:
        """Build human-readable rationale for ranking."""
        reasons = []

        if coverage_score > 0.5:
            reasons.append(f"underexplored type ({node_data.type})")

        if depth_score > 0.5:
            reasons.append("shallow branch")

        if recency_score > 0.5:
            reasons.append("not recently visited")

        if diversity_score > 0.7:
            reasons.append("diverse topic")

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
