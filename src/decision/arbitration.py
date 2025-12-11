"""
Utility-based strategy arbitration layer.

Replaces first-applicable-wins selection with multi-scorer arbitration
that weighs all applicable strategies to select the highest utility option.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from decision.strategy import Strategy, FocusTarget

from core.graph import Graph, Node
from core.state import (
    GraphState,
    CoverageState,
    Momentum,
    NodeFocusTracker,
    EdgeFocusTracker
)
from core.history import History, Turn

logger = logging.getLogger(__name__)


@dataclass
class ScoringContext:
    """All state needed for scoring decisions."""

    graph: Graph
    graph_state: GraphState
    coverage_state: CoverageState
    momentum: Momentum
    history: History
    recent_questions: List[str]  # Last 6 questions for deduplication
    node_focus_tracker: Optional[NodeFocusTracker] = None
    edge_focus_tracker: Optional[EdgeFocusTracker] = None

    @classmethod
    def build(
        cls,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum,
        history: History,
        node_focus_tracker: Optional[NodeFocusTracker] = None,
        edge_focus_tracker: Optional[EdgeFocusTracker] = None
    ) -> "ScoringContext":
        """Build scoring context from interview state."""
        return cls(
            graph=graph,
            graph_state=graph_state,
            coverage_state=coverage_state,
            momentum=momentum,
            history=history,
            recent_questions=history.get_recent_questions(6),
            node_focus_tracker=node_focus_tracker,
            edge_focus_tracker=edge_focus_tracker
        )


class StrategyScorer(ABC):
    """
    Base class for all strategy scorers.

    Each scorer returns a multiplier (0.0-2.0) where:
    - 1.0 = neutral (no effect)
    - < 1.0 = penalty (reduce likelihood)
    - > 1.0 = boost (increase likelihood)
    """

    name: str = "base_scorer"
    weight: float = 1.0  # Global weight for this scorer

    @abstractmethod
    def score(
        self,
        strategy: "Strategy",
        focus: "FocusTarget",
        context: ScoringContext
    ) -> float:
        """
        Score a strategy+focus combination.

        Args:
            strategy: The strategy being evaluated
            focus: The focus target for this strategy
            context: Full scoring context

        Returns:
            Multiplier between 0.0 and 2.0
        """
        pass


class RedundancyScorer(StrategyScorer):
    """
    Prevents repetitive questions.

    Generates a hypothetical question template and compares to recent questions
    using Jaccard similarity. Heavy penalty if similarity > threshold.

    Problem Fixed: Asking "how does X lead to Y" in 4+ variations
    """

    name = "redundancy"
    similarity_threshold: float = 0.85
    penalty_multiplier: float = 0.2

    def __init__(self, threshold: float = 0.85, penalty: float = 0.2):
        self.similarity_threshold = threshold
        self.penalty_multiplier = penalty

    def score(
        self,
        strategy: "Strategy",
        focus: "FocusTarget",
        context: ScoringContext
    ) -> float:
        if not context.recent_questions:
            return 1.0

        # Generate hypothetical question template
        hypothetical = self._generate_template_question(strategy, focus)
        if not hypothetical:
            return 1.0

        # Check similarity to recent questions
        for past_q in context.recent_questions:
            similarity = self._jaccard_similarity(hypothetical, past_q)
            if similarity > self.similarity_threshold:
                logger.info(
                    f"[Redundancy] Penalizing {strategy.id}: "
                    f"{similarity:.2f} similar to recent question"
                )
                return self.penalty_multiplier

        return 1.0

    def _generate_template_question(
        self,
        strategy: "Strategy",
        focus: "FocusTarget"
    ) -> str:
        """Generate a template question based on strategy and focus."""
        if focus.node:
            label = focus.node.label
            if strategy.id == "connect_isolate":
                return f"how does {label} relate connect"
            elif strategy.id == "resolve_schema_tension":
                return f"how does {label} lead to"
            elif strategy.id == "deepen_branch":
                return f"what does {label} mean matters"
            elif strategy.id == "resolve_ambiguity":
                return f"what do you mean by {label} example"

        if focus.node_pair:
            n1, n2 = focus.node_pair
            return f"how does {n1.label} relate to {n2.label} lead connection"

        if focus.element:
            return f"{focus.element.id} think feel about"

        if focus.coverage_gap:
            return f"{focus.coverage_gap.element_id} thoughts reaction"

        return ""

    def _jaccard_similarity(self, q1: str, q2: str) -> float:
        """Calculate Jaccard similarity between two questions."""
        words1 = set(self._normalize(q1).split())
        words2 = set(self._normalize(q2).split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'same', 'that', 'this', 'these',
            'those', 'it', 'its', 'you', 'your', 'i', 'me', 'my', 'we',
            'our', 'they', 'their', 'and', 'or', 'but', 'if', 'then'
        }
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return ' '.join(words)


class KnowledgeCeilingScorer(StrategyScorer):
    """
    Stops drilling topics respondent lacks knowledge about.

    Detects "I don't know" patterns in recent responses for the focus element.
    Severe penalty to avoid frustrating the respondent.

    Problem Fixed: Continuing to ask about enzymes after user said "I don't know"
    """

    name = "knowledge_ceiling"
    penalty_multiplier: float = 0.1
    lookback_turns: int = 3

    # Patterns indicating lack of knowledge
    ceiling_patterns = [
        "i don't know",
        "i do not know",
        "not sure",
        "don't understand",
        "i heard about",
        "i guess",
        "i suppose",
        "no idea",
        "never thought about",
        "haven't considered",
        "don't really know",
        "couldn't say",
        "can't say",
        "not familiar",
        "don't have much",
        "don't know much"
    ]

    def __init__(self, lookback: int = 3, penalty: float = 0.1):
        self.lookback_turns = lookback
        self.penalty_multiplier = penalty

    def score(
        self,
        strategy: "Strategy",
        focus: "FocusTarget",
        context: ScoringContext
    ) -> float:
        # Get the element or node being focused on
        target_label = self._get_focus_label(focus)
        if not target_label:
            return 1.0

        # Check if knowledge ceiling detected for this target
        if self._detect_knowledge_ceiling(target_label, context.history):
            logger.info(
                f"[KnowledgeCeiling] Penalizing {strategy.id} "
                f"for target '{target_label}' - respondent lacks knowledge"
            )
            return self.penalty_multiplier

        return 1.0

    def _get_focus_label(self, focus: "FocusTarget") -> str:
        """Extract the main label from focus target."""
        if focus.node:
            return focus.node.label.lower()
        if focus.element:
            return focus.element.id.lower()
        if focus.coverage_gap:
            return focus.coverage_gap.element_id.lower()
        if focus.node_pair:
            # Return both labels for pair
            return f"{focus.node_pair[0].label} {focus.node_pair[1].label}".lower()
        return ""

    def _detect_knowledge_ceiling(
        self,
        target_label: str,
        history: History
    ) -> bool:
        """Check if recent responses show knowledge ceiling for this target."""
        recent_turns = history.get_recent(self.lookback_turns)

        for turn in recent_turns:
            response = turn.response.lower()

            # Check if response mentions the target
            target_words = target_label.split()
            target_mentioned = any(word in response for word in target_words if len(word) > 2)

            if target_mentioned:
                # Check for ceiling patterns
                for pattern in self.ceiling_patterns:
                    if pattern in response:
                        return True

        return False


class MomentumAlignmentScorer(StrategyScorer):
    """
    Adjusts strategy selection based on engagement level.

    Low momentum: boost breadth strategies, penalize depth strategies
    High momentum: boost depth strategies

    Problem Fixed: Continuing depth probing during 9 turns of low momentum
    """

    name = "momentum_alignment"

    # Strategy categorization
    breadth_strategies = ["explore_breadth", "introduce_seed", "ensure_coverage"]
    depth_strategies = ["deepen_branch", "resolve_schema_tension", "connect_isolate"]

    # Multipliers
    low_momentum_breadth_boost: float = 1.5
    low_momentum_depth_penalty: float = 0.5
    high_momentum_depth_boost: float = 1.3

    def __init__(
        self,
        breadth_boost: float = 1.5,
        depth_penalty: float = 0.5,
        depth_boost: float = 1.3
    ):
        self.low_momentum_breadth_boost = breadth_boost
        self.low_momentum_depth_penalty = depth_penalty
        self.high_momentum_depth_boost = depth_boost

    def score(
        self,
        strategy: "Strategy",
        focus: "FocusTarget",
        context: ScoringContext
    ) -> float:
        momentum_level = context.momentum.level

        if momentum_level == "low":
            if strategy.id in self.breadth_strategies:
                logger.info(
                    f"[Momentum] Boosting {strategy.id} for low momentum"
                )
                return self.low_momentum_breadth_boost
            elif strategy.id in self.depth_strategies:
                logger.info(
                    f"[Momentum] Penalizing {strategy.id} for low momentum"
                )
                return self.low_momentum_depth_penalty

        elif momentum_level == "high":
            if strategy.id in self.depth_strategies:
                logger.info(
                    f"[Momentum] Boosting {strategy.id} for high momentum"
                )
                return self.high_momentum_depth_boost

        return 1.0


class RecencyDiversityScorer(StrategyScorer):
    """
    Penalizes recently-used strategies to encourage diversity.

    Simple recency penalty: if strategy used in last N turns, apply penalty.

    Problem Fixed: Strategy monopolization (73% resolve_schema_tension)
    """

    name = "recency_diversity"
    lookback_turns: int = 2
    recency_penalty: float = 0.7

    def __init__(self, lookback: int = 2, penalty: float = 0.7):
        self.lookback_turns = lookback
        self.recency_penalty = penalty

    def score(
        self,
        strategy: "Strategy",
        focus: "FocusTarget",
        context: ScoringContext
    ) -> float:
        # Get strategies used in recent turns
        recent_turns = context.history.get_recent(self.lookback_turns)
        recent_strategies = [t.strategy_used for t in recent_turns]

        # Count how many times this strategy was used recently
        usage_count = recent_strategies.count(strategy.id)

        if usage_count > 0:
            # Apply cumulative penalty for repeated use
            penalty = self.recency_penalty ** usage_count
            logger.info(
                f"[RecencyDiversity] Penalizing {strategy.id}: "
                f"used {usage_count}x in last {self.lookback_turns} turns "
                f"(penalty: {penalty:.2f})"
            )
            return penalty

        return 1.0


class VerticalLadderingScorer(StrategyScorer):
    """
    Detects horizontal saturation and boosts upward linking.

    When graph has many concrete nodes but few abstract/value nodes,
    boosts strategies that encourage vertical exploration.

    Problem Fixed: All nodes are sensory/mechanical, no values/meaning
    """

    name = "vertical_laddering"
    boost_multiplier: float = 1.5
    value_proximity_boost: float = 1.8
    value_closure_boost: float = 2.0
    near_value_depth: int = 2

    # Node types considered "abstract" or "value" (meaning-focused)
    abstract_types = ["value", "belief", "goal", "need", "motivation", "meaning"]
    concrete_types = ["feature", "attribute", "action", "process", "sensory"]

    # High abstraction types for value proximity
    high_abstraction_types = ["psychosocial_consequence", "value", "belief", "goal"]

    # Strategies that encourage vertical exploration
    vertical_strategies = ["deepen_branch"]  # Uses upward_linking tactic

    def __init__(
        self,
        boost: float = 1.5,
        value_proximity_boost: float = 1.8,
        value_closure_boost: float = 2.0,
        near_value_depth: int = 2
    ):
        self.boost_multiplier = boost
        self.value_proximity_boost = value_proximity_boost
        self.value_closure_boost = value_closure_boost
        self.near_value_depth = near_value_depth

    def score(
        self,
        strategy: "Strategy",
        focus: "FocusTarget",
        context: ScoringContext
    ) -> float:
        # Check value proximity FIRST for deepen_branch
        if strategy.id == "deepen_branch":
            proximity_score = self._get_value_proximity_score(focus, context)
            if proximity_score > 1.0:
                return proximity_score

        # Check if graph is horizontally saturated
        if not self._is_horizontally_saturated(context):
            return 1.0

        # Boost strategies that use upward_linking
        if strategy.id in self.vertical_strategies:
            logger.info(
                f"[{self.name}] {strategy.id} | "
                f"score={self.boost_multiplier:.2f} | reason=horizontal saturation detected"
            )
            return self.boost_multiplier

        # Also check if strategy's tactics include upward_linking
        if "upward_linking" in strategy.suggested_tactics:
            logger.info(
                f"[{self.name}] {strategy.id} | "
                f"score={self.boost_multiplier:.2f} | reason=includes upward_linking tactic"
            )
            return self.boost_multiplier

        return 1.0

    def _get_value_proximity_score(self, focus: "FocusTarget", context: ScoringContext) -> float:
        """Calculate boost based on proximity to value nodes."""
        if not focus.node:
            return 1.0

        node = focus.node

        # If current node is high abstraction: return value_closure_boost
        if node.node_type and any(t in node.node_type.lower() for t in self.high_abstraction_types):
            logger.info(
                f"[{self.name}] deepen_branch | "
                f"score={self.value_closure_boost:.2f} | reason=node is high abstraction type ({node.node_type})"
            )
            return self.value_closure_boost

        # Check depth to nearest value node
        depth = self._get_depth_to_value(node, context.graph)
        if depth is not None and depth <= self.near_value_depth:
            logger.info(
                f"[{self.name}] deepen_branch | "
                f"score={self.value_proximity_boost:.2f} | reason=node is {depth} steps from value node"
            )
            return self.value_proximity_boost

        return 1.0

    def _get_depth_to_value(self, node: Node, graph: Graph) -> Optional[int]:
        """Get minimum depth from node to any value-type node via outgoing edges using BFS."""
        from collections import deque

        # BFS queue: (current_node_id, depth)
        queue = deque([(node.id, 0)])
        visited = {node.id}

        while queue:
            current_id, depth = queue.popleft()

            # Get outgoing edges
            outgoing = graph.get_outgoing_edges(current_id)

            for edge in outgoing:
                target_id = edge.target_id

                if target_id in visited:
                    continue

                visited.add(target_id)
                target_node = graph.get_node(target_id)

                if target_node and target_node.node_type:
                    # Check if target is a value-type node
                    if any(t in target_node.node_type.lower() for t in self.high_abstraction_types):
                        return depth + 1

                # Add to queue for further exploration
                queue.append((target_id, depth + 1))

        return None  # No value node reachable

    def _is_horizontally_saturated(self, context: ScoringContext) -> bool:
        """Check if graph has too many concrete nodes vs abstract."""
        nodes = list(context.graph.nodes.values())

        if len(nodes) < 5:
            return False  # Not enough nodes to judge

        abstract_count = 0
        concrete_count = 0

        for node in nodes:
            if not node.node_type:
                continue
            node_type = node.node_type.lower()
            if any(t in node_type for t in self.abstract_types):
                abstract_count += 1
            elif any(t in node_type for t in self.concrete_types):
                concrete_count += 1

        # Consider saturated if < 20% abstract nodes
        total = abstract_count + concrete_count
        if total > 0:
            abstract_ratio = abstract_count / total
            if abstract_ratio < 0.2:
                return True

        return False


class BranchHealthScorer(StrategyScorer):
    """
    Detects terminal/stale branches and triggers breadth exploration.

    A branch is "stale" if the active branch node hasn't produced
    new edges in N turns.

    Problem Fixed: Getting stuck on exhausted exploration paths
    """

    name = "branch_health"
    stale_threshold: int = 2  # Reduced from 3
    breadth_boost: float = 1.8  # Increased from 1.5
    depth_penalty: float = 0.3  # More severe (was 0.6)
    severe_stale_threshold: int = 4
    severe_depth_penalty: float = 0.1
    connect_isolate_penalty: float = 0.5

    def __init__(
        self,
        stale_threshold: int = 2,
        breadth_boost: float = 1.8,
        depth_penalty: float = 0.3,
        severe_stale_threshold: int = 4,
        severe_depth_penalty: float = 0.1,
        connect_isolate_penalty: float = 0.5
    ):
        self.stale_threshold = stale_threshold
        self.breadth_boost = breadth_boost
        self.depth_penalty = depth_penalty
        self.severe_stale_threshold = severe_stale_threshold
        self.severe_depth_penalty = severe_depth_penalty
        self.connect_isolate_penalty = connect_isolate_penalty

    def score(
        self,
        strategy: "Strategy",
        focus: "FocusTarget",
        context: ScoringContext
    ) -> float:
        # Get stale turn count
        stale_turns = self._get_stale_turn_count(context)

        # Check if current branch is stale
        if stale_turns < self.stale_threshold:
            return 1.0

        # Boost breadth strategies
        if strategy.id == "explore_breadth":
            logger.info(
                f"[{self.name}] {strategy.id} | "
                f"score={self.breadth_boost:.2f} | reason=branch_stale (turns={stale_turns})"
            )
            return self.breadth_boost

        # Boost introduce_seed for very stale branches
        if strategy.id == "introduce_seed" and stale_turns >= self.severe_stale_threshold:
            logger.info(
                f"[{self.name}] {strategy.id} | "
                f"score={self.breadth_boost:.2f} | reason=branch_very_stale (turns={stale_turns})"
            )
            return self.breadth_boost

        # Apply penalties to depth strategies
        if strategy.id in ["deepen_branch", "resolve_schema_tension"]:
            if stale_turns >= self.severe_stale_threshold:
                logger.info(
                    f"[{self.name}] {strategy.id} | "
                    f"score={self.severe_depth_penalty:.2f} | reason=branch_very_stale (turns={stale_turns})"
                )
                return self.severe_depth_penalty
            else:
                logger.info(
                    f"[{self.name}] {strategy.id} | "
                    f"score={self.depth_penalty:.2f} | reason=branch_stale (turns={stale_turns})"
                )
                return self.depth_penalty

        # Penalize connect_isolate on stale branches
        if strategy.id == "connect_isolate" and stale_turns >= self.stale_threshold:
            logger.info(
                f"[{self.name}] {strategy.id} | "
                f"score={self.connect_isolate_penalty:.2f} | reason=branch_stale (turns={stale_turns})"
            )
            return self.connect_isolate_penalty

        return 1.0

    def _get_stale_turn_count(self, context: ScoringContext) -> int:
        """Count how many turns the branch has been stale."""
        if not context.graph_state.active_branch:
            return 0

        # Get the focus node of the active branch
        branch_node = context.graph_state.active_branch[-1]
        branch_node_id = branch_node.id

        # Count consecutive turns without growth from the end
        stale_count = 0
        recent_turns = list(reversed(context.history.turns))

        for turn in recent_turns:
            has_growth = False
            for source_id, target_id in turn.extracted_edges:
                if source_id == branch_node_id or target_id == branch_node_id:
                    has_growth = True
                    break

            if has_growth:
                break  # Stop counting at first turn with growth

            stale_count += 1

        return stale_count


class CoverageQualityScorer(StrategyScorer):
    """
    Distinguishes knowledge lack from exploration lack.

    Adjusts coverage urgency based on whether gap is due to:
    - Knowledge lack: respondent doesn't know about topic (reduce urgency)
    - Exploration lack: we haven't asked yet (maintain urgency)

    Problem Fixed: Treating "don't know" as "need to explore more"
    """

    name = "coverage_quality"
    knowledge_lack_penalty: float = 0.4
    exploration_boost: float = 1.2
    first_touch_boost: float = 2.5
    exhaustion_threshold: int = 2
    exhaustion_penalty: float = 0.15

    def __init__(
        self,
        penalty: float = 0.4,
        boost: float = 1.2,
        first_touch_boost: float = 2.5,
        exhaustion_threshold: int = 2,
        exhaustion_penalty: float = 0.15
    ):
        self.knowledge_lack_penalty = penalty
        self.exploration_boost = boost
        self.first_touch_boost = first_touch_boost
        self.exhaustion_threshold = exhaustion_threshold
        self.exhaustion_penalty = exhaustion_penalty

    def score(
        self,
        strategy: "Strategy",
        focus: "FocusTarget",
        context: ScoringContext
    ) -> float:
        # Only applies to ensure_coverage strategy
        if strategy.id != "ensure_coverage":
            return 1.0

        if not focus.coverage_gap:
            return 1.0

        element_id = focus.coverage_gap.element_id

        # Check focus count for this element
        focus_count = context.coverage_state.get_focus_count(element_id)

        # PRIORITY 1: First touch - never been explored
        if focus_count == 0:
            logger.info(
                f"[{self.name}] {strategy.id} | "
                f"score={self.first_touch_boost:.2f} | reason=first_touch for '{element_id}'"
            )
            return self.first_touch_boost

        # PRIORITY 2: Check for exhaustion - probed multiple times without producing edges
        if focus_count >= self.exhaustion_threshold:
            if not self._produced_edges_recently(element_id, context):
                logger.info(
                    f"[{self.name}] {strategy.id} | "
                    f"score={self.exhaustion_penalty:.2f} | reason=exhausted '{element_id}' (focus_count={focus_count}, no edges)"
                )
                return self.exhaustion_penalty

        # Check if this is knowledge lack or exploration lack
        is_knowledge_lack = self._is_knowledge_lack(element_id, context)

        if is_knowledge_lack:
            logger.info(
                f"[{self.name}] {strategy.id} | "
                f"score={self.knowledge_lack_penalty:.2f} | reason=knowledge_lack for '{element_id}'"
            )
            return self.knowledge_lack_penalty

        return 1.0

    def _produced_edges_recently(self, element_id: str, context: ScoringContext) -> bool:
        """Check if probing this element produced new edges in recent turns."""
        for turn in context.history.turns:
            focus_element = turn.metadata.get("focus_element_id")
            if focus_element == element_id and turn.extracted_edges:
                return True
        return False

    def _is_knowledge_lack(
        self,
        element_id: str,
        context: ScoringContext
    ) -> bool:
        """Check if coverage gap is due to knowledge lack."""
        # Use KnowledgeCeilingScorer's patterns
        ceiling_patterns = KnowledgeCeilingScorer.ceiling_patterns

        # Check all turns for this element
        for turn in context.history.turns:
            response = turn.response.lower()

            # Check if response mentions the element
            if element_id.lower() in response:
                for pattern in ceiling_patterns:
                    if pattern in response:
                        return True

        return False


class SchemaTensionReadinessScorer(StrategyScorer):
    """
    Controls timing of schema tension resolution.

    Boosts resolve_schema_tension ONLY when both nodes in the invalid edge
    have been explored at least once. Penalizes premature tension exploration.
    """

    name = "schema_tension_readiness"

    # Parameters
    readiness_boost: float = 1.6  # Boost when both nodes explored
    premature_penalty: float = 0.4  # Penalty when exploring too early
    min_exploration_count: int = 1  # Minimum times each node must be explored

    def __init__(
        self,
        readiness_boost: float = 1.6,
        premature_penalty: float = 0.4,
        min_exploration: int = 1
    ):
        self.readiness_boost = readiness_boost
        self.premature_penalty = premature_penalty
        self.min_exploration_count = min_exploration

    def score(self, strategy, focus, context):
        if strategy.id != "resolve_schema_tension":
            return 1.0

        if not focus.node_pair:
            return 1.0

        node1, node2 = focus.node_pair

        # Check exploration counts for both nodes
        count1 = self._get_node_exploration_count(node1.id, context)
        count2 = self._get_node_exploration_count(node2.id, context)

        both_explored = (
            count1 >= self.min_exploration_count and
            count2 >= self.min_exploration_count
        )

        if both_explored:
            logger.info(
                f"[schema_tension_readiness] resolve_schema_tension | "
                f"score={self.readiness_boost:.2f} | "
                f"reason=both_nodes_explored ('{node1.label}'={count1}, '{node2.label}'={count2})"
            )
            return self.readiness_boost
        else:
            logger.info(
                f"[schema_tension_readiness] resolve_schema_tension | "
                f"score={self.premature_penalty:.2f} | "
                f"reason=premature_exploration ('{node1.label}'={count1}, '{node2.label}'={count2})"
            )
            return self.premature_penalty

    def _get_node_exploration_count(self, node_id: str, context: ScoringContext) -> int:
        """Count how many times a node has been the focus of a question."""
        count = 0
        for turn in context.history.turns:
            focus_node_id = turn.metadata.get("focus_node_id")
            if focus_node_id == node_id:
                count += 1
        return count


class ReflectionModeScorer(StrategyScorer):
    """
    Detects when interview should enter reflection/closing mode.

    Triggers when:
    - Coverage gaps = 0 (all elements touched)
    - No new nodes for N turns
    - Value nodes exist in graph

    Applies heavy penalty to depth strategies and boosts reflection strategies.
    """

    name = "reflection_mode"

    # Parameters
    no_new_nodes_threshold: int = 3
    depth_penalty: float = 0.2
    reflection_boost: float = 2.0
    breadth_boost_in_reflection: float = 1.3
    min_value_nodes: int = 1

    # Strategy classifications
    depth_strategies = ["deepen_branch", "resolve_schema_tension", "connect_isolate"]
    reflection_strategies = ["introduce_seed"]

    def __init__(
        self,
        no_new_nodes_threshold: int = 3,
        depth_penalty: float = 0.2,
        reflection_boost: float = 2.0,
        min_value_nodes: int = 1
    ):
        self.no_new_nodes_threshold = no_new_nodes_threshold
        self.depth_penalty = depth_penalty
        self.reflection_boost = reflection_boost
        self.min_value_nodes = min_value_nodes

    def score(self, strategy, focus, context):
        if not self._should_enter_reflection_mode(context):
            return 1.0

        if strategy.id in self.depth_strategies:
            logger.info(
                f"[reflection_mode] {strategy.id} | "
                f"score={self.depth_penalty:.2f} | reason=reflection_mode_active"
            )
            return self.depth_penalty

        if strategy.id in self.reflection_strategies:
            logger.info(
                f"[reflection_mode] {strategy.id} | "
                f"score={self.reflection_boost:.2f} | reason=reflection_mode_active"
            )
            return self.reflection_boost

        if strategy.id == "explore_breadth":
            return 1.3  # Slight boost

        return 1.0

    def _should_enter_reflection_mode(self, context: ScoringContext) -> bool:
        """Check if all conditions for reflection mode are met."""
        # Condition 1: Coverage gaps = 0
        if context.coverage_state.gaps:
            return False

        # Condition 2: No new nodes for N turns
        if not self._no_new_nodes_recently(context):
            return False

        # Condition 3: Value nodes exist
        if not self._has_value_nodes(context):
            return False

        logger.info(
            f"[reflection_mode] TRIGGERED | "
            f"coverage_complete=True, no_new_nodes_turns>={self.no_new_nodes_threshold}, "
            f"value_nodes>={self.min_value_nodes}"
        )
        return True

    def _no_new_nodes_recently(self, context: ScoringContext) -> bool:
        """Check if no new nodes were extracted in recent turns."""
        recent_turns = context.history.get_recent(self.no_new_nodes_threshold)
        for turn in recent_turns:
            if turn.extracted_nodes:
                return False
        return True

    def _has_value_nodes(self, context: ScoringContext) -> bool:
        """Check if graph has value-type nodes."""
        value_types = ["value", "belief", "goal", "need", "motivation"]
        value_count = 0
        for node in context.graph.nodes.values():
            if node.node_type and any(t in node.node_type.lower() for t in value_types):
                value_count += 1
        return value_count >= self.min_value_nodes


class ArbitrationEngine:
    """
    Orchestrates all scorers to select the best strategy.

    Collects all applicable strategies, scores each with all scorers,
    and returns the highest utility option.
    """

    def __init__(
        self,
        scorers: List[StrategyScorer],
        weights: Optional[Dict[str, float]] = None
    ):
        self.scorers = scorers
        self.weights = weights or {}

    def select_best(
        self,
        candidates: List[Tuple["Strategy", "FocusTarget"]],
        context: ScoringContext
    ) -> Tuple[float, "Strategy", "FocusTarget"]:
        """
        Score all candidates and return the highest utility option.

        Args:
            candidates: List of (strategy, focus) pairs
            context: Scoring context

        Returns:
            Tuple of (score, strategy, focus) for the winner
        """
        if not candidates:
            raise ValueError("No candidates to evaluate")

        scored = []

        for strategy, focus in candidates:
            total_score = 1.0
            score_details = []
            weighted_details = []

            for scorer in self.scorers:
                # Get weight for this scorer
                weight = self.weights.get(scorer.name, 1.0)

                # Get raw score
                raw_score = scorer.score(strategy, focus, context)

                # Apply weight (weighted average would require different math)
                # Using weighted product: score^weight
                weighted_score = raw_score ** weight if weight != 1.0 else raw_score

                total_score *= weighted_score
                score_details.append(f"{scorer.name}={raw_score:.2f}")

                # Add weighted score detail if weight is not 1.0
                if weight != 1.0:
                    weighted_details.append(
                        f"{scorer.name}={raw_score:.2f}^{weight:.1f}={weighted_score:.3f}"
                    )

            scored.append((total_score, strategy, focus, score_details))

            # Build enhanced log message
            log_parts = [
                f"[Arbitration] {strategy.id} -> {focus.describe()}:",
                f"total={total_score:.3f}",
                f"| raw=[{', '.join(score_details)}]"
            ]
            if weighted_details:
                log_parts.append(f"| weighted=[{', '.join(weighted_details)}]")

            logger.debug(" ".join(log_parts))

        # Sort by score (descending) and pick winner
        scored.sort(key=lambda x: x[0], reverse=True)
        winner = scored[0]

        logger.info(
            f"[Arbitration] Selected {winner[1].id} "
            f"(utility score: {winner[0]:.3f})"
        )

        # Log runner-up for context
        if len(scored) > 1:
            runner_up = scored[1]
            logger.debug(
                f"[Arbitration] Runner-up: {runner_up[1].id} "
                f"(utility score: {runner_up[0]:.3f})"
            )

        return winner[0], winner[1], winner[2]

    @classmethod
    def create_default(cls) -> "ArbitrationEngine":
        """Create engine with all default scorers."""
        scorers = [
            RedundancyScorer(),
            KnowledgeCeilingScorer(),
            MomentumAlignmentScorer(),
            RecencyDiversityScorer(),
            VerticalLadderingScorer(),
            BranchHealthScorer(),
            CoverageQualityScorer(),
            SchemaTensionReadinessScorer(),
            ReflectionModeScorer(),
        ]
        return cls(scorers)

    @classmethod
    def from_config(cls, config: Dict) -> "ArbitrationEngine":
        """
        Create engine from configuration dict.

        Config format:
        {
            "scorers": {
                "redundancy": {"weight": 1.0, "threshold": 0.85},
                "knowledge_ceiling": {"weight": 1.0, "lookback_turns": 3},
                ...
            }
        }
        """
        scorer_configs = config.get("scorers", {})
        weights = {}
        scorers = []

        # Create each configured scorer
        for scorer_name, scorer_config in scorer_configs.items():
            weight = scorer_config.get("weight", 1.0)
            weights[scorer_name] = weight

            if scorer_name == "redundancy":
                scorers.append(RedundancyScorer(
                    threshold=scorer_config.get("threshold", 0.85),
                    penalty=scorer_config.get("penalty", 0.2)
                ))

            elif scorer_name == "knowledge_ceiling":
                scorers.append(KnowledgeCeilingScorer(
                    lookback=scorer_config.get("lookback_turns", 3),
                    penalty=scorer_config.get("penalty", 0.1)
                ))

            elif scorer_name == "momentum_alignment":
                scorers.append(MomentumAlignmentScorer(
                    breadth_boost=scorer_config.get("breadth_boost", 1.5),
                    depth_penalty=scorer_config.get("depth_penalty", 0.5),
                    depth_boost=scorer_config.get("depth_boost", 1.3)
                ))

            elif scorer_name == "recency_diversity":
                scorers.append(RecencyDiversityScorer(
                    lookback=scorer_config.get("lookback_turns", 2),
                    penalty=scorer_config.get("penalty", 0.7)
                ))

            elif scorer_name == "vertical_laddering":
                scorers.append(VerticalLadderingScorer(
                    boost=scorer_config.get("boost", 1.5),
                    value_proximity_boost=scorer_config.get("value_proximity_boost", 1.8),
                    value_closure_boost=scorer_config.get("value_closure_boost", 2.0),
                    near_value_depth=scorer_config.get("near_value_depth", 2)
                ))

            elif scorer_name == "branch_health":
                scorers.append(BranchHealthScorer(
                    stale_threshold=scorer_config.get("stale_threshold", 2),
                    breadth_boost=scorer_config.get("breadth_boost", 1.8),
                    depth_penalty=scorer_config.get("depth_penalty", 0.3),
                    severe_stale_threshold=scorer_config.get("severe_stale_threshold", 4),
                    severe_depth_penalty=scorer_config.get("severe_depth_penalty", 0.1),
                    connect_isolate_penalty=scorer_config.get("connect_isolate_penalty", 0.5)
                ))

            elif scorer_name == "coverage_quality":
                scorers.append(CoverageQualityScorer(
                    penalty=scorer_config.get("penalty", 0.4),
                    boost=scorer_config.get("boost", 1.2),
                    first_touch_boost=scorer_config.get("first_touch_boost", 2.5),
                    exhaustion_threshold=scorer_config.get("exhaustion_threshold", 2),
                    exhaustion_penalty=scorer_config.get("exhaustion_penalty", 0.15)
                ))

            elif scorer_name == "schema_tension_readiness":
                scorers.append(SchemaTensionReadinessScorer(
                    readiness_boost=scorer_config.get("readiness_boost", 1.6),
                    premature_penalty=scorer_config.get("premature_penalty", 0.4),
                    min_exploration=scorer_config.get("min_exploration", 1)
                ))

            elif scorer_name == "reflection_mode":
                scorers.append(ReflectionModeScorer(
                    no_new_nodes_threshold=scorer_config.get("no_new_nodes_threshold", 3),
                    depth_penalty=scorer_config.get("depth_penalty", 0.2),
                    reflection_boost=scorer_config.get("reflection_boost", 2.0),
                    min_value_nodes=scorer_config.get("min_value_nodes", 1)
                ))

        # If no scorers configured, use defaults
        if not scorers:
            return cls.create_default()

        return cls(scorers, weights)
