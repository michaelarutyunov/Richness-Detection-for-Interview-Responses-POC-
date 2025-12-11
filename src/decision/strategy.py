"""
Strategy definitions and selection logic.
The core decision engine for interview flow.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
from pathlib import Path
from pydantic import BaseModel, Field
import yaml

logger = logging.getLogger(__name__)

from core.graph import Graph, Node, Edge
from core.state import (
    GraphState,
    CoverageState,
    CoverageGap,
    ReferenceElement,
    Momentum,
    NodeFocusTracker,
    EdgeFocusTracker
)
from core.history import History

if TYPE_CHECKING:
    from decision.arbitration import ArbitrationEngine, ScoringContext


class Tactic(BaseModel):
    """
    A conversational move - describes HOW to probe.
    Loaded from interview_logic.yaml.
    """
    
    id: str = Field(description="Tactic identifier")
    description: str = Field(description="How to execute this conversational move")
    
    @classmethod
    def load_all(cls, path: str) -> Dict[str, "Tactic"]:
        """Load all tactics from YAML file."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        tactics = {}
        raw_tactics = data.get('tactics', {})
        for tactic_id, tactic_data in raw_tactics.items():
            tactics[tactic_id] = cls(
                id=tactic_id,
                description=tactic_data.get('description', '')
            )
        
        return tactics


class FocusTarget(BaseModel):
    """
    What a strategy is targeting.
    Varies by strategy type - could be a node, node pair, or concept element.
    """

    model_config = {"arbitrary_types_allowed": True}

    node: Optional[Node] = Field(default=None, description="Single node focus")
    node_pair: Optional[Tuple[Node, Node]] = Field(default=None, description="Pair of nodes")
    element: Optional[ReferenceElement] = Field(default=None, description="Concept element")
    coverage_gap: Optional[CoverageGap] = Field(default=None, description="Specific coverage gap")
    
    def describe(self) -> str:
        """Human-readable description for LLM prompt."""
        if self.node:
            desc = f"Node: '{self.node.label}'"
            if self.node.node_type:
                desc += f" (type: {self.node.node_type})"
            return desc
        
        if self.node_pair:
            n1, n2 = self.node_pair
            return f"Node pair: '{n1.label}' and '{n2.label}'"
        
        if self.element:
            return f"Concept element: {self.element.id} - '{self.element.content[:100]}...'" if len(self.element.content) > 100 else f"Concept element: {self.element.id} - '{self.element.content}'"
        
        if self.coverage_gap:
            gap = self.coverage_gap
            if gap.gap_type == "unconnected":
                return f"Coverage gap: {gap.element_id} not connected to {gap.target_element}"
            return f"Coverage gap: {gap.element_id} ({gap.gap_type})"
        
        return "No specific focus"
    
    @classmethod
    def from_node(cls, node: Node) -> "FocusTarget":
        return cls(node=node)
    
    @classmethod
    def from_node_pair(cls, node1: Node, node2: Node) -> "FocusTarget":
        return cls(node_pair=(node1, node2))
    
    @classmethod
    def from_element(cls, element: ReferenceElement) -> "FocusTarget":
        return cls(element=element)
    
    @classmethod
    def from_coverage_gap(cls, gap: CoverageGap, element: ReferenceElement) -> "FocusTarget":
        return cls(coverage_gap=gap, element=element)


class Strategy(BaseModel):
    """
    A graph-building strategy.
    Defines intent, conditions, and suggested tactics.
    """
    
    id: str = Field(description="Strategy identifier")
    intent: str = Field(description="What this strategy aims to achieve")
    applies_when: str = Field(description="Human-readable condition description")
    suggested_tactics: List[str] = Field(
        default_factory=list,
        description="Tactic IDs that can achieve this strategy"
    )
    llm_guidance: str = Field(
        default="",
        description="Guidance for LLM on how to execute"
    )
    
    def applies(
        self,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum
    ) -> bool:
        """
        Evaluate whether this strategy should fire.
        Implemented per strategy type via _check_* methods.
        """
        check_method = getattr(self, f"_check_{self.id}", None)
        if check_method:
            return check_method(graph_state, coverage_state, momentum)
        return False
    
    def get_focus(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState,
        node_focus_tracker: Optional[NodeFocusTracker] = None,
        edge_focus_tracker: Optional[EdgeFocusTracker] = None
    ) -> FocusTarget:
        """
        Select what to target for this strategy.
        Implemented per strategy type via _focus_* methods.
        """
        focus_method = getattr(self, f"_focus_{self.id}", None)
        if focus_method:
            # Pass trackers to methods that support them
            import inspect
            sig = inspect.signature(focus_method)
            kwargs = {}
            if 'node_focus_tracker' in sig.parameters:
                kwargs['node_focus_tracker'] = node_focus_tracker
            if 'edge_focus_tracker' in sig.parameters:
                kwargs['edge_focus_tracker'] = edge_focus_tracker
            return focus_method(graph, graph_state, coverage_state, **kwargs)
        return FocusTarget()
    
    # --- Strategy-specific condition checks ---
    
    def _check_ensure_coverage(
        self,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum
    ) -> bool:
        """Coverage gaps exist for non-exhausted reference elements."""
        # Only trigger if there are gaps for elements that haven't been exhausted
        return len(coverage_state.get_non_exhausted_gaps()) > 0
    
    def _check_resolve_ambiguity(
        self,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum
    ) -> bool:
        """A node is flagged as ambiguous."""
        return len(graph_state.ambiguous_nodes) > 0
    
    def _check_connect_isolate(
        self,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum
    ) -> bool:
        """A node exists with no edges. Skip if low momentum."""
        if not graph_state.isolated_nodes:
            return False
        # Skip if low momentum - probing requires engagement
        if momentum.level == "low":
            logger.info("[Strategy] Low momentum: deferring connect_isolate")
            return False
        return True
    
    def _check_resolve_schema_tension(
        self,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum
    ) -> bool:
        """An edge exists that violates schema rules."""
        return len(graph_state.invalid_edges) > 0
    
    def _check_deepen_branch(
        self,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum
    ) -> bool:
        """Active branch exists, momentum adequate, not at terminal."""
        if not graph_state.active_branch:
            return False
        if momentum.level == "low":
            return False
        # Check if last node in branch is terminal
        if graph_state.terminal_nodes:
            last_node = graph_state.active_branch[-1]
            if last_node.id in [n.id for n in graph_state.terminal_nodes]:
                return False
        return True
    
    def _check_explore_breadth(
        self,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum
    ) -> bool:
        """Current branch stalling or unexplored nodes available."""
        # Branch stalling (low momentum) or no active branch
        branch_stalling = momentum.level == "low" or not graph_state.active_branch
        has_unexplored = len(graph_state.unexplored_nodes) > 0
        return branch_stalling and has_unexplored
    
    def _check_introduce_seed(
        self,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum
    ) -> bool:
        """All branches exhausted, coverage complete - fallback strategy."""
        # This is the fallback - always applicable if nothing else fires
        # But prefer other strategies, so only true if:
        # - Coverage satisfied
        # - No unexplored nodes
        # - No structural issues
        coverage_ok = coverage_state.is_satisfied()
        no_unexplored = len(graph_state.unexplored_nodes) == 0
        no_issues = not graph_state.has_structural_issues()
        return coverage_ok and no_unexplored and no_issues
    
    # --- Strategy-specific focus selection ---
    
    def _focus_ensure_coverage(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState
    ) -> FocusTarget:
        """Focus on coverage gap, rotating to avoid exhaustion."""
        gaps = coverage_state.get_non_exhausted_gaps()
        if not gaps:
            return FocusTarget()

        # Sort gaps by focus count (least focused first) to rotate
        # This prevents always picking the same element
        sorted_gaps = sorted(
            gaps,
            key=lambda g: coverage_state.get_focus_count(g.element_id)
        )

        gap = sorted_gaps[0]
        element = coverage_state.reference_elements.get(gap.element_id)
        if element:
            # Record that we're focusing on this element
            coverage_state.record_element_focus(gap.element_id)
            logger.info(f"[Focus] Element '{gap.element_id}' focus count: {coverage_state.get_focus_count(gap.element_id)}")
            return FocusTarget.from_coverage_gap(gap, element)
        return FocusTarget()
    
    def _focus_resolve_ambiguity(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState
    ) -> FocusTarget:
        """Focus on most recent ambiguous node."""
        if graph_state.ambiguous_nodes:
            # Sort by timestamp, most recent first
            sorted_nodes = sorted(
                graph_state.ambiguous_nodes,
                key=lambda n: n.timestamp,
                reverse=True
            )
            return FocusTarget.from_node(sorted_nodes[0])
        return FocusTarget()
    
    def _focus_connect_isolate(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState,
        node_focus_tracker: Optional[NodeFocusTracker] = None
    ) -> FocusTarget:
        """Focus on isolated node, respecting exhaustion."""
        candidates = list(graph_state.isolated_nodes)

        # Filter out exhausted nodes if tracker provided
        if node_focus_tracker:
            candidates = node_focus_tracker.filter_non_exhausted(candidates)
            if not candidates:
                logger.info("[Focus] All isolated nodes exhausted")
                return FocusTarget()
            # Sort by focus count (least focused first), then by timestamp
            candidates = sorted(
                candidates,
                key=lambda n: (node_focus_tracker.get_focus_count(n.id), -n.timestamp.timestamp())
            )
        else:
            # Default: sort by timestamp (most recent first)
            candidates = sorted(candidates, key=lambda n: n.timestamp, reverse=True)

        if candidates:
            return FocusTarget.from_node(candidates[0])
        return FocusTarget()
    
    def _focus_resolve_schema_tension(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState,
        edge_focus_tracker: Optional[EdgeFocusTracker] = None
    ) -> FocusTarget:
        """Focus on an invalid edge that needs resolution."""
        candidates = list(graph_state.invalid_edges)

        if edge_focus_tracker:
            candidates = edge_focus_tracker.filter_non_exhausted(candidates)
            if not candidates:
                logger.info("[Focus] All invalid edges exhausted")
                return FocusTarget()
            candidates = sorted(
                candidates,
                key=lambda e: edge_focus_tracker.get_focus_count(e.id)
            )

        if candidates:
            edge = candidates[0]
            source = graph.get_node(edge.source_id)
            target = graph.get_node(edge.target_id)
            if source and target:
                return FocusTarget.from_node_pair(source, target)
        return FocusTarget()
    
    def _focus_deepen_branch(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState
    ) -> FocusTarget:
        """Focus on most recent node in active branch."""
        if graph_state.active_branch:
            return FocusTarget.from_node(graph_state.active_branch[-1])
        return FocusTarget()
    
    def _focus_explore_breadth(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState
    ) -> FocusTarget:
        """Focus on highest-potential unexplored node."""
        if graph_state.unexplored_nodes:
            # Prioritize by: connection count (more connected = more context)
            def score_node(node: Node) -> int:
                return len(graph.get_edges_for_node(node.id))
            
            sorted_nodes = sorted(
                graph_state.unexplored_nodes,
                key=score_node,
                reverse=True
            )
            return FocusTarget.from_node(sorted_nodes[0])
        return FocusTarget()
    
    def _focus_introduce_seed(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState
    ) -> FocusTarget:
        """No specific focus - open-ended."""
        return FocusTarget()


class StrategySelector(BaseModel):
    """
    Selects strategies based on current state.

    Supports two selection modes:
    - Legacy: Sequential threshold check - first applicable strategy wins
    - Arbitration: Utility-based scoring of all applicable strategies
    """

    model_config = {"arbitrary_types_allowed": True}

    strategies: List[Strategy] = Field(
        default_factory=list,
        description="Priority-ordered list of strategies"
    )
    tactics: Dict[str, Tactic] = Field(
        default_factory=dict,
        description="Available tactics"
    )
    use_arbitration: bool = Field(
        default=False,
        description="Whether to use utility-based arbitration"
    )
    _arbitration_engine: Optional["ArbitrationEngine"] = None

    def set_arbitration_engine(self, engine: "ArbitrationEngine") -> None:
        """Set the arbitration engine and enable arbitration mode."""
        self._arbitration_engine = engine
        self.use_arbitration = True
        logger.info("[StrategySelector] Arbitration mode enabled")
    
    @classmethod
    def load(cls, path: str, enable_arbitration: bool = True) -> "StrategySelector":
        """
        Load strategies and tactics from interview_logic.yaml.

        Args:
            path: Path to interview_logic.yaml
            enable_arbitration: Whether to enable arbitration if config found

        Returns:
            Configured StrategySelector
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Load tactics
        tactics = {}
        raw_tactics = data.get('tactics', {})
        for tactic_id, tactic_data in raw_tactics.items():
            tactics[tactic_id] = Tactic(
                id=tactic_id,
                description=tactic_data.get('description', '')
            )

        # Load strategies in priority order
        # The YAML order determines priority
        strategies = []
        raw_strategies = data.get('strategies', {})
        for strategy_id, strategy_data in raw_strategies.items():
            strategies.append(Strategy(
                id=strategy_id,
                intent=strategy_data.get('intent', ''),
                applies_when=strategy_data.get('applies_when', ''),
                suggested_tactics=strategy_data.get('suggested_tactics', []),
                llm_guidance=strategy_data.get('llm_guidance', '')
            ))

        selector = cls(strategies=strategies, tactics=tactics)

        # Load arbitration config if present and enabled
        arbitration_config = data.get('arbitration', {})
        if enable_arbitration and arbitration_config.get('enabled', False):
            from decision.arbitration import ArbitrationEngine
            engine = ArbitrationEngine.from_config(arbitration_config)
            selector.set_arbitration_engine(engine)
            logger.info("[StrategySelector] Loaded with arbitration enabled")

        return selector
    
    def select(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum,
        node_focus_tracker: Optional[NodeFocusTracker] = None,
        edge_focus_tracker: Optional[EdgeFocusTracker] = None,
        history: Optional[History] = None
    ) -> Tuple[Strategy, FocusTarget]:
        """
        Select best strategy and its focus.

        Uses either legacy (first-applicable-wins) or arbitration mode
        depending on configuration.

        Args:
            graph: Current knowledge graph
            graph_state: Computed graph state
            coverage_state: Coverage tracking state
            momentum: Current momentum assessment
            node_focus_tracker: Optional tracker for node focus exhaustion
            edge_focus_tracker: Optional tracker for edge focus exhaustion
            history: Optional conversation history (required for arbitration)

        Returns:
            Tuple of (selected strategy, focus target)
        """
        # Log input state for debugging with exhaustion info
        total_gaps = len(coverage_state.get_gaps())
        non_exhausted_gaps = len(coverage_state.get_non_exhausted_gaps())
        exhausted_count = len(coverage_state.exhausted_elements)

        logger.info(
            f"[Strategy Input] coverage_gaps={total_gaps} (non-exhausted={non_exhausted_gaps}, exhausted={exhausted_count}), "
            f"isolated={len(graph_state.isolated_nodes)}, "
            f"ambiguous={len(graph_state.ambiguous_nodes)}, "
            f"momentum={momentum.level}"
        )

        if coverage_state.exhausted_elements:
            logger.info(f"[Strategy Input] Exhausted elements: {coverage_state.exhausted_elements}")

        if node_focus_tracker and node_focus_tracker.exhausted_nodes:
            logger.info(f"[Strategy Input] Exhausted nodes: {node_focus_tracker.exhausted_nodes}")

        if edge_focus_tracker and edge_focus_tracker.exhausted_edges:
            logger.info(f"[Strategy Input] Exhausted edges: {edge_focus_tracker.exhausted_edges}")

        # Choose selection mode
        if self.use_arbitration and self._arbitration_engine and history:
            return self._select_arbitrated(
                graph, graph_state, coverage_state, momentum,
                node_focus_tracker, edge_focus_tracker, history
            )
        else:
            return self._select_legacy(
                graph, graph_state, coverage_state, momentum,
                node_focus_tracker, edge_focus_tracker
            )

    def _select_legacy(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum,
        node_focus_tracker: Optional[NodeFocusTracker] = None,
        edge_focus_tracker: Optional[EdgeFocusTracker] = None
    ) -> Tuple[Strategy, FocusTarget]:
        """
        Legacy selection: first applicable strategy wins.
        """
        for strategy in self.strategies:
            applies = strategy.applies(graph_state, coverage_state, momentum)
            if applies:
                focus = strategy.get_focus(graph, graph_state, coverage_state, node_focus_tracker, edge_focus_tracker)
                # Skip if focus is empty due to exhaustion
                if not focus.node and not focus.element and not focus.node_pair and not focus.coverage_gap:
                    logger.info(f"[Strategy Skip] {strategy.id}: all targets exhausted")
                    continue
                logger.info(f"[Strategy Selected] {strategy.id} -> {focus.describe()}")
                return strategy, focus
            else:
                logger.debug(f"[Strategy Skip] {strategy.id}: conditions not met")

        # Fallback - should not reach here if introduce_seed is last
        if self.strategies:
            fallback = self.strategies[-1]
            focus = fallback.get_focus(graph, graph_state, coverage_state)
            logger.warning(f"[Strategy Fallback] Using {fallback.id}")
            return fallback, focus

        # Emergency fallback
        logger.error("[Strategy] No strategies available - using emergency fallback")
        return Strategy(id="fallback", intent="Continue conversation", applies_when="always"), FocusTarget()

    def _select_arbitrated(
        self,
        graph: Graph,
        graph_state: GraphState,
        coverage_state: CoverageState,
        momentum: Momentum,
        node_focus_tracker: Optional[NodeFocusTracker],
        edge_focus_tracker: Optional[EdgeFocusTracker],
        history: History
    ) -> Tuple[Strategy, FocusTarget]:
        """
        Arbitrated selection: score all applicable strategies and pick best.
        """
        from decision.arbitration import ScoringContext

        # Collect all applicable (strategy, focus) pairs
        candidates = []
        for strategy in self.strategies:
            if strategy.applies(graph_state, coverage_state, momentum):
                focus = strategy.get_focus(
                    graph, graph_state, coverage_state,
                    node_focus_tracker, edge_focus_tracker
                )
                # Skip if focus is empty
                if not focus.node and not focus.element and not focus.node_pair and not focus.coverage_gap:
                    logger.debug(f"[Arbitration] Skipping {strategy.id}: all targets exhausted")
                    continue
                candidates.append((strategy, focus))
                logger.debug(f"[Arbitration] Candidate: {strategy.id} -> {focus.describe()}")

        # If no candidates, fall back to legacy selection
        if not candidates:
            logger.warning("[Arbitration] No valid candidates - falling back to legacy selection")
            return self._select_legacy(
                graph, graph_state, coverage_state, momentum,
                node_focus_tracker, edge_focus_tracker
            )

        # Build scoring context
        context = ScoringContext.build(
            graph=graph,
            graph_state=graph_state,
            coverage_state=coverage_state,
            momentum=momentum,
            history=history,
            node_focus_tracker=node_focus_tracker,
            edge_focus_tracker=edge_focus_tracker
        )

        # Let arbitration engine choose
        score, strategy, focus = self._arbitration_engine.select_best(candidates, context)
        logger.info(f"[Strategy Selected] {strategy.id} -> {focus.describe()} (arbitration score: {score:.3f})")
        return strategy, focus
    
    def get_tactic(self, tactic_id: str) -> Optional[Tactic]:
        """Get a tactic by ID."""
        return self.tactics.get(tactic_id)
    
    def get_tactics_for_strategy(self, strategy: Strategy) -> List[Tactic]:
        """Get all tactics suggested for a strategy."""
        return [
            self.tactics[tid] 
            for tid in strategy.suggested_tactics 
            if tid in self.tactics
        ]
    
    def format_tactics_for_prompt(self, strategy: Strategy) -> str:
        """Format suggested tactics for LLM prompt."""
        tactics = self.get_tactics_for_strategy(strategy)
        if not tactics:
            return "No specific tactics suggested."
        
        lines = []
        for tactic in tactics:
            lines.append(f"- **{tactic.id}**: {tactic.description}")
        
        return "\n".join(lines)
