"""
Computed interview state.
Assessed after each turn to provide inputs for strategy selection.
"""

from typing import Dict, List, Optional, Literal, TYPE_CHECKING, Any
from dataclasses import field
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.graph import Node

from core.graph import Graph, Node, Edge
from core.schema import Schema
from core.history import Turn


class Momentum(BaseModel):
    """Assessment of respondent engagement level with history tracking."""

    level: Literal["high", "neutral", "low"] = Field(
        default="neutral",
        description="Overall engagement level"
    )
    indicators: List[str] = Field(
        default_factory=list,
        description="Signals that led to this assessment"
    )

    # History tracking for fatigue detection
    history: List[tuple] = Field(
        default_factory=list,
        description="List of (turn_number, level) tuples"
    )
    consecutive_low_count: int = Field(
        default=0,
        description="Number of consecutive low momentum turns"
    )
    fatigue_threshold: int = Field(
        default=3,
        description="Consecutive low turns to trigger fatigue"
    )
    llm_response: Optional[Any] = Field(
        default=None,
        description="LLMResponse object for token tracking"
    )

    def record(self, turn_number: int) -> None:
        """Record this momentum reading in history."""
        self.history.append((turn_number, self.level))
        if self.level == "low":
            self.consecutive_low_count += 1
        else:
            self.consecutive_low_count = 0

    def is_fatigued(self) -> bool:
        """Check if sustained low engagement detected."""
        return self.consecutive_low_count >= self.fatigue_threshold

    @classmethod
    def default(cls) -> "Momentum":
        """Create default neutral momentum."""
        return cls(level="neutral", indicators=[])


class NodeFocusTracker(BaseModel):
    """
    Tracks how many times each node has been the focus target.
    Prevents strategy stuck loops by exhausting frequently-focused nodes.
    """

    focus_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="node_id -> number of times focused"
    )
    exhaustion_threshold: int = Field(
        default=2,
        description="Times a node can be focused before exhaustion"
    )
    exhausted_nodes: List[str] = Field(
        default_factory=list,
        description="Node IDs that have been exhausted"
    )

    def record_focus(self, node_id: str) -> None:
        """Record that a node was focused on."""
        self.focus_counts[node_id] = self.focus_counts.get(node_id, 0) + 1
        if self.focus_counts[node_id] >= self.exhaustion_threshold:
            if node_id not in self.exhausted_nodes:
                self.exhausted_nodes.append(node_id)

    def is_exhausted(self, node_id: str) -> bool:
        """Check if a node is exhausted."""
        return node_id in self.exhausted_nodes

    def get_focus_count(self, node_id: str) -> int:
        """Get focus count for a node."""
        return self.focus_counts.get(node_id, 0)

    def filter_non_exhausted(self, nodes: List["Node"]) -> List["Node"]:
        """Filter out exhausted nodes from a list."""
        return [n for n in nodes if n.id not in self.exhausted_nodes]


class EdgeFocusTracker(BaseModel):
    """Tracks focus on edges to prevent stuck loops on invalid edges."""

    focus_counts: Dict[str, int] = Field(default_factory=dict)
    exhaustion_threshold: int = Field(default=2)
    exhausted_edges: List[str] = Field(default_factory=list)

    def record_focus(self, edge_id: str) -> None:
        self.focus_counts[edge_id] = self.focus_counts.get(edge_id, 0) + 1
        if self.focus_counts[edge_id] >= self.exhaustion_threshold:
            if edge_id not in self.exhausted_edges:
                self.exhausted_edges.append(edge_id)

    def is_exhausted(self, edge_id: str) -> bool:
        return edge_id in self.exhausted_edges

    def get_focus_count(self, edge_id: str) -> int:
        return self.focus_counts.get(edge_id, 0)

    def filter_non_exhausted(self, edges: List["Edge"]) -> List["Edge"]:
        return [e for e in edges if e.id not in self.exhausted_edges]


class GraphState(BaseModel):
    """
    Computed state of the knowledge graph.
    Recalculated after each turn.
    """
    
    isolated_nodes: List[Node] = Field(
        default_factory=list,
        description="Nodes with no edges"
    )
    ambiguous_nodes: List[Node] = Field(
        default_factory=list,
        description="Nodes flagged for clarification"
    )
    invalid_edges: List[Edge] = Field(
        default_factory=list,
        description="Edges that violate schema rules"
    )
    active_branch: Optional[List[Node]] = Field(
        default=None,
        description="Current line of exploration (most recent path)"
    )
    branch_depth: int = Field(
        default=0,
        description="Depth of active branch"
    )
    unexplored_nodes: List[Node] = Field(
        default_factory=list,
        description="Nodes mentioned but not yet probed"
    )
    terminal_nodes: List[Node] = Field(
        default_factory=list,
        description="Nodes at terminal types (e.g., values)"
    )
    
    # Metrics
    total_nodes: int = Field(default=0)
    total_edges: int = Field(default=0)
    isolation_ratio: float = Field(default=0.0)
    
    @classmethod
    def compute(
        cls,
        graph: Graph,
        schema: Schema,
        history: List[Turn]
    ) -> "GraphState":
        """
        Compute current graph state.
        
        Args:
            graph: Current knowledge graph
            schema: Methodology schema
            history: Conversation history
            
        Returns:
            Computed GraphState
        """
        # Basic queries
        isolated = graph.get_isolated_nodes()
        ambiguous = graph.get_ambiguous_nodes()
        
        # Find invalid edges
        invalid_edges = []
        for edge in graph.edges.values():
            source = graph.get_node(edge.source_id)
            target = graph.get_node(edge.target_id)
            if source and target and source.node_type and target.node_type:
                if not schema.is_valid_edge(
                    source.node_type,
                    target.node_type,
                    edge.relation_type
                ):
                    invalid_edges.append(edge)
        
        # Find terminal nodes
        terminal = [
            node for node in graph.nodes.values()
            if node.node_type and schema.is_terminal_type(node.node_type)
        ]
        
        # Compute active branch (path from most recent node)
        active_branch = None
        branch_depth = 0
        if history:
            recent_turn = history[-1]
            if recent_turn.extracted_nodes:
                # Start from most recent node
                recent_node_id = recent_turn.extracted_nodes[-1]
                active_branch = cls._trace_branch(graph, recent_node_id)
                branch_depth = len(active_branch) if active_branch else 0
        
        # Find unexplored nodes (have no outgoing probes in history)
        explored_node_ids = set()
        for turn in history:
            # Nodes that were focus of a question are "explored"
            focus_node_id = turn.metadata.get("focus_node_id")
            if focus_node_id:
                explored_node_ids.add(focus_node_id)
        
        unexplored = [
            node for node in graph.nodes.values()
            if node.id not in explored_node_ids
            and not node.is_ambiguous
            and node.id not in [n.id for n in isolated]
        ]
        
        return cls(
            isolated_nodes=isolated,
            ambiguous_nodes=ambiguous,
            invalid_edges=invalid_edges,
            active_branch=active_branch,
            branch_depth=branch_depth,
            unexplored_nodes=unexplored,
            terminal_nodes=terminal,
            total_nodes=len(graph.nodes),
            total_edges=len(graph.edges),
            isolation_ratio=graph.compute_isolation_ratio()
        )
    
    @staticmethod
    def _trace_branch(graph: Graph, start_node_id: str) -> List[Node]:
        """
        Trace the branch containing a node.
        Follows incoming edges to find the path to this node.
        """
        branch = []
        current_id = start_node_id
        visited = set()
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = graph.get_node(current_id)
            if node:
                branch.insert(0, node)  # Prepend to get root-to-leaf order
            
            # Follow first incoming edge (simplification)
            incoming = graph.get_incoming_edges(current_id)
            if incoming:
                current_id = incoming[0].source_id
            else:
                break
        
        return branch
    
    def has_structural_issues(self) -> bool:
        """Check if graph has any structural problems requiring healing."""
        return bool(
            self.isolated_nodes or 
            self.ambiguous_nodes or 
            self.invalid_edges
        )
    
    def get_priority_issue(self) -> Optional[str]:
        """Get the highest priority structural issue type."""
        if self.ambiguous_nodes:
            return "ambiguity"
        if self.isolated_nodes:
            return "isolation"
        if self.invalid_edges:
            return "invalid_edge"
        return None


class CoverageRequirements(BaseModel):
    """Requirements for coverage of a reference element."""
    
    mention: bool = Field(
        default=True,
        description="Element must be mentioned"
    )
    reaction: bool = Field(
        default=True,
        description="Must capture evaluative stance"
    )
    comprehension: bool = Field(
        default=False,
        description="Must verify understanding"
    )
    connections_to: List[str] = Field(
        default_factory=list,
        description="Must be connected to these other elements"
    )


class ReferenceElement(BaseModel):
    """A key element from the stimulus concept."""

    id: str = Field(description="Element identifier (e.g., 'insight', 'promise', 'rtb')")
    content: str = Field(description="Actual text from concept")
    element_type: str = Field(
        default="unknown",
        description="Element type: problem, solution, evidence, etc."
    )
    requirements: CoverageRequirements = Field(
        default_factory=CoverageRequirements
    )


class CoverageGap(BaseModel):
    """A specific coverage requirement that hasn't been met."""
    
    element_id: str = Field(description="Which element has the gap")
    gap_type: Literal["unmentioned", "no_reaction", "no_comprehension", "unconnected"] = Field(
        description="Type of coverage gap"
    )
    target_element: Optional[str] = Field(
        default=None,
        description="For 'unconnected' gaps, which element should it connect to"
    )


class CoverageState(BaseModel):
    """
    Tracks coverage of stimulus concept elements.
    Updated after each graph update.
    """

    reference_elements: Dict[str, ReferenceElement] = Field(
        default_factory=dict,
        description="Elements parsed from concept"
    )
    element_node_mappings: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="element_id -> list of node_ids that map to it"
    )
    element_reactions: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="element_id -> reaction (positive/negative/neutral/None)"
    )
    element_comprehension: Dict[str, bool] = Field(
        default_factory=dict,
        description="element_id -> comprehension verified"
    )
    gaps: List[CoverageGap] = Field(
        default_factory=list,
        description="Current coverage gaps"
    )
    # Topic exhaustion tracking
    element_focus_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="element_id -> number of times this element was the focus target"
    )
    exhausted_elements: List[str] = Field(
        default_factory=list,
        description="Element IDs that have been exhausted (asked about enough times)"
    )
    exhaustion_threshold: int = Field(
        default=3,
        description="Number of consecutive focuses before element is considered exhausted"
    )
    
    @classmethod
    def initialize(
        cls,
        concept_text: str,
        element_config: Optional[Dict] = None
    ) -> "CoverageState":
        """
        Initialize coverage state from concept text.
        
        In practice, this would use LLM to parse concept into elements.
        For now, accepts optional manual element config.
        
        Args:
            concept_text: The stimulus concept text
            element_config: Optional dict mapping element_id to content
            
        Returns:
            Initialized CoverageState
        """
        state = cls()
        
        if element_config:
            for element_id, config in element_config.items():
                if isinstance(config, str):
                    # Simple: just content string
                    state.reference_elements[element_id] = ReferenceElement(
                        id=element_id,
                        content=config
                    )
                elif isinstance(config, dict):
                    # Full config with requirements and element_type
                    requirements = CoverageRequirements(
                        **config.get("requirements", {})
                    )
                    state.reference_elements[element_id] = ReferenceElement(
                        id=element_id,
                        content=config.get("content", ""),
                        element_type=config.get("element_type", "unknown"),
                        requirements=requirements
                    )
                
                # Initialize tracking
                state.element_node_mappings[element_id] = []
                state.element_reactions[element_id] = None
                state.element_comprehension[element_id] = False
        
        # Compute initial gaps
        state._recompute_gaps()
        
        return state
    
    def update(
        self,
        graph: Graph,
        node_element_mappings: Dict[str, str]
    ) -> None:
        """
        Update coverage state after graph update.
        
        Args:
            graph: Current graph
            node_element_mappings: node_id -> element_id for new nodes
        """
        # Update mappings
        for node_id, element_id in node_element_mappings.items():
            if element_id in self.element_node_mappings:
                if node_id not in self.element_node_mappings[element_id]:
                    self.element_node_mappings[element_id].append(node_id)
        
        # Recompute gaps
        self._recompute_gaps(graph)
    
    def record_reaction(self, element_id: str, reaction: str) -> None:
        """Record a reaction to an element."""
        if element_id in self.element_reactions:
            self.element_reactions[element_id] = reaction
            self._recompute_gaps()
    
    def record_comprehension(self, element_id: str, verified: bool = True) -> None:
        """Record comprehension verification for an element."""
        if element_id in self.element_comprehension:
            self.element_comprehension[element_id] = verified
            self._recompute_gaps()
    
    def _recompute_gaps(self, graph: Optional[Graph] = None) -> None:
        """Recompute coverage gaps based on current state."""
        self.gaps = []
        
        for element_id, element in self.reference_elements.items():
            reqs = element.requirements
            
            # Check mention
            if reqs.mention:
                if not self.element_node_mappings.get(element_id):
                    self.gaps.append(CoverageGap(
                        element_id=element_id,
                        gap_type="unmentioned"
                    ))
                    continue  # Can't check other requirements if not mentioned
            
            # Check reaction
            if reqs.reaction:
                if self.element_reactions.get(element_id) is None:
                    self.gaps.append(CoverageGap(
                        element_id=element_id,
                        gap_type="no_reaction"
                    ))
            
            # Check comprehension
            if reqs.comprehension:
                if not self.element_comprehension.get(element_id):
                    self.gaps.append(CoverageGap(
                        element_id=element_id,
                        gap_type="no_comprehension"
                    ))
            
            # Check connections
            if reqs.connections_to and graph:
                element_nodes = self.element_node_mappings.get(element_id, [])
                
                for target_element_id in reqs.connections_to:
                    target_nodes = self.element_node_mappings.get(target_element_id, [])
                    
                    # Check if any edge exists between element nodes and target nodes
                    connected = False
                    for source_id in element_nodes:
                        for target_id in target_nodes:
                            if (graph.get_edge_between(source_id, target_id) or
                                graph.get_edge_between(target_id, source_id)):
                                connected = True
                                break
                        if connected:
                            break
                    
                    if not connected:
                        self.gaps.append(CoverageGap(
                            element_id=element_id,
                            gap_type="unconnected",
                            target_element=target_element_id
                        ))
    
    def is_satisfied(self) -> bool:
        """Check if all coverage requirements are met."""
        return len(self.gaps) == 0
    
    def get_gaps(self) -> List[CoverageGap]:
        """Get current coverage gaps."""
        return self.gaps

    def get_non_exhausted_gaps(self) -> List[CoverageGap]:
        """Get coverage gaps excluding exhausted elements."""
        return [g for g in self.gaps if g.element_id not in self.exhausted_elements]

    def record_element_focus(self, element_id: str) -> None:
        """Record that an element was focused on this turn."""
        if element_id not in self.element_focus_counts:
            self.element_focus_counts[element_id] = 0
        self.element_focus_counts[element_id] += 1

        # Check if exhaustion threshold reached
        if self.element_focus_counts[element_id] >= self.exhaustion_threshold:
            if element_id not in self.exhausted_elements:
                self.exhausted_elements.append(element_id)

    def reset_focus_count(self, element_id: str) -> None:
        """Reset focus count when switching to different element."""
        # Reset counts for OTHER elements (topic switch resets staleness)
        for eid in self.element_focus_counts:
            if eid != element_id:
                self.element_focus_counts[eid] = 0

    def is_element_exhausted(self, element_id: str) -> bool:
        """Check if an element has been exhausted."""
        return element_id in self.exhausted_elements

    def get_focus_count(self, element_id: str) -> int:
        """Get current focus count for an element."""
        return self.element_focus_counts.get(element_id, 0)
    
    def get_gaps_by_type(self, gap_type: str) -> List[CoverageGap]:
        """Get gaps of a specific type."""
        return [g for g in self.gaps if g.gap_type == gap_type]
    
    def get_element_status(self, element_id: str) -> Dict:
        """Get detailed status for an element."""
        element = self.reference_elements.get(element_id)
        if not element:
            return {}
        
        return {
            "id": element_id,
            "content": element.content,
            "mapped_nodes": self.element_node_mappings.get(element_id, []),
            "reaction": self.element_reactions.get(element_id),
            "comprehension_verified": self.element_comprehension.get(element_id, False),
            "gaps": [g for g in self.gaps if g.element_id == element_id]
        }
    
    def summary(self) -> str:
        """Human-readable coverage summary."""
        lines = []
        lines.append(f"Coverage: {len(self.reference_elements)} elements, {len(self.gaps)} gaps")
        lines.append("")
        
        for element_id, element in self.reference_elements.items():
            status = self.get_element_status(element_id)
            node_count = len(status["mapped_nodes"])
            reaction = status["reaction"] or "none"
            
            line = f"  • {element_id}: {node_count} nodes, reaction={reaction}"
            if status["gaps"]:
                gap_types = [g.gap_type for g in status["gaps"]]
                line += f" [GAPS: {', '.join(gap_types)}]"
            lines.append(line)
        
        return "\n".join(lines)
