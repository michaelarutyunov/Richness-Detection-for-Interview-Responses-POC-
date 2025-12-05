"""
Core data models for the graph-driven interview system.
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class NeedName(str, Enum):
    """Names of graph needs that can be detected."""
    BRIDGE_ISOLATION = "bridge_isolation"
    DEPTH_COMPLETION = "depth_completion"
    SEED_EXPANSION = "seed_expansion"
    DEAD_END_RESOLUTION = "dead_end_resolution"


class StrategyName(str, Enum):
    """Names of strategies for addressing graph needs."""
    BRIDGE_BUILDING = "bridge_building"
    DEPTH_COMPLETION = "depth_completion"
    SEED_EXPANSION = "seed_expansion"
    DEAD_END_RESOLUTION = "dead_end_resolution"




class Need(BaseModel):
    """Represents a structural need detected in the graph."""
    name: NeedName
    score: float = Field(ge=0.0, le=1.0, description="Priority score for this need")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the need")
    
    def __str__(self) -> str:
        return f"Need({self.name.value}, score={self.score:.2f})"


class Node(BaseModel):
    """Represents a concept node in the knowledge graph."""
    id: str
    label: str
    type: str
    creation_turn: int = Field(ge=0)
    visit_count: int = Field(ge=0, default=0)
    last_visit_turn: Optional[int] = None
    source_quotes: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    def increment_visit(self, turn_number: int) -> None:
        """Increment visit count and update last visit turn."""
        self.visit_count += 1
        self.last_visit_turn = turn_number


class Edge(BaseModel):
    """Represents a relationship between nodes in the knowledge graph."""
    id: str
    type: str
    source: str
    target: str
    creation_turn: int = Field(ge=0)
    source_quote: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GraphState(BaseModel):
    """Represents the current state of the knowledge graph."""
    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: Dict[str, Edge] = Field(default_factory=dict)
    turn_number: int = Field(ge=0, default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.id] = edge
    
    def get_isolated_nodes(self) -> List[Node]:
        """Find nodes with no incoming or outgoing edges."""
        connected_nodes: Set[str] = set()
        
        # Collect all nodes that are connected by edges
        for edge in self.edges.values():
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        # Return nodes that are not in the connected set
        return [node for node_id, node in self.nodes.items() if node_id not in connected_nodes]
    
    def get_node_count(self) -> int:
        """Get total number of nodes."""
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        """Get total number of edges."""
        return len(self.edges)
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.type == node_type]
    
    def get_average_depth(self) -> float:
        """Calculate average depth of the graph (simple metric)."""
        if not self.nodes:
            return 0.0

        # Calculate average depth across all nodes
        total_depth = 0
        for node_id in self.nodes:
            depth = self._calculate_node_depth(node_id, visited=set())
            total_depth += depth

        return float(total_depth) / len(self.nodes)
    
    def _calculate_node_depth(self, node_id: str, visited: Set[str], depth: int = 0) -> int:
        """Calculate depth from a given node using DFS."""
        if node_id in visited:
            return depth
        
        visited.add(node_id)
        max_depth = depth
        
        # Find all edges starting from this node
        for edge in self.edges.values():
            if edge.source == node_id and edge.target not in visited:
                child_depth = self._calculate_node_depth(edge.target, visited, depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def get_density(self) -> float:
        """Calculate graph density."""
        node_count = self.get_node_count()
        if node_count < 2:
            return 0.0
        
        max_possible_edges = node_count * (node_count - 1)  # Directed graph
        return len(self.edges) / max_possible_edges if max_possible_edges > 0 else 0.0


class InterviewState(BaseModel):
    """Represents the current state of the interview."""
    session_id: str
    turn_number: int = Field(ge=0, default=0)

    question_history: List[str] = Field(default_factory=list)
    tactic_usage: Dict[str, int] = Field(default_factory=dict)  # tactic_id -> usage count
    last_depth_turn: int = Field(ge=0, default=0)  # Track when depth tactics were last used

    emotional_state: str = "neutral"
    created_at: datetime = Field(default_factory=datetime.now)
    
    def increment_turn(self) -> None:
        """Increment the turn number."""
        self.turn_number += 1
    
    def add_question(self, question: str) -> None:
        """Add a question to the history."""
        self.question_history.append(question)
    
    def record_tactic_usage(self, tactic_id: str) -> None:
        """Record that a tactic was used."""
        self.tactic_usage[tactic_id] = self.tactic_usage.get(tactic_id, 0) + 1
    
    def get_tactic_usage_count(self, tactic_id: str) -> int:
        """Get how many times a tactic has been used."""
        return self.tactic_usage.get(tactic_id, 0)


class SchemaTactic(BaseModel):
    """Schema-driven tactic definition."""
    id: str
    intent: str
    trigger: Dict[str, Any] = Field(default_factory=dict)
    pattern: Dict[str, Any] = Field(default_factory=dict)
    followups: Dict[str, str] = Field(default_factory=dict)
    produces_node_types: List[str] = Field(default_factory=list)
    valid_edge_types: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def min_turn(self) -> int:
        """Get minimum turn constraint."""
        return self.constraints.get("min_turn", 0)
    
    @property
    def max_visit_count(self) -> int:
        """Get maximum visit count constraint."""
        return self.constraints.get("max_visit_count", 10)


class Tactic(BaseModel):
    """
    DEPRECATED: Legacy tactic model - use SchemaTactic instead.

    This class exists for backward compatibility with code that expects the legacy
    Tactic format. New code should use SchemaTactic directly from YAML schemas.

    Migration path:
    - SchemaTactics are loaded from YAML (schemas/means_end_chain_v*.yaml)
    - Converted to Tactic via SchemaDrivenTacticLoader._convert_schema_tactic_to_tactic()
    - Used throughout interview pipeline

    TODO: Complete migration to SchemaTactic and remove this class.
    See BUG-020 for details.
    """
    id: str
    name: str
    description: str = Field(default="", description="Description of the tactic")
    min_turn: int = Field(ge=0, default=0)
    max_visit_count: int = Field(ge=0, default=10)


    templates: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategyConfig(BaseModel):
    """Configuration for a strategy."""
    name: StrategyName
    description: str
    tactics: List[str]  # List of tactic IDs
    priority: float = Field(ge=0.0, le=1.0, default=0.5)


class GraphNeedsConfig(BaseModel):
    """Configuration for graph needs detection."""
    min_nodes_for_seed_expansion: int = 4
    isolation_threshold: float = 0.1  # Nodes with <10% connections considered isolated
    depth_completion_threshold: float = 0.3  # Average depth below this triggers need
    target_depth: int = 5  # Target depth for graph (used in depth scoring)
    dead_end_threshold: float = 0.6  # Score above which dead-end is detected (productivity check)
    dead_end_probe_count: int = 3  # Consecutive turns before triggering (grace period)


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_decisions: bool = True
    log_graph_state: bool = True
    log_tactic_selection: bool = True