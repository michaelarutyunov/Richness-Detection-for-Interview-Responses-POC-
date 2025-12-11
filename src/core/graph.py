"""
Graph data structure and low-level operations.
Pure data, no interview logic.
"""

from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class Node(BaseModel):
    """Represents a concept node in the interview knowledge graph."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str = Field(description="Respondent's language for this concept")
    node_type: Optional[str] = Field(
        default=None, 
        description="Schema-defined type (e.g., 'attribute', 'value')"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    is_ambiguous: bool = Field(
        default=False,
        description="Flagged for clarification before further use"
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional data (quotes, element mappings, etc.)"
    )
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False


class Edge(BaseModel):
    """Represents a relationship between concepts in the knowledge graph."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = Field(description="ID of source node")
    target_id: str = Field(description="ID of target node")
    relation_type: str = Field(description="Schema-defined edge type (e.g., 'leads_to')")
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional data (quotes, turn extracted, etc.)"
    )
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.id == other.id
        return False


class Graph(BaseModel):
    """
    Knowledge graph built during interview.
    Stores nodes (concepts) and edges (relationships).
    """
    
    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: Dict[str, Edge] = Field(default_factory=dict)
    
    # --- Node operations ---
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        return self.nodes.get(node_id)
    
    def get_node_by_label(self, label: str) -> Optional[Node]:
        """Find a node by its label (case-insensitive)."""
        label_lower = label.lower()
        for node in self.nodes.values():
            if node.label.lower() == label_lower:
                return node
        return None
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connected edges."""
        if node_id not in self.nodes:
            return False
        
        # Remove connected edges
        edges_to_remove = [
            edge_id for edge_id, edge in self.edges.items()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
        
        del self.nodes[node_id]
        return True
    
    # --- Edge operations ---
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.id] = edge
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Retrieve an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_edge_between(self, source_id: str, target_id: str) -> Optional[Edge]:
        """Find an edge between two specific nodes."""
        for edge in self.edges.values():
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge by ID."""
        if edge_id in self.edges:
            del self.edges[edge_id]
            return True
        return False
    
    def get_edges_for_node(self, node_id: str) -> List[Edge]:
        """Get all edges connected to a node (incoming and outgoing)."""
        return [
            edge for edge in self.edges.values()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
    
    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get edges where node is the source."""
        return [
            edge for edge in self.edges.values()
            if edge.source_id == node_id
        ]
    
    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Get edges where node is the target."""
        return [
            edge for edge in self.edges.values()
            if edge.target_id == node_id
        ]
    
    # --- Graph queries ---
    
    def get_neighbors(self, node_id: str) -> List[Node]:
        """Get all nodes directly connected to the given node."""
        neighbor_ids: Set[str] = set()
        for edge in self.edges.values():
            if edge.source_id == node_id:
                neighbor_ids.add(edge.target_id)
            elif edge.target_id == node_id:
                neighbor_ids.add(edge.source_id)
        
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]
    
    def get_isolated_nodes(self) -> List[Node]:
        """Get nodes with no edges."""
        connected_ids: Set[str] = set()
        for edge in self.edges.values():
            connected_ids.add(edge.source_id)
            connected_ids.add(edge.target_id)
        
        return [
            node for node in self.nodes.values()
            if node.id not in connected_ids
        ]
    
    def get_ambiguous_nodes(self) -> List[Node]:
        """Get nodes flagged as ambiguous."""
        return [node for node in self.nodes.values() if node.is_ambiguous]
    
    def get_recent_nodes(self, n: int) -> List[Node]:
        """Get the n most recently added nodes."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_nodes[:n]
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type."""
        return [
            node for node in self.nodes.values()
            if node.node_type == node_type
        ]
    
    # --- Graph metrics ---
    
    def get_subgraph_depth(self, node_id: str, direction: str = "up") -> int:
        """
        Calculate depth from a node in specified direction.
        
        Args:
            node_id: Starting node
            direction: "up" (follow outgoing edges) or "down" (follow incoming)
        
        Returns:
            Maximum depth reachable from node
        """
        if node_id not in self.nodes:
            return 0
        
        visited: Set[str] = set()
        
        def dfs(current_id: str, depth: int) -> int:
            if current_id in visited:
                return depth - 1
            visited.add(current_id)
            
            if direction == "up":
                next_edges = self.get_outgoing_edges(current_id)
                next_ids = [e.target_id for e in next_edges]
            else:
                next_edges = self.get_incoming_edges(current_id)
                next_ids = [e.source_id for e in next_edges]
            
            if not next_ids:
                return depth
            
            return max(dfs(nid, depth + 1) for nid in next_ids)
        
        return dfs(node_id, 0)
    
    def compute_density(self) -> float:
        """
        Compute graph density (edges / possible edges).
        Returns 0 if fewer than 2 nodes.
        """
        n = len(self.nodes)
        if n < 2:
            return 0.0
        
        possible_edges = n * (n - 1)  # Directed graph
        return len(self.edges) / possible_edges
    
    def compute_isolation_ratio(self) -> float:
        """Ratio of isolated nodes to total nodes."""
        if not self.nodes:
            return 0.0
        return len(self.get_isolated_nodes()) / len(self.nodes)
    
    # --- Serialization ---
    
    def summary(self) -> str:
        """
        Human-readable summary for LLM context.
        Shows nodes grouped by type with their connections.
        """
        if not self.nodes:
            return "Graph is empty."
        
        lines = []
        lines.append(f"Graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
        lines.append("")
        
        # Group nodes by type
        by_type: Dict[str, List[Node]] = {}
        for node in self.nodes.values():
            type_key = node.node_type or "untyped"
            if type_key not in by_type:
                by_type[type_key] = []
            by_type[type_key].append(node)
        
        for node_type, nodes in sorted(by_type.items()):
            lines.append(f"[{node_type}]")
            for node in nodes:
                # Get connections
                outgoing = self.get_outgoing_edges(node.id)
                connections = []
                for edge in outgoing:
                    target = self.get_node(edge.target_id)
                    if target:
                        connections.append(f"--{edge.relation_type}--> {target.label}")
                
                node_line = f"  • {node.label}"
                if node.is_ambiguous:
                    node_line += " [AMBIGUOUS]"
                if connections:
                    node_line += f" ({'; '.join(connections)})"
                lines.append(node_line)
            lines.append("")
        
        # Note isolated nodes
        isolated = self.get_isolated_nodes()
        if isolated:
            lines.append(f"Isolated nodes: {', '.join(n.label for n in isolated)}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Export graph as dictionary for analysis/storage."""
        return {
            "nodes": [node.model_dump() for node in self.nodes.values()],
            "edges": [edge.model_dump() for edge in self.edges.values()],
            "metrics": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "density": self.compute_density(),
                "isolation_ratio": self.compute_isolation_ratio()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Graph":
        """Reconstruct graph from dictionary."""
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(Node(**node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(Edge(**edge_data))
        return graph
