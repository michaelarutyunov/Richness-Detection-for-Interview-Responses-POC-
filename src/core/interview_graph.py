"""
Interview Graph for AI Interview System.

Wraps NetworkX DiGraph with domain-specific operations for managing
the interview knowledge graph.
"""

import logging
from typing import Any

import networkx as nx

from src.core.data_models import Edge, GraphDelta, Node, Opportunity, OpportunityAction
from src.core.schema_manager import SchemaManager

logger = logging.getLogger(__name__)


class InterviewGraph:
    """Manages the interview knowledge graph using NetworkX."""

    def __init__(self, schema_manager: SchemaManager):
        """
        Initialize Interview Graph.

        Args:
            schema_manager: Schema manager for validation and weights
        """
        self.schema = schema_manager
        self.graph = nx.DiGraph()
        self._node_counter = 0
        self._edge_counter = 0

    def add_node(self, node: Node) -> bool:
        """
        Add node to graph with validation.

        Args:
            node: Node to add

        Returns:
            bool: True if node was added (new), False if merged with existing

        Raises:
            ValueError: Unknown node type
        """
        # Validate node type exists in schema
        try:
            self.schema.get_node_type(node.type)
        except KeyError as e:
            raise ValueError(f"Unknown node type: {node.type}") from e

        # Check if node already exists (merge if so)
        if self.graph.has_node(node.id):
            existing_node = self.graph.nodes[node.id]["data"]
            existing_node.visit_count += 1
            # Update last visit turn if provided in the new node
            if node.last_visit_turn is not None:
                existing_node.last_visit_turn = node.last_visit_turn
            existing_node.source_quotes.extend(node.source_quotes)
            logger.debug(
                f"Merged node {node.id}, now {existing_node.visit_count} visits "
                f"(last visit: turn {existing_node.last_visit_turn})"
            )
            return False  # Not a new node

        # Add new node
        self.graph.add_node(node.id, data=node, node_type=node.type, label=node.label)
        self._node_counter += 1
        logger.debug(f"Added node {node.id} (type={node.type}, label={node.label})")
        return True

    def add_edge(self, edge: Edge) -> bool:
        """
        Add edge to graph with validation.

        Args:
            edge: Edge to add

        Returns:
            bool: True if edge was added (new), False if already exists

        Raises:
            ValueError: Invalid edge (nodes don't exist or edge type not allowed)
        """
        # Validate source and target nodes exist
        if not self.graph.has_node(edge.source):
            raise ValueError(f"Source node not found: {edge.source}")
        if not self.graph.has_node(edge.target):
            raise ValueError(f"Target node not found: {edge.target}")

        # Validate edge type is allowed
        source_type = self.graph.nodes[edge.source]["node_type"]
        target_type = self.graph.nodes[edge.target]["node_type"]

        if not self.schema.is_valid_edge(edge.type, source_type, target_type):
            raise ValueError(f"Invalid edge: {edge.type} from {source_type} to {target_type}")

        # Check if edge already exists
        if self.graph.has_edge(edge.source, edge.target):
            logger.debug(f"Edge {edge.source} -> {edge.target} already exists")
            return False

        # Add edge
        self.graph.add_edge(edge.source, edge.target, data=edge, edge_type=edge.type)
        self._edge_counter += 1
        logger.debug(f"Added edge {edge.source} -> {edge.target} (type={edge.type})")
        return True

    def apply_delta(self, delta: GraphDelta, turn_number: int) -> tuple[int, int]:
        """
        Apply a graph delta from response processing.

        Args:
            delta: Graph changes to apply
            turn_number: Current turn number

        Returns:
            Tuple[int, int]: (nodes_added, edges_added)
        """
        nodes_added = 0
        edges_added = 0

        # Add nodes first
        for node in delta.nodes_added:
            node.creation_turn = turn_number
            if self.add_node(node):
                nodes_added += 1

        # Then add edges (nodes must exist)
        for edge in delta.edges_added:
            edge.creation_turn = turn_number
            try:
                if self.add_edge(edge):
                    edges_added += 1
            except ValueError as e:
                # Log validation error but don't crash
                logger.warning(f"Skipped invalid edge: {e}")

        logger.info(
            f"Applied delta for turn {turn_number}: " f"+{nodes_added} nodes, +{edges_added} edges"
        )

        return nodes_added, edges_added

    def get_node(self, node_id: str) -> Node | None:
        """
        Get node by ID.

        Args:
            node_id: Node identifier

        Returns:
            Optional[Node]: Node if found, None otherwise
        """
        if not self.graph.has_node(node_id):
            return None
        return self.graph.nodes[node_id]["data"]

    def get_neighbors(self, node_id: str, direction: str = "out") -> list[Node]:
        """
        Get neighbor nodes.

        Args:
            node_id: Node identifier
            direction: "out" for successors, "in" for predecessors

        Returns:
            List[Node]: Neighbor nodes

        Raises:
            ValueError: Invalid direction or node not found
        """
        if not self.graph.has_node(node_id):
            raise ValueError(f"Node not found: {node_id}")

        if direction == "out":
            neighbor_ids = list(self.graph.successors(node_id))
        elif direction == "in":
            neighbor_ids = list(self.graph.predecessors(node_id))
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'out' or 'in'.")

        return [self.graph.nodes[nid]["data"] for nid in neighbor_ids]

    def calculate_coverage(self) -> dict[str, Any]:
        """
        Calculate coverage metrics per node type.

        Returns:
            Dict with:
                - overall: float (0.0-1.0) - % of node types with at least 1 node
                - by_type: Dict[str, float] - 1.0 if type covered, 0.0 otherwise
                - node_counts: Dict[str, int] - Count of nodes per type
        """
        coverage = {"overall": 0.0, "by_type": {}, "node_counts": {}}

        # Count nodes per type
        type_counts: dict[str, int] = {}
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id]["node_type"]
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        # Calculate coverage per type (at least 1 node = covered)
        all_types = [nt.name for nt in self.schema.node_types]
        if not all_types:
            return coverage

        covered_types = len([t for t in all_types if type_counts.get(t, 0) > 0])

        coverage["overall"] = covered_types / len(all_types)
        coverage["by_type"] = {t: 1.0 if type_counts.get(t, 0) > 0 else 0.0 for t in all_types}
        coverage["node_counts"] = type_counts

        return coverage

    def calculate_richness(self) -> float:
        """
        Calculate total graph richness per schema formula.

        Returns:
            float: Total richness score
        """
        total_richness = 0.0

        # Sum node richness
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]["data"]
            node_type = node_data.type
            weight = self.schema.get_richness_weight(node_type)
            total_richness += weight

        # Sum edge richness
        for source, target in self.graph.edges():
            edge_data = self.graph.edges[source, target]["data"]
            edge_type = edge_data.type
            boost = self.schema.get_richness_boost(edge_type)
            total_richness += boost

        return total_richness

    def get_expansion_opportunities(self) -> list[Opportunity]:
        """
        Identify promising nodes for exploration.

        Returns:
            List[Opportunity]: Opportunities sorted by priority (descending)
        """
        opportunities = []

        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]["data"]

            # Score based on:
            # 1. Visit count (low = unexplored)
            # 2. Out-degree (low = not elaborated)
            # 3. Node type richness (high weight = valuable)

            visit_count = node_data.visit_count
            out_degree = self.graph.out_degree(node_id)
            node_weight = self.schema.get_richness_weight(node_data.type)

            # Lower visit count = higher priority
            recency_score = 1.0 / (visit_count + 1)

            # Lower out-degree = more opportunity
            depth_score = 1.0 / (out_degree + 1)

            # Combine scores
            priority = (recency_score * 2.0) + (depth_score * 3.0) + (node_weight * 1.0)

            opportunities.append(
                Opportunity(
                    action=OpportunityAction.DIG_DEEPER,
                    target_node_id=node_id,
                    priority=priority,
                    rationale=(
                        f"Node '{node_data.label}' has {visit_count} visits, "
                        f"{out_degree} outgoing edges"
                    ),
                )
            )

        # Sort by priority (descending)
        opportunities.sort(key=lambda o: o.priority, reverse=True)

        return opportunities

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize graph state to dict.

        Returns:
            Dict: Serializable graph state
        """
        return {
            "nodes": [
                {"id": node_id, **self.graph.nodes[node_id]["data"].model_dump()}
                for node_id in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    **self.graph.edges[source, target]["data"].model_dump(),
                }
                for source, target in self.graph.edges()
            ],
            "node_counter": self._node_counter,
            "edge_counter": self._edge_counter,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """
        Deserialize graph state from dict.

        Args:
            data: Serialized graph state
        """
        # Clear existing graph
        self.graph.clear()
        self._node_counter = data.get("node_counter", 0)
        self._edge_counter = data.get("edge_counter", 0)

        # Restore nodes
        for node_data in data.get("nodes", []):
            node = Node(**node_data)
            self.add_node(node)

        # Restore edges
        for edge_data in data.get("edges", []):
            edge = Edge(**edge_data)
            try:
                self.add_edge(edge)
            except ValueError as e:
                logger.warning(f"Skipped edge during deserialization: {e}")

    def export_graphml(self, path: str) -> None:
        """
        Export to GraphML format for visualization.

        Args:
            path: Output file path
        """
        # Create a clean graph for export (without Pydantic objects)
        export_graph = nx.DiGraph()

        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]["data"]
            export_graph.add_node(
                node_id,
                type=node_data.type,
                label=node_data.label,
                creation_turn=node_data.creation_turn,
                visit_count=node_data.visit_count,
            )

        for source, target in self.graph.edges():
            edge_data = self.graph.edges[source, target]["data"]
            export_graph.add_edge(
                source,
                target,
                type=edge_data.type,
                creation_turn=edge_data.creation_turn,
                quote=edge_data.source_quote,
            )

        nx.write_graphml(export_graph, path)
        logger.info(f"Exported graph to {path}")

    @property
    def node_count(self) -> int:
        """Get number of nodes in graph."""
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Get number of edges in graph."""
        return self.graph.number_of_edges()
