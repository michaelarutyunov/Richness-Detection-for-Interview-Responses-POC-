"""
Schema definitions for interview methodologies.
Loads from YAML and validates graph structure against methodology rules.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class NodeTypeDefinition(BaseModel):
    """Definition of a node type within a schema."""
    
    name: str = Field(description="Node type identifier")
    description: str = Field(description="Human-readable description")
    llm_prompt: str = Field(
        default="",
        description="Instructions for LLM on how to identify this type"
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Example instances of this type"
    )
    is_terminal: bool = Field(
        default=False,
        description="Whether this is an end-state type (e.g., 'value')"
    )


class EdgeTypeDefinition(BaseModel):
    """Definition of an edge type within a schema."""
    
    name: str = Field(description="Edge type identifier")
    description: str = Field(description="Human-readable description")
    llm_prompt: str = Field(
        default="",
        description="Instructions for LLM on when to use this edge type"
    )
    valid_sources: List[str] = Field(
        default_factory=list,
        description="Node types that can be sources for this edge"
    )
    valid_targets: List[str] = Field(
        default_factory=list,
        description="Node types that can be targets for this edge"
    )
    
    def get_valid_transitions(self) -> List[Tuple[str, str]]:
        """Generate all valid (source, target) pairs."""
        return [
            (source, target)
            for source in self.valid_sources
            for target in self.valid_targets
        ]


class Schema(BaseModel):
    """
    Interview methodology schema.
    Defines valid node types, edge types, and their relationships.
    """
    
    name: str = Field(description="Schema identifier")
    description: str = Field(description="Human-readable description")
    version: str = Field(default="1.0.0")
    node_types: Dict[str, NodeTypeDefinition] = Field(default_factory=dict)
    edge_types: Dict[str, EdgeTypeDefinition] = Field(default_factory=dict)
    extraction_guidance: str = Field(
        default="",
        description="General guidance for LLM extraction"
    )
    
    @classmethod
    def load(cls, path: str) -> "Schema":
        """
        Load schema from YAML file.
        
        Args:
            path: Path to schema YAML file
            
        Returns:
            Loaded Schema instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Handle nested 'schema' key if present
        if 'schema' in data:
            schema_data = data['schema']
        else:
            schema_data = data
        
        # Parse node types
        node_types = {}
        raw_node_types = schema_data.get('node_types', {})
        for type_name, type_data in raw_node_types.items():
            # Handle both flat and nested formats
            if isinstance(type_data, dict):
                node_types[type_name] = NodeTypeDefinition(
                    name=type_data.get('name', type_name),
                    description=type_data.get('description', ''),
                    llm_prompt=type_data.get('llm_prompt', ''),
                    examples=type_data.get('examples', []),
                    is_terminal=type_data.get('is_terminal', False)
                )
        
        # Parse edge types
        edge_types = {}
        raw_edge_types = schema_data.get('edge_types', [])
        
        # Handle both list and dict formats
        if isinstance(raw_edge_types, list):
            for edge_data in raw_edge_types:
                edge_name = edge_data.get('name', '')
                edge_types[edge_name] = EdgeTypeDefinition(
                    name=edge_name,
                    description=edge_data.get('description', ''),
                    llm_prompt=edge_data.get('llm_prompt', ''),
                    valid_sources=edge_data.get('valid_sources', []),
                    valid_targets=edge_data.get('valid_targets', [])
                )
        elif isinstance(raw_edge_types, dict):
            for edge_name, edge_data in raw_edge_types.items():
                edge_types[edge_name] = EdgeTypeDefinition(
                    name=edge_data.get('name', edge_name),
                    description=edge_data.get('description', ''),
                    llm_prompt=edge_data.get('llm_prompt', ''),
                    valid_sources=edge_data.get('valid_sources', []),
                    valid_targets=edge_data.get('valid_targets', [])
                )
        
        return cls(
            name=schema_data.get('name', path.stem),
            description=schema_data.get('description', ''),
            version=schema_data.get('version', '1.0.0'),
            node_types=node_types,
            edge_types=edge_types,
            extraction_guidance=schema_data.get('extraction_guidance', '')
        )
    
    def is_valid_edge(
        self, 
        source_type: str, 
        target_type: str, 
        relation_type: str
    ) -> bool:
        """
        Check if an edge is valid according to schema rules.
        
        Args:
            source_type: Node type of source
            target_type: Node type of target
            relation_type: Edge type
            
        Returns:
            True if edge is valid
        """
        if relation_type not in self.edge_types:
            return False
        
        edge_def = self.edge_types[relation_type]
        return (
            source_type in edge_def.valid_sources and
            target_type in edge_def.valid_targets
        )
    
    def is_terminal_type(self, node_type: str) -> bool:
        """Check if a node type is terminal (end-state)."""
        if node_type not in self.node_types:
            return False
        return self.node_types[node_type].is_terminal
    
    def get_valid_edge_types(
        self,
        source_type: str,
        target_type: str
    ) -> List[str]:
        """
        Get all edge types valid between two node types.

        Args:
            source_type: Node type of source
            target_type: Node type of target

        Returns:
            List of valid edge type names
        """
        valid = []
        for edge_name, edge_def in self.edge_types.items():
            if (source_type in edge_def.valid_sources and
                target_type in edge_def.valid_targets):
                valid.append(edge_name)
        return valid

    def get_default_edge_type(self) -> str:
        """
        Get default edge type for fallback when LLM doesn't specify one.

        Returns first edge type from schema, prioritizing 'relates_to' if available.
        This ensures schema-agnostic fallback behavior.

        Returns:
            Default edge type name
        """
        if not self.edge_types:
            return "relates_to"  # Ultimate fallback

        # Prefer 'relates_to' if available (most generic edge type)
        if "relates_to" in self.edge_types:
            return "relates_to"

        # Otherwise return first edge type
        return next(iter(self.edge_types.keys()))

    def get_node_type_order(self) -> List[str]:
        """
        Get node types in hierarchical order (concrete to abstract).
        Based on edge flow patterns.
        
        Returns:
            List of node type names from base to terminal
        """
        # Find types that are only sources (base types)
        # Find types that are only targets (terminal types)
        # Order the rest by their position in the flow
        
        all_types = set(self.node_types.keys())
        source_only: set = set()
        target_only: set = set()
        
        for edge_def in self.edge_types.values():
            for source in edge_def.valid_sources:
                if source in all_types:
                    # Check if this type is ever a target
                    is_target = any(
                        source in e.valid_targets 
                        for e in self.edge_types.values()
                    )
                    if not is_target:
                        source_only.add(source)
            
            for target in edge_def.valid_targets:
                if target in all_types:
                    # Check if this type is ever a source
                    is_source = any(
                        target in e.valid_sources 
                        for e in self.edge_types.values()
                    )
                    if not is_source:
                        target_only.add(target)
        
        # Build order: sources first, then middle, then targets
        middle = all_types - source_only - target_only
        
        # Simple ordering - could be improved with topological sort
        ordered = list(source_only) + list(middle) + list(target_only)
        return ordered
    
    def get_extraction_prompt(self) -> str:
        """
        Generate comprehensive extraction prompt for LLM.
        Includes all node types, edge types, and guidance.
        """
        lines = []
        
        lines.append("# Extraction Schema")
        lines.append("")
        lines.append(f"**Methodology**: {self.name}")
        lines.append(f"**Description**: {self.description}")
        lines.append("")
        
        # Node types
        lines.append("## Node Types")
        lines.append("")
        for type_name, type_def in self.node_types.items():
            lines.append(f"### {type_name}")
            lines.append(f"{type_def.description}")
            if type_def.llm_prompt:
                lines.append("")
                lines.append(type_def.llm_prompt.strip())
            if type_def.examples:
                lines.append("")
                lines.append(f"Examples: {', '.join(type_def.examples)}")
            if type_def.is_terminal:
                lines.append("")
                lines.append("*This is a terminal type (end-state).*")
            lines.append("")
        
        # Edge types
        lines.append("## Edge Types (CHOOSE CAREFULLY)")
        lines.append("")
        lines.append("**CRITICAL**: Read each edge type's guidance before selecting.")
        lines.append("")
        for edge_name, edge_def in self.edge_types.items():
            lines.append(f"### {edge_name}")
            lines.append(f"{edge_def.description}")
            if edge_def.llm_prompt:
                lines.append("")
                lines.append(edge_def.llm_prompt.strip())
            lines.append("")
            lines.append(f"Valid: {edge_def.valid_sources} → {edge_def.valid_targets}")
            lines.append("")
        
        # General guidance
        if self.extraction_guidance:
            lines.append("## Extraction Guidance")
            lines.append("")
            lines.append(self.extraction_guidance.strip())
        
        return "\n".join(lines)

    def validate_extraction_result(self, nodes: List, edges: List, strict: bool = False) -> List[str]:
        """
        Validate extraction result quality.

        Args:
            nodes: List of Node objects extracted
            edges: List of Edge objects extracted
            strict: If True, return errors; if False, return warnings

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        # Check edge-to-node ratio
        if len(nodes) > 0:
            ratio = len(edges) / len(nodes)
            if ratio < 0.3:
                warnings.append(
                    f"Low edge density: {ratio:.2f} (target >0.5). "
                    f"Extracted {len(edges)} edges for {len(nodes)} nodes."
                )

        # Check for isolated nodes (nodes not in any edge)
        if nodes and edges:
            node_ids = {getattr(n, 'id', None) for n in nodes}
            connected_ids = set()
            for e in edges:
                if hasattr(e, 'source_id'):
                    connected_ids.add(e.source_id)
                if hasattr(e, 'target_id'):
                    connected_ids.add(e.target_id)

            isolated = node_ids - connected_ids - {None}
            if isolated and len(isolated) / len(node_ids) > 0.5:
                warnings.append(
                    f"High isolation: {len(isolated)}/{len(node_ids)} nodes not connected to any edge"
                )

        # Check for invalid edge types
        for edge in edges:
            relation_type = getattr(edge, 'relation_type', None)
            if relation_type and relation_type not in self.edge_types:
                warnings.append(
                    f"Invalid edge type: '{relation_type}' not in schema edge types: "
                    f"{list(self.edge_types.keys())}"
                )

        # Check for invalid node types
        for node in nodes:
            node_type = getattr(node, 'node_type', None)
            if node_type and node_type not in self.node_types:
                warnings.append(
                    f"Invalid node type: '{node_type}' not in schema node types: "
                    f"{list(self.node_types.keys())}"
                )

        return warnings

    def validate_graph(self, graph: "Graph") -> List[str]:
        """
        Validate a graph against this schema.
        
        Args:
            graph: Graph to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        # Avoid circular import
        from core.graph import Graph
        
        errors = []
        
        # Check node types
        for node in graph.nodes.values():
            if node.node_type and node.node_type not in self.node_types:
                errors.append(
                    f"Node '{node.label}' has invalid type: {node.node_type}"
                )
        
        # Check edges
        for edge in graph.edges.values():
            # Check edge type exists
            if edge.relation_type not in self.edge_types:
                errors.append(
                    f"Edge has invalid type: {edge.relation_type}"
                )
                continue
            
            # Check source/target types are valid for this edge
            source = graph.get_node(edge.source_id)
            target = graph.get_node(edge.target_id)
            
            if source and target and source.node_type and target.node_type:
                if not self.is_valid_edge(
                    source.node_type, 
                    target.node_type, 
                    edge.relation_type
                ):
                    errors.append(
                        f"Invalid edge: {source.node_type} --{edge.relation_type}--> "
                        f"{target.node_type} (nodes: {source.label} -> {target.label})"
                    )
        
        return errors
