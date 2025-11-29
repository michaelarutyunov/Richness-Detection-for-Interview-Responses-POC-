"""
Schema Manager for AI Interview System.

Loads and validates YAML schema manifests, provides typed access to
node types, edge types, and interview configuration.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from src.core.data_models import EdgeTypeConfig, NodeTypeConfig, SchemaManifest, SeedNodeConfig

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages schema loading, validation, and access."""

    def __init__(self, schema_path: str):
        """
        Initialize Schema Manager.

        Args:
            schema_path: Path to YAML schema file
        """
        self.schema_path = Path(schema_path)
        self._manifest: SchemaManifest | None = None
        self._node_type_map: dict[str, NodeTypeConfig] = {}
        self._edge_type_map: dict[str, EdgeTypeConfig] = {}
        self._seed_node_map: dict[str, SeedNodeConfig] = {}

    def load_schema(self) -> SchemaManifest:
        """
        Load and parse YAML schema file.

        Returns:
            SchemaManifest: Parsed and validated schema

        Raises:
            FileNotFoundError: Schema file doesn't exist
            ValueError: Failed to parse or validate schema
        """
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        try:
            with open(self.schema_path, encoding="utf-8") as f:
                schema_data = yaml.safe_load(f)

            # Parse into Pydantic model (validates structure)
            self._manifest = SchemaManifest(**schema_data)

            # Build lookup maps for fast access
            self._node_type_map = {nt.name: nt for nt in self._manifest.node_types}
            self._edge_type_map = {et.name: et for et in self._manifest.edge_types}
            self._seed_node_map = {sn.name: sn for sn in self._manifest.seed_nodes}

            logger.info(
                f"Loaded schema '{self._manifest.domain}' v{self._manifest.schema_version} "
                f"with {len(self._node_type_map)} node types, {len(self._edge_type_map)} edge types"
            )

            return self._manifest

        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML in {self.schema_path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load schema from {self.schema_path}: {e}") from e

    def validate_schema(self) -> bool:
        """
        Validate schema consistency and rules.

        Returns:
            bool: True if schema is valid

        Raises:
            ValueError: Schema not loaded or validation failed
        """
        if not self._manifest:
            raise ValueError("Schema not loaded. Call load_schema() first.")

        node_type_names = set(self._node_type_map.keys())

        # Check 1: All edge valid_sources exist as node types
        for edge_type in self._manifest.edge_types:
            for source_type in edge_type.valid_sources:
                if source_type not in node_type_names:
                    raise ValueError(
                        f"Edge type '{edge_type.name}' references "
                        f"unknown source type '{source_type}'"
                    )
            for target_type in edge_type.valid_targets:
                if target_type not in node_type_names:
                    raise ValueError(
                        f"Edge type '{edge_type.name}' references "
                        f"unknown target type '{target_type}'"
                    )

        # Check 2: All seed nodes reference valid node types
        for seed in self._manifest.seed_nodes:
            if seed.type not in node_type_names:
                raise ValueError(
                    f"Seed node '{seed.name}' references " f"unknown type '{seed.type}'"
                )

        logger.info("Schema validation passed")
        return True

    def get_node_type(self, type_name: str) -> NodeTypeConfig:
        """
        Get configuration for a specific node type.

        Args:
            type_name: Name of node type (e.g., 'attribute')

        Returns:
            NodeTypeConfig: Configuration for this node type

        Raises:
            KeyError: Node type not found in schema
        """
        if type_name not in self._node_type_map:
            raise KeyError(f"Node type '{type_name}' not found in schema")
        return self._node_type_map[type_name]

    def get_edge_type(self, type_name: str) -> EdgeTypeConfig:
        """
        Get configuration for a specific edge type.

        Args:
            type_name: Name of edge type (e.g., 'leads_to')

        Returns:
            EdgeTypeConfig: Configuration for this edge type

        Raises:
            KeyError: Edge type not found in schema
        """
        if type_name not in self._edge_type_map:
            raise KeyError(f"Edge type '{type_name}' not found in schema")
        return self._edge_type_map[type_name]

    def is_valid_edge(self, edge_type: str, source_type: str, target_type: str) -> bool:
        """
        Validate if an edge connection is allowed by schema.

        Args:
            edge_type: Type of edge (e.g., 'leads_to')
            source_type: Type of source node
            target_type: Type of target node

        Returns:
            bool: True if this edge is valid per schema rules
        """
        edge_config = self._edge_type_map.get(edge_type)
        if not edge_config:
            return False

        return source_type in edge_config.valid_sources and target_type in edge_config.valid_targets

    def get_richness_weight(self, node_type: str) -> float:
        """
        Get richness weight for a node type.

        Args:
            node_type: Type of node

        Returns:
            float: Richness weight for this node type

        Raises:
            KeyError: Node type not found
        """
        config = self.get_node_type(node_type)
        return config.richness_weight

    def get_richness_boost(self, edge_type: str) -> float:
        """
        Get richness boost for an edge type.

        Args:
            edge_type: Type of edge

        Returns:
            float: Richness boost for this edge type

        Raises:
            KeyError: Edge type not found
        """
        config = self.get_edge_type(edge_type)
        return config.richness_boost

    def get_probing_prompt(self, node_type: str, node_label: str) -> str:
        """
        Get template for probing a specific node.

        Args:
            node_type: Type of node
            node_label: Label of the node

        Returns:
            str: Probing prompt with {node} replaced by node_label

        Raises:
            KeyError: Node type not found
        """
        config = self.get_node_type(node_type)
        return config.probing_prompt.replace("{node}", node_label)

    def get_llm_extraction_prompt(self, node_type: str) -> str:
        """
        Get LLM prompt for extracting this node type.

        Args:
            node_type: Type of node

        Returns:
            str: LLM extraction prompt for this node type

        Raises:
            KeyError: Node type not found
        """
        config = self.get_node_type(node_type)
        return config.llm_extraction_prompt

    def get_interview_config(self) -> dict[str, Any]:
        """
        Get interview configuration parameters.

        Returns:
            Dict: Interview configuration

        Raises:
            ValueError: Schema not loaded
        """
        if not self._manifest:
            raise ValueError("Schema not loaded")
        return self._manifest.interview_config

    def get_seed_nodes(self) -> list[SeedNodeConfig]:
        """
        Get all seed node configurations.

        Returns:
            List[SeedNodeConfig]: List of seed node configs

        Raises:
            ValueError: Schema not loaded
        """
        if not self._manifest:
            raise ValueError("Schema not loaded")
        return self._manifest.seed_nodes

    @property
    def schema_version(self) -> str | None:
        """Get schema version."""
        return self._manifest.schema_version if self._manifest else None

    @property
    def domain(self) -> str | None:
        """Get schema domain."""
        return self._manifest.domain if self._manifest else None

    @property
    def node_types(self) -> list[NodeTypeConfig]:
        """Get all node type configurations."""
        return list(self._node_type_map.values())

    @property
    def edge_types(self) -> list[EdgeTypeConfig]:
        """Get all edge type configurations."""
        return list(self._edge_type_map.values())
