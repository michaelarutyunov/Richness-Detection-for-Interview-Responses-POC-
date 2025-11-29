"""
Validator for LLM extraction output.

Validates extracted nodes and edges against schema rules with 4-stage validation:
1. Structure: Check required fields present
2. Schema: Validate types exist, label format correct
3. Graph: Ensure edges reference valid nodes
4. Semantic: Verify quotes appear in participant response
"""

import logging
import re
from dataclasses import dataclass

from src.core.interview_graph import InterviewGraph
from src.core.schema_manager import SchemaManager

logger = logging.getLogger(__name__)


@dataclass
class ExtractedNode:
    """Cleaned and validated node from LLM extraction."""

    type: str
    label: str
    quote: str


@dataclass
class ExtractedEdge:
    """Cleaned and validated edge from LLM extraction."""

    type: str
    source: str
    target: str
    quote: str


@dataclass
class ValidationResult:
    """Result of validation process."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    cleaned_nodes: list[ExtractedNode]
    cleaned_edges: list[ExtractedEdge]


class Validator:
    """Validates LLM extraction output against schema."""

    def __init__(self, schema_manager: SchemaManager):
        """
        Initialize validator.

        Args:
            schema_manager: Schema manager for validation rules
        """
        self.schema = schema_manager

    def validate(
        self,
        raw_extraction: dict,
        existing_graph: InterviewGraph,
        participant_response: str,
    ) -> ValidationResult:
        """
        Validate LLM extraction output with 4-stage validation.

        Args:
            raw_extraction: Raw extraction dict from LLM
            existing_graph: Current graph state
            participant_response: Original participant response text

        Returns:
            ValidationResult: Validation result with cleaned nodes/edges
        """
        errors = []
        warnings = []
        cleaned_nodes = []
        cleaned_edges = []

        # Stage 1: Structure validation
        if not isinstance(raw_extraction, dict):
            errors.append("Extraction output is not a dictionary")
            return ValidationResult(False, errors, warnings, [], [])

        nodes_raw = raw_extraction.get("nodes_added", [])
        edges_raw = raw_extraction.get("edges_added", [])

        if not isinstance(nodes_raw, list):
            errors.append("nodes_added is not a list")
            return ValidationResult(False, errors, warnings, [], [])

        if not isinstance(edges_raw, list):
            errors.append("edges_added is not a list")
            return ValidationResult(False, errors, warnings, [], [])

        # Stage 2: Node validation (schema + semantic)
        for idx, node_raw in enumerate(nodes_raw):
            try:
                node_type = node_raw.get("type")
                label = node_raw.get("label")
                quote = node_raw.get("quote", "")

                # Check required fields
                if not node_type:
                    errors.append(f"Node {idx}: Missing 'type' field")
                    continue
                if not label:
                    errors.append(f"Node {idx}: Missing 'label' field")
                    continue

                # Validate type exists in schema
                try:
                    self.schema.get_node_type(node_type)
                except KeyError:
                    errors.append(f"Node {idx} ('{label}'): Unknown node type '{node_type}'")
                    continue

                # Validate label format (lowercase_with_underscores, 3-40 chars)
                if not re.match(r"^[a-z_]{3,40}$", label):
                    errors.append(
                        f"Node {idx} ('{label}'): Invalid label format. "
                        f"Must be lowercase_with_underscores, 3-40 characters"
                    )
                    continue

                # Semantic validation: Check quote appears in response
                if not quote:
                    warnings.append(f"Node '{label}': Empty quote")
                elif quote.lower() not in participant_response.lower():
                    warnings.append(
                        f"Node '{label}': Quote '{quote[:50]}...' not found in response"
                    )

                # Node is valid
                cleaned_nodes.append(ExtractedNode(type=node_type, label=label, quote=quote))

            except Exception as e:
                errors.append(f"Node {idx}: Unexpected validation error: {e}")
                logger.exception(f"Unexpected error validating node {idx}")

        # Stage 3 & 4: Edge validation (graph + semantic)
        # Build set of all available node labels
        new_node_labels = {n.label for n in cleaned_nodes}
        existing_node_labels = set()

        for node_id in existing_graph.graph.nodes():
            node_data = existing_graph.graph.nodes[node_id]["data"]
            existing_node_labels.add(node_data.label)

        all_node_labels = new_node_labels | existing_node_labels

        # Build label-to-type mapping for edge validation
        label_to_type = {}
        for node in cleaned_nodes:
            label_to_type[node.label] = node.type
        for node_id in existing_graph.graph.nodes():
            node_data = existing_graph.graph.nodes[node_id]["data"]
            label_to_type[node_data.label] = node_data.type

        for idx, edge_raw in enumerate(edges_raw):
            try:
                edge_type = edge_raw.get("type")
                source = edge_raw.get("source")
                target = edge_raw.get("target")
                quote = edge_raw.get("quote", "")

                # Check required fields
                if not edge_type:
                    errors.append(f"Edge {idx}: Missing 'type' field")
                    continue
                if not source:
                    errors.append(f"Edge {idx}: Missing 'source' field")
                    continue
                if not target:
                    errors.append(f"Edge {idx}: Missing 'target' field")
                    continue

                # Validate edge type exists
                try:
                    self.schema.get_edge_type(edge_type)
                except KeyError:
                    errors.append(
                        f"Edge {idx} ({source}->{target}): Unknown edge type '{edge_type}'"
                    )
                    continue

                # Validate source and target nodes exist
                if source not in all_node_labels:
                    errors.append(f"Edge {idx}: Source node '{source}' not found in nodes")
                    continue
                if target not in all_node_labels:
                    errors.append(f"Edge {idx}: Target node '{target}' not found in nodes")
                    continue

                # Validate edge type compatibility with node types
                source_type = label_to_type[source]
                target_type = label_to_type[target]

                if not self.schema.is_valid_edge(edge_type, source_type, target_type):
                    errors.append(
                        f"Edge {idx}: Invalid edge '{edge_type}' from {source_type} to {target_type}. "
                        f"Check schema rules for allowed connections."
                    )
                    continue

                # Semantic validation: Check quote
                if not quote:
                    warnings.append(f"Edge {source}->{target}: Empty quote")
                elif quote.lower() not in participant_response.lower():
                    warnings.append(
                        f"Edge {source}->{target}: Quote '{quote[:50]}...' not found in response"
                    )

                # Edge is valid
                cleaned_edges.append(
                    ExtractedEdge(type=edge_type, source=source, target=target, quote=quote)
                )

            except Exception as e:
                errors.append(f"Edge {idx}: Unexpected validation error: {e}")
                logger.exception(f"Unexpected error validating edge {idx}")

        # Determine overall validity
        is_valid = len(errors) == 0

        if errors:
            logger.warning(f"Validation found {len(errors)} errors, {len(warnings)} warnings")
        elif warnings:
            logger.info(f"Validation passed with {len(warnings)} warnings")
        else:
            logger.info("Validation passed cleanly")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cleaned_nodes=cleaned_nodes,
            cleaned_edges=cleaned_edges,
        )
