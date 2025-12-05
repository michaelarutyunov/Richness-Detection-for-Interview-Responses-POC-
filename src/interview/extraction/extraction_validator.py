"""
Streamlined 2-stage validator for concept extraction.
Excludes semantic validation and misclassification detection for efficiency.
"""

import logging
import re
from typing import List, Dict, Any, Set
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class ExtractionValidator:
    """Validates extracted concepts with essential checks only."""
    
    def __init__(self, schema_path: str = "schemas/means_end_chain_v0.2.yaml"):
        """Initialize with schema for validation rules."""
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.node_types: Dict[str, dict] = {}
        self.edge_types: Dict[str, dict] = {}
        self._parse_schema()
        logger.info(f"ExtractionValidator initialized with schema: {schema_path}")
    
    def _load_schema(self) -> dict:
        """Load schema file."""
        if not self.schema_path.exists():
            logger.warning(f"Schema file not found: {self.schema_path}, using defaults")
            return self._get_default_schema()
        
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading schema: {e}, using defaults")
            return self._get_default_schema()
    
    def _get_default_schema(self) -> dict:
        """Default schema for validation."""
        return {
            "node_types": [
                {"name": "attribute", "validation_regex": "^[a-z_]{3,40}$"},
                {"name": "functional_consequence", "validation_regex": "^[a-z_]{3,40}$"},
                {"name": "psychosocial_consequence", "validation_regex": "^[a-z_]{3,40}$"},
                {"name": "value", "validation_regex": "^[a-z_]{3,40}$"}
            ],
            "edge_types": [
                {"name": "leads_to", "valid_sources": ["attribute", "functional_consequence", "psychosocial_consequence"], "valid_targets": ["functional_consequence", "psychosocial_consequence", "value"]},
                {"name": "blocks", "valid_sources": ["attribute", "functional_consequence"], "valid_targets": ["psychosocial_consequence", "value"]},
                {"name": "correlates_with", "valid_sources": ["attribute", "functional_consequence", "psychosocial_consequence"], "valid_targets": ["attribute", "functional_consequence", "psychosocial_consequence"]},
                {"name": "enables", "valid_sources": ["attribute", "functional_consequence"], "valid_targets": ["attribute", "functional_consequence", "psychosocial_consequence"]},
                {"name": "exemplifies", "valid_sources": ["functional_consequence", "psychosocial_consequence", "attribute"], "valid_targets": ["attribute", "functional_consequence", "psychosocial_consequence"]}
            ]
        }
    
    def _parse_schema(self):
        """Parse schema into validation structures."""
        # Parse node types
        for nt in self.schema.get("node_types", []):
            name = nt.get("name", "")
            if name:
                self.node_types[name] = nt
        
        # Parse edge types
        for et in self.schema.get("edge_types", []):
            name = et.get("name", "")
            if name:
                self.edge_types[name] = et
    
    def validate_extraction(self, extraction: dict, original_response: str) -> dict:
        """
        2-stage validation: Structure + Schema only.
        
        Args:
            extraction: Raw extraction from LLM
            original_response: Original participant response (unused in MVP)
            
        Returns:
            Dict with validated nodes/edges and any errors
        """
        logger.debug("Starting 2-stage validation")
        errors = []
        
        # Stage 1: Structure validation
        structure_valid, structure_errors = self._validate_structure(extraction)
        if not structure_valid:
            errors.extend(structure_errors)
            logger.warning(f"Structure validation failed: {structure_errors}")
            return {"nodes": [], "edges": [], "errors": errors}
        
        # Stage 2: Schema validation
        raw_nodes = extraction.get("nodes", [])
        raw_edges = extraction.get("edges", [])

        valid_nodes, node_errors, rejected_nodes = self._validate_nodes(raw_nodes)
        valid_edges, edge_errors = self._validate_edges(raw_edges, valid_nodes, rejected_nodes)
        
        errors.extend(node_errors)
        errors.extend(edge_errors)
        
        if errors:
            logger.warning(f"Schema validation found {len(errors)} issues")
        
        return {
            "nodes": valid_nodes,
            "edges": valid_edges,
            "errors": errors
        }
    
    def _validate_structure(self, extraction: dict) -> tuple[bool, List[str]]:
        """Stage 1: Check required fields exist."""
        errors = []
        
        # Check top-level structure
        if not isinstance(extraction, dict):
            errors.append("Extraction must be a dictionary")
            return False, errors
        
        # Check required keys
        if "nodes" not in extraction:
            errors.append("Missing 'nodes' field")
        
        if "edges" not in extraction:
            errors.append("Missing 'edges' field")
        
        # Validate nodes structure
        nodes = extraction.get("nodes", [])
        if not isinstance(nodes, list):
            errors.append("'nodes' must be a list")
        else:
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    errors.append(f"Node {i} must be a dictionary")
                    continue
                
                required_fields = ["type", "label", "quote"]
                for field in required_fields:
                    if field not in node:
                        errors.append(f"Node {i} missing required field: {field}")
        
        # Validate edges structure
        edges = extraction.get("edges", [])
        if not isinstance(edges, list):
            errors.append("'edges' must be a list")
        else:
            for i, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    errors.append(f"Edge {i} must be a dictionary")
                    continue
                
                required_fields = ["type", "source", "target", "quote", "confidence"]
                for field in required_fields:
                    if field not in edge:
                        errors.append(f"Edge {i} missing required field: {field}")
        
        return len(errors) == 0, errors
    
    def _validate_nodes(self, nodes: List[dict]) -> tuple[List[dict], List[str], Dict[str, str]]:
        """Stage 2: Validate node types and formats.

        Returns:
            Tuple of (valid_nodes, errors, rejected_nodes)
            rejected_nodes is a dict mapping label -> rejection_reason
        """
        valid_nodes = []
        errors = []
        rejected_nodes = {}  # Track rejected nodes with their rejection reasons
        
        for i, node in enumerate(nodes):
            try:
                # Validate node type
                node_type = node.get("type", "")
                if node_type not in self.node_types:
                    # Track rejection if label exists
                    if "label" in node:
                        rejected_nodes[node["label"]] = f"Invalid type '{node_type}'"
                    errors.append(f"Node {i}: Invalid type '{node_type}'")
                    continue

                # Validate label format using regex if available
                if "label" not in node:
                    errors.append(f"Node {i}: Missing label field")
                    continue

                label = node["label"]
                if not isinstance(label, str) or not label.strip():
                    rejected_nodes[str(label)] = "Empty or invalid label"
                    errors.append(f"Node {i}: Empty or invalid label")
                    continue

                # Check regex pattern if defined, with auto-normalization
                node_type_config = self.node_types[node_type]
                if "validation_regex" in node_type_config:
                    pattern = node_type_config["validation_regex"]
                    # Auto-normalize label: lowercase and replace spaces with underscores
                    normalized_label = label.lower().replace(" ", "_")
                    if not re.match(pattern, normalized_label):
                        rejected_nodes[label] = f"Label doesn't match pattern '{pattern}' (even after normalization)"
                        errors.append(f"Node {i}: Label '{label}' (normalized: '{normalized_label}') doesn't match pattern '{pattern}'")
                        continue
                    # Update node with normalized label if it was changed
                    if normalized_label != label:
                        logger.debug(f"Normalized label '{label}' to '{normalized_label}'")
                        node["label"] = normalized_label

                # Validate quote exists
                quote = node.get("quote", "")
                if not quote:
                    rejected_nodes[node["label"]] = "Missing quote"
                    errors.append(f"Node {i}: Missing quote")
                    continue
                
                # Node is valid
                valid_nodes.append(node)
                
            except Exception as e:
                if "label" in node:
                    rejected_nodes[node["label"]] = f"Validation error - {str(e)}"
                errors.append(f"Node {i}: Validation error - {str(e)}")
                continue

        return valid_nodes, errors, rejected_nodes
    
    def _validate_edges(self, edges: List[dict], valid_nodes: List[dict], rejected_nodes: Dict[str, str] = None) -> tuple[List[dict], List[str]]:
        """Stage 2: Validate edge types and references.

        Args:
            edges: List of edge dictionaries to validate
            valid_nodes: List of valid nodes from node validation
            rejected_nodes: Dict mapping rejected node labels to rejection reasons

        Returns:
            Tuple of (valid_edges, errors)
        """
        valid_edges = []
        errors = []
        if rejected_nodes is None:
            rejected_nodes = {}

        # Build set of valid node labels for quick lookup
        valid_labels = {node.get("label", "") for node in valid_nodes}
        
        for i, edge in enumerate(edges):
            try:
                # Validate edge type
                edge_type = edge.get("type", "")
                if edge_type not in self.edge_types:
                    errors.append(f"Edge {i}: Invalid type '{edge_type}'")
                    continue
                
                edge_config = self.edge_types[edge_type]
                
                # Validate source and target exist
                source = edge.get("source", "")
                target = edge.get("target", "")

                # Check source node with diagnostic information
                if source not in valid_labels:
                    if source in rejected_nodes:
                        errors.append(f"Edge {i}: Source node '{source}' was rejected ({rejected_nodes[source]})")
                    else:
                        errors.append(f"Edge {i}: Source node '{source}' not found in extraction")
                    continue

                # Check target node with diagnostic information
                if target not in valid_labels:
                    if target in rejected_nodes:
                        errors.append(f"Edge {i}: Target node '{target}' was rejected ({rejected_nodes[target]})")
                    else:
                        errors.append(f"Edge {i}: Target node '{target}' not found in extraction")
                    continue
                
                # Validate edge constraints (source/target type compatibility)
                if self._should_validate_edge_constraints():
                    source_node = next((n for n in valid_nodes if n.get("label") == source), None)
                    target_node = next((n for n in valid_nodes if n.get("label") == target), None)

                    # Validate constraints even if nodes not fully found (already validated existence above)
                    valid_sources = edge_config.get("valid_sources", [])
                    valid_targets = edge_config.get("valid_targets", [])

                    # Validate source node type constraint
                    if source_node:
                        source_type = source_node.get("type", "")
                        if valid_sources and source_type not in valid_sources:
                            errors.append(f"Edge {i}: Source type '{source_type}' not valid for edge type '{edge_type}'")
                            continue

                    # Validate target node type constraint
                    if target_node:
                        target_type = target_node.get("type", "")
                        if valid_targets and target_type not in valid_targets:
                            errors.append(f"Edge {i}: Target type '{target_type}' not valid for edge type '{edge_type}'")
                            continue
                
                # Validate confidence score
                try:
                    confidence = float(edge.get("confidence", 0))
                    if not (0.0 <= confidence <= 1.0):
                        errors.append(f"Edge {i}: Confidence {confidence} must be between 0.0 and 1.0")
                        continue
                except (ValueError, TypeError):
                    errors.append(f"Edge {i}: Invalid confidence score")
                    continue
                
                # Validate quote exists
                quote = edge.get("quote", "")
                if not quote:
                    errors.append(f"Edge {i}: Missing quote")
                    continue
                
                # Edge is valid
                valid_edges.append(edge)
                
            except Exception as e:
                errors.append(f"Edge {i}: Validation error - {str(e)}")
                continue
        
        return valid_edges, errors
    
    def _should_validate_edge_constraints(self) -> bool:
        """Check if edge type constraints should be validated."""
        # For MVP, always validate edge constraints if defined in schema
        return True
    
    def get_validation_summary(self) -> dict:
        """Get summary of validation configuration."""
        return {
            "node_types_count": len(self.node_types),
            "edge_types_count": len(self.edge_types),
            "validation_stages": 2,
            "excluded_validations": ["semantic_quote_verification", "misclassification_detection"]
        }