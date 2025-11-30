"""
Relationship Extractor for extracting implicit and explicit relationships.

This component enhances graph extraction by:
1. Detecting causal language patterns in participant responses
2. Inferring implicit relationships based on temporal/logical connections
3. Assigning confidence scores to all relationships
4. Supporting multiple extraction modes (conservative, balanced, aggressive)
"""

import logging
import re
from typing import Any

from src.core.data_models import ExtractedEdge, ExtractedNode
from src.core.interview_graph import InterviewGraph

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """Extract both explicit and implicit relationships with confidence scores."""

    # Causal language patterns (ordered by strength)
    CAUSAL_PATTERNS = {
        "explicit": [
            (r"\b(causes?|results? in|leads? to|creates?)\b", 1.0),
            (r"\b(enables?|allows?|makes? it possible)\b", 0.95),
        ],
        "strong": [
            (r"\b(because|since|due to)\b", 0.9),
            (r"\b(so|therefore|thus|hence)\b", 0.85),
            (r"\b(means|that's why|which is why)\b", 0.8),
        ],
        "moderate": [
            (r"\b(then|after that|which allows?)\b", 0.7),
            (r"\b(and so|as a result)\b", 0.7),
            (r"\b(helps? me|lets? me)\b", 0.65),
        ],
        "weak": [
            (r"\b(related to|associated with|connected to)\b", 0.5),
            (r"\b(goes? with|comes? with)\b", 0.5),
            (r"\b(and|while|when)\b", 0.4),
        ],
    }

    def __init__(
        self,
        mode: str = "balanced",
        confidence_threshold: float = 0.6,
        enable_implicit_extraction: bool = True,
        max_inference_per_turn: int = 5,
    ):
        """
        Initialize relationship extractor.

        Args:
            mode: Extraction mode - "conservative", "balanced", or "aggressive"
            confidence_threshold: Minimum confidence to extract relationship
            enable_implicit_extraction: Whether to infer implicit relationships
            max_inference_per_turn: Maximum number of inferred relationships per turn
        """
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.enable_implicit_extraction = enable_implicit_extraction
        self.max_inference_per_turn = max_inference_per_turn

        # Adjust thresholds based on mode
        if mode == "conservative":
            self.confidence_threshold = max(0.8, confidence_threshold)
        elif mode == "aggressive":
            self.confidence_threshold = min(0.5, confidence_threshold)

        logger.info(
            f"RelationshipExtractor initialized: mode={mode}, "
            f"threshold={self.confidence_threshold}"
        )

    def enhance_extraction(
        self,
        participant_response: str,
        extracted_nodes: list[ExtractedNode],
        extracted_edges: list[ExtractedEdge],
        existing_graph: InterviewGraph,
    ) -> list[ExtractedEdge]:
        """
        Enhance edge extraction with implicit relationships.

        This is Stage 2 of the two-stage extraction pipeline.
        Stage 1 (LLM) has already extracted explicit edges.
        This stage infers additional implicit relationships.

        Args:
            participant_response: Original participant text
            extracted_nodes: Nodes extracted by LLM in this turn
            extracted_edges: Edges extracted by LLM in this turn
            existing_graph: Current graph state

        Returns:
            List of additional inferred edges (with confidence scores)
        """
        if not self.enable_implicit_extraction:
            logger.debug("Implicit extraction disabled, skipping")
            return []

        # Detect causal language in response
        causal_markers = self._detect_causal_language(participant_response)
        if not causal_markers:
            logger.debug("No causal language detected, skipping implicit extraction")
            return []

        logger.info(f"Detected {len(causal_markers)} causal markers in response")

        # Infer implicit relationships
        inferred_edges = self._infer_implicit_relationships(
            participant_response,
            extracted_nodes,
            extracted_edges,
            causal_markers,
        )

        # Filter by confidence threshold
        filtered_edges = [
            edge for edge in inferred_edges if edge.confidence >= self.confidence_threshold
        ]

        # Limit number of inferred edges
        if len(filtered_edges) > self.max_inference_per_turn:
            logger.warning(
                f"Limiting inferred edges from {len(filtered_edges)} to "
                f"{self.max_inference_per_turn}"
            )
            # Keep highest confidence edges
            filtered_edges.sort(key=lambda e: e.confidence, reverse=True)
            filtered_edges = filtered_edges[: self.max_inference_per_turn]

        logger.info(
            f"Inferred {len(filtered_edges)} additional relationships "
            f"(filtered from {len(inferred_edges)})"
        )

        return filtered_edges

    def _detect_causal_language(self, text: str) -> list[dict[str, Any]]:
        """
        Detect causal language markers in text.

        Args:
            text: Participant response text

        Returns:
            List of detected markers with positions and confidence scores
        """
        markers = []
        text_lower = text.lower()

        for strength_level, patterns in self.CAUSAL_PATTERNS.items():
            for pattern, base_confidence in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    markers.append(
                        {
                            "text": match.group(0),
                            "start": match.start(),
                            "end": match.end(),
                            "strength": strength_level,
                            "confidence": base_confidence,
                        }
                    )

        # Sort by position in text
        markers.sort(key=lambda m: m["start"])
        return markers

    def _infer_implicit_relationships(
        self,
        participant_response: str,
        extracted_nodes: list[ExtractedNode],
        existing_edges: list[ExtractedEdge],
        causal_markers: list[dict],
    ) -> list[ExtractedEdge]:
        """
        Infer implicit relationships based on causal markers and node proximity.

        Strategy:
        1. For each causal marker, find nodes mentioned before and after it
        2. Infer relationship type based on node types and marker
        3. Assign confidence based on marker strength and proximity
        4. Avoid duplicating existing edges

        Args:
            participant_response: Original text
            extracted_nodes: Nodes in this response
            existing_edges: Edges already extracted by LLM
            causal_markers: Detected causal language markers

        Returns:
            List of inferred edges
        """
        inferred = []

        if len(extracted_nodes) < 2:
            logger.debug("Not enough nodes for implicit relationship inference")
            return []

        # Create set of existing edge pairs to avoid duplicates
        existing_pairs = {
            (edge.source, edge.target) for edge in existing_edges
        }

        # For each causal marker, try to infer relationships
        for marker in causal_markers:
            # Find nodes mentioned before and after marker
            nodes_before = self._find_nodes_near_marker(
                extracted_nodes, participant_response, marker["start"], direction="before"
            )
            nodes_after = self._find_nodes_near_marker(
                extracted_nodes, participant_response, marker["end"], direction="after"
            )

            # Try to create relationships
            for node_before in nodes_before[:2]:  # Limit to 2 closest nodes
                for node_after in nodes_after[:2]:
                    # Skip if relationship already exists
                    if (node_before.label, node_after.label) in existing_pairs:
                        continue

                    # Infer edge type and confidence
                    edge = self._create_inferred_edge(
                        node_before,
                        node_after,
                        marker,
                        participant_response,
                    )

                    if edge:
                        inferred.append(edge)
                        existing_pairs.add((edge.source, edge.target))

        return inferred

    def _find_nodes_near_marker(
        self,
        nodes: list[ExtractedNode],
        text: str,
        marker_pos: int,
        direction: str = "before",
    ) -> list[ExtractedNode]:
        """
        Find nodes mentioned near a causal marker.

        Args:
            nodes: List of extracted nodes
            text: Full participant response
            marker_pos: Position of causal marker in text
            direction: "before" or "after" marker

        Returns:
            List of nodes, sorted by proximity to marker
        """
        text_lower = text.lower()
        nearby_nodes = []

        for node in nodes:
            # Find all occurrences of node label in text
            label_pattern = re.escape(node.label.replace("_", " "))
            matches = list(re.finditer(label_pattern, text_lower, re.IGNORECASE))

            # Also check quote
            if node.quote and node.quote.lower() in text_lower:
                quote_pos = text_lower.find(node.quote.lower())
                matches.append(type("Match", (), {"start": lambda: quote_pos, "end": lambda: quote_pos + len(node.quote)})())

            for match in matches:
                node_pos = match.start()

                # Check if node is in correct direction from marker
                if direction == "before" and node_pos < marker_pos:
                    distance = marker_pos - node_pos
                    nearby_nodes.append((distance, node))
                elif direction == "after" and node_pos > marker_pos:
                    distance = node_pos - marker_pos
                    nearby_nodes.append((distance, node))

        # Sort by proximity
        nearby_nodes.sort(key=lambda x: x[0])
        return [node for _, node in nearby_nodes]

    def _create_inferred_edge(
        self,
        source_node: ExtractedNode,
        target_node: ExtractedNode,
        marker: dict,
        full_text: str,
    ) -> ExtractedEdge | None:
        """
        Create an inferred edge between two nodes.

        Args:
            source_node: Source node (before marker)
            target_node: Target node (after marker)
            marker: Causal language marker
            full_text: Full participant response

        Returns:
            ExtractedEdge or None if no valid relationship
        """
        # Determine edge type based on node types and marker
        edge_type = self._infer_edge_type(
            source_node.type, target_node.type, marker["text"]
        )

        if not edge_type:
            return None

        # Calculate confidence
        # Base on marker strength, reduce slightly for inference
        confidence = marker["confidence"] * 0.85

        # Adjust based on extraction mode
        if self.mode == "conservative":
            confidence *= 0.9
        elif self.mode == "aggressive":
            confidence *= 1.1
            confidence = min(confidence, 1.0)

        # Extract quote around the relationship
        quote = self._extract_relationship_quote(
            source_node, target_node, full_text, marker
        )

        return ExtractedEdge(
            type=edge_type,
            source=source_node.label,
            target=target_node.label,
            quote=quote,
            confidence=confidence,
        )

    def _infer_edge_type(
        self, source_type: str, target_type: str, marker_text: str
    ) -> str | None:
        """
        Infer edge type based on node types and causal marker.

        Priority order:
        1. Upward relationships (leads_to) - means-end chain
        2. Enablement (enables) - practical causation
        3. Correlation (correlates_with) - associations

        Args:
            source_type: Type of source node
            target_type: Type of target node
            marker_text: Causal language marker text

        Returns:
            Edge type name or None if no valid type
        """
        marker_lower = marker_text.lower()

        # Upward relationships (means-end chain)
        upward_pairs = {
            ("attribute", "functional_consequence"): "leads_to",
            ("attribute", "psychosocial_consequence"): "leads_to",
            ("attribute", "value"): "leads_to",  # Direct skip possible
            ("functional_consequence", "psychosocial_consequence"): "leads_to",
            ("functional_consequence", "value"): "leads_to",
            ("psychosocial_consequence", "value"): "leads_to",
        }

        if (source_type, target_type) in upward_pairs:
            return upward_pairs[(source_type, target_type)]

        # Check for enablement language FIRST (before same-level check)
        enablement_words = ["enables", "allows", "lets", "means", "makes possible", "helps"]
        if any(word in marker_lower for word in enablement_words):
            # Valid for: attribute/functional → attribute/functional/psychosocial
            if source_type in ["attribute", "functional_consequence"] and target_type in [
                "attribute",
                "functional_consequence",
                "psychosocial_consequence",
            ]:
                return "enables"

        # Same-level relationships (fallback)
        if source_type == target_type:
            # Check for explicit correlation language
            if any(word in marker_lower for word in ["related", "associated", "with", "goes"]):
                return "correlates_with"
            # Default same-level enablement (if not caught above)
            elif source_type in ["attribute", "functional_consequence"]:
                return "enables"
            else:
                return "correlates_with"  # Fallback for psychosocial/value same-level

        # No valid relationship inferred
        return None

    def _extract_relationship_quote(
        self,
        source_node: ExtractedNode,
        target_node: ExtractedNode,
        full_text: str,
        marker: dict,
    ) -> str:
        """
        Extract a quote that captures the inferred relationship.

        Args:
            source_node: Source node
            target_node: Target node
            full_text: Full participant response
            marker: Causal marker dictionary

        Returns:
            Quote string capturing the relationship
        """
        # Try to extract sentence containing both nodes and marker
        sentences = re.split(r"[.!?]", full_text)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            source_in = source_node.label.replace("_", " ") in sentence_lower
            target_in = target_node.label.replace("_", " ") in sentence_lower
            marker_in = marker["text"] in sentence_lower

            if source_in and target_in and marker_in:
                return sentence.strip()

        # Fallback: combine node quotes with marker
        return f"{source_node.quote} {marker['text']} {target_node.quote}"
