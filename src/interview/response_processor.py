"""
Response Processor for extracting graph deltas from participant responses.

Orchestrates the extraction pipeline:
1. Build extraction prompt
2. Call LLM with retry
3. Parse function call result
4. Validate against schema
5. Build GraphDelta with metadata
"""

import logging

from src.core.data_models import Edge, GraphDelta, Node
from src.core.interview_graph import InterviewGraph
from src.interview.prompt_builder import PromptBuilder
from src.interview.validator import Validator
from src.llm.base_client import BaseLLMClient
from src.llm.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class ResponseProcessor:
    """Processes participant responses to extract graph deltas."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_builder: PromptBuilder,
        validator: Validator,
    ):
        """
        Initialize response processor.

        Args:
            llm_client: LLM client for extraction
            prompt_builder: Builds prompts from templates
            validator: Validates LLM output against schema
        """
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.validator = validator

    async def process_response(
        self,
        participant_response: str,
        conversation_history: list[dict[str, str]],
        existing_graph: InterviewGraph,
        turn_number: int,
    ) -> GraphDelta:
        """
        Extract graph delta from participant response.

        Pipeline stages:
        1. Build extraction prompt with schema context
        2. Call LLM with retry logic
        3. Parse function call result
        4. Validate against schema (4-stage validation)
        5. Build GraphDelta with metadata

        Args:
            participant_response: Latest response from participant
            conversation_history: Recent conversation turns
            existing_graph: Current graph state
            turn_number: Current turn number

        Returns:
            GraphDelta: Validated graph changes with metadata
        """
        logger.info(f"Processing response for turn {turn_number}")

        # Stage 1: Build prompt
        messages, function_schema = self.prompt_builder.build_extraction_prompt(
            participant_response=participant_response,
            conversation_history=conversation_history,
            existing_graph=existing_graph,
        )

        # Stage 2: Call LLM with retry
        try:
            llm_response = await self.llm.generate_with_retry(
                messages=messages, function_schema=function_schema
            )
        except LLMProviderError as e:
            logger.error(f"LLM call failed after retries: {e}")
            return GraphDelta(
                nodes_added=[],
                edges_added=[],
                richness_score=0.0,
                extraction_metadata={
                    "error": str(e),
                    "model_used": self.llm.model,
                    "latency_ms": 0,
                    "tokens_used": 0,
                },
            )

        # Stage 3: Parse function call
        if not llm_response.function_call:
            logger.warning("No function call in LLM response")
            return GraphDelta(
                nodes_added=[],
                edges_added=[],
                richness_score=0.0,
                extraction_metadata={
                    "model_used": llm_response.model_used,
                    "latency_ms": llm_response.latency_ms,
                    "tokens_used": llm_response.tokens_used,
                    "warning": "No function call returned",
                },
            )

        raw_extraction = llm_response.function_call

        # Stage 4: Validate
        validation_result = self.validator.validate(
            raw_extraction=raw_extraction,
            existing_graph=existing_graph,
            participant_response=participant_response,
        )

        if not validation_result.is_valid:
            logger.warning(
                f"Validation failed with {len(validation_result.errors)} errors: "
                f"{validation_result.errors[:3]}"
            )

        # Stage 5: Build GraphDelta
        nodes = self._build_nodes(validation_result.cleaned_nodes, turn_number)
        edges = self._build_edges(
            validation_result.cleaned_edges, turn_number, nodes, existing_graph
        )
        richness = self._calculate_richness(nodes, edges, existing_graph)

        delta = GraphDelta(
            nodes_added=nodes,
            edges_added=edges,
            richness_score=richness,
            extraction_metadata={
                "model_used": llm_response.model_used,
                "latency_ms": llm_response.latency_ms,
                "tokens_used": llm_response.tokens_used,
                "validation_errors": validation_result.errors,
                "validation_warnings": validation_result.warnings,
                "nodes_extracted": len(nodes),
                "edges_extracted": len(edges),
            },
        )

        logger.info(
            f"Extraction complete: {len(nodes)} nodes, {len(edges)} edges, "
            f"richness={richness:.2f} ({llm_response.latency_ms}ms, "
            f"{llm_response.tokens_used} tokens)"
        )

        return delta

    def _build_nodes(self, cleaned_nodes: list, turn_number: int) -> list[Node]:
        """
        Convert validated ExtractedNodes to Node objects.

        Args:
            cleaned_nodes: Validated nodes from validator
            turn_number: Current turn number

        Returns:
            List[Node]: Node objects ready for graph
        """
        nodes = []
        for extracted in cleaned_nodes:
            node = Node(
                id=extracted.label,  # Use label as ID
                type=extracted.type,
                label=extracted.label,
                source_quotes=[extracted.quote],
                creation_turn=turn_number,
                visit_count=0,  # Will be set to 1 when added to graph
            )
            nodes.append(node)

        return nodes

    def _build_edges(
        self,
        cleaned_edges: list,
        turn_number: int,
        new_nodes: list[Node],
        existing_graph: InterviewGraph,
    ) -> list[Edge]:
        """
        Convert validated ExtractedEdges to Edge objects.

        Args:
            cleaned_edges: Validated edges from validator
            turn_number: Current turn number
            new_nodes: Nodes being added in this delta
            existing_graph: Current graph state

        Returns:
            List[Edge]: Edge objects ready for graph
        """
        edges = []

        # Build label-to-id mapping
        label_to_id = {}
        for node in new_nodes:
            label_to_id[node.label] = node.id

        # Add existing nodes to mapping
        for node_id in existing_graph.graph.nodes():
            node_data = existing_graph.graph.nodes[node_id]["data"]
            label_to_id[node_data.label] = node_id

        for extracted in cleaned_edges:
            # Resolve source and target to node IDs
            source_id = label_to_id.get(extracted.source)
            target_id = label_to_id.get(extracted.target)

            if not source_id or not target_id:
                logger.warning(
                    f"Skipping edge {extracted.source}->{extracted.target}: "
                    f"Could not resolve node IDs"
                )
                continue

            # Generate unique edge ID: source-type-target
            edge_id = f"{source_id}-{extracted.type}-{target_id}"

            edge = Edge(
                id=edge_id,
                type=extracted.type,
                source=source_id,
                target=target_id,
                source_quote=extracted.quote,
                creation_turn=turn_number,
            )
            edges.append(edge)

        return edges

    def _calculate_richness(
        self, nodes: list[Node], edges: list[Edge], existing_graph: InterviewGraph
    ) -> float:
        """
        Calculate richness score for this delta.

        Args:
            nodes: Nodes being added
            edges: Edges being added
            existing_graph: Current graph state

        Returns:
            float: Richness score for this delta
        """
        richness = 0.0

        # Sum node richness
        for node in nodes:
            try:
                weight = existing_graph.schema.get_richness_weight(node.type)
                richness += weight
            except KeyError:
                logger.warning(f"Unknown node type for richness: {node.type}")

        # Sum edge richness
        for edge in edges:
            try:
                boost = existing_graph.schema.get_richness_boost(edge.type)
                richness += boost
            except KeyError:
                logger.warning(f"Unknown edge type for richness: {edge.type}")

        return richness
