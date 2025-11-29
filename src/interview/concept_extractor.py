"""
Concept Extractor - Extract seed nodes from initial concept/stimulus.

Bootstraps the interview graph with initial nodes from the product/concept description.
"""

import logging

from src.core.data_models import GraphDelta, Node
from src.core.interview_graph import InterviewGraph
from src.interview.prompt_builder import PromptBuilder
from src.interview.validator import Validator
from src.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Extract seed nodes from concept description to bootstrap interview."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_builder: PromptBuilder,
        validator: Validator,
    ):
        """
        Initialize concept extractor.

        Args:
            llm_client: LLM client for extraction
            prompt_builder: Prompt builder for templates
            validator: Validator for extracted nodes
        """
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.validator = validator

    async def extract_seed_nodes(
        self, concept_description: str, schema_type: str = "means_end_chain"
    ) -> GraphDelta:
        """
        Extract seed nodes from concept description.

        Args:
            concept_description: Product/concept description text
            schema_type: Schema to use for extraction

        Returns:
            GraphDelta with seed nodes
        """
        logger.info(f"Extracting seed nodes from concept: {concept_description[:50]}...")

        # Build extraction prompt for concept
        messages, function_schema = self._build_concept_prompt(concept_description, schema_type)

        try:
            # Call LLM with retry
            llm_response = await self.llm.generate_with_retry(
                messages, function_schema, max_retries=2
            )

            # Handle no function call
            if not llm_response.function_call:
                logger.warning("LLM did not return function call for concept extraction")
                return self._build_empty_delta(
                    warning="No function call returned",
                    model_used=llm_response.model_used,
                    tokens=llm_response.tokens_used,
                    latency=llm_response.latency_ms,
                )

            # Validate extraction
            empty_graph = InterviewGraph(self.validator.schema)
            validation_result = self.validator.validate(
                llm_response.function_call, empty_graph, concept_description
            )

            # Build nodes from validated extraction
            nodes = self._build_nodes(validation_result.cleaned_nodes, turn_number=0)

            # Calculate richness (edges will be added later in interview)
            richness = sum(empty_graph.schema.get_richness_weight(node.type) for node in nodes)

            # Build metadata
            metadata = {
                "model_used": llm_response.model_used,
                "tokens_used": llm_response.tokens_used,
                "latency_ms": llm_response.latency_ms,
                "validation_errors": validation_result.errors,
                "validation_warnings": validation_result.warnings,
                "nodes_extracted": len(validation_result.cleaned_nodes),
                "edges_extracted": 0,  # Concept extraction focuses on nodes only
            }

            logger.info(
                f"Extracted {len(nodes)} seed nodes "
                f"(richness: {richness:.2f}, "
                f"{len(validation_result.errors)} errors, "
                f"{len(validation_result.warnings)} warnings)"
            )

            return GraphDelta(
                nodes_added=nodes,
                edges_added=[],
                richness_score=richness,
                extraction_metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return self._build_empty_delta(
                error=str(e), model_used=self.llm.model, tokens=0, latency=0
            )

    def _build_concept_prompt(
        self, concept_description: str, schema_type: str
    ) -> tuple[list[dict], dict]:
        """Build extraction prompt for concept description."""
        # Get schema context
        node_types = [
            f"- {nt.name}: {nt.description} (weight: {nt.richness_weight})"
            for nt in self.validator.schema.node_types
        ]

        # System prompt for concept extraction
        system_prompt = f"""You are extracting initial concepts from a product/concept description to start an interview.

Extract key attributes, benefits, or features mentioned in the description as graph nodes.

**Available Node Types:**
{chr(10).join(node_types)}

**Rules:**
1. Extract only concepts explicitly mentioned in the description
2. Use appropriate node types (typically 'attribute' or 'functional_consequence' for initial extraction)
3. Labels must be lowercase_with_underscores, 3-40 characters
4. Include exact quotes from the description
5. Focus on concrete, specific concepts (not vague generalizations)
6. Extract 2-5 seed nodes to bootstrap the interview

**Example:**
Description: "This coffee maker is affordable and makes great coffee quickly."
Extraction:
- attribute: affordable_price (quote: "affordable")
- attribute: coffee_quality (quote: "makes great coffee")
- functional_consequence: quick_brewing (quote: "makes coffee quickly")
"""

        # User prompt
        user_prompt = f"""Extract seed nodes from this concept description:

"{concept_description}"

Identify 2-5 key concepts to start the interview. Focus on concrete attributes or consequences mentioned."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Function schema
        function_schema = {
            "name": "extract_graph_delta",
            "description": "Extract seed nodes from concept description",
            "parameters": {
                "type": "object",
                "required": ["nodes_added", "edges_added"],
                "properties": {
                    "nodes_added": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "label", "quote"],
                            "properties": {
                                "type": {"type": "string"},
                                "label": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                        },
                    },
                    "edges_added": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Leave empty for concept extraction",
                    },
                },
            },
        }

        return messages, function_schema

    def _build_nodes(self, cleaned_nodes: list, turn_number: int) -> list[Node]:
        """Build Node objects from validated extraction."""
        nodes = []

        for extracted in cleaned_nodes:
            node = Node(
                id=extracted.label,
                type=extracted.type,
                label=extracted.label,
                source_quotes=[extracted.quote],
                creation_turn=turn_number,
                visit_count=0,  # Not yet visited in interview
            )
            nodes.append(node)

        return nodes

    def _build_empty_delta(
        self,
        error: str = None,
        warning: str = None,
        model_used: str = "",
        tokens: int = 0,
        latency: int = 0,
    ) -> GraphDelta:
        """Build empty delta with error/warning."""
        metadata = {
            "model_used": model_used,
            "tokens_used": tokens,
            "latency_ms": latency,
            "validation_errors": [],
            "validation_warnings": [],
            "nodes_extracted": 0,
            "edges_extracted": 0,
        }

        if error:
            metadata["error"] = error
        if warning:
            metadata["warning"] = warning

        return GraphDelta(
            nodes_added=[], edges_added=[], richness_score=0.0, extraction_metadata=metadata
        )
