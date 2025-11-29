"""
Prompt builder for LLM extraction tasks.

Loads YAML templates and builds prompts with schema context.
"""

import logging
from pathlib import Path

import yaml

from src.core.interview_graph import InterviewGraph

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds LLM prompts from YAML templates with schema context."""

    def __init__(self, prompts_path: str = "prompts/extraction_prompts.yaml"):
        """
        Initialize prompt builder.

        Args:
            prompts_path: Path to extraction prompts YAML file
        """
        self.prompts_path = Path(prompts_path)
        self._prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        """Load prompts from YAML file."""
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_path}")

        with open(self.prompts_path, encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        logger.info(f"Loaded prompts from {self.prompts_path}")
        return prompts

    def build_extraction_prompt(
        self,
        participant_response: str,
        conversation_history: list[dict[str, str]],
        existing_graph: InterviewGraph,
    ) -> tuple[list[dict[str, str]], dict]:
        """
        Build extraction prompt from template.

        Args:
            participant_response: Latest response from participant
            conversation_history: Recent conversation turns
            existing_graph: Current interview graph state

        Returns:
            Tuple of (messages, function_schema) for LLM API call
        """
        # Get graph extraction templates
        graph_extraction = self._prompts["graph_extraction"]

        # Build node types description
        node_types_lines = []
        for nt in existing_graph.schema.node_types:
            node_types_lines.append(
                f"- **{nt.name}**: {nt.description} (richness weight: {nt.richness_weight})"
            )
        node_types_description = "\n".join(node_types_lines)

        # Build edge types description
        edge_types_lines = []
        for et in existing_graph.schema.edge_types:
            valid_sources = ", ".join(et.valid_sources)
            valid_targets = ", ".join(et.valid_targets)
            edge_types_lines.append(
                f"- **{et.name}**: {et.description} "
                f"(richness boost: {et.richness_boost})\n"
                f"  - Valid: {valid_sources} → {valid_targets}"
            )
        edge_types_description = "\n".join(edge_types_lines)

        # Build existing nodes list
        existing_nodes_lines = []
        for node_id in existing_graph.graph.nodes():
            node_data = existing_graph.graph.nodes[node_id]["data"]
            existing_nodes_lines.append(f"- {node_data.label} ({node_data.type})")

        existing_nodes = (
            "\n".join(existing_nodes_lines) if existing_nodes_lines else "(empty graph)"
        )

        # Build conversation context (last 3 turns)
        conversation_lines = []
        for msg in conversation_history[-3:]:
            role = msg["role"].capitalize()
            content = msg["content"]
            conversation_lines.append(f"{role}: {content}")

        conversation_context = (
            "\n".join(conversation_lines) if conversation_lines else "(start of interview)"
        )

        # Format user prompt with context
        user_prompt = graph_extraction["user_prompt_template"].format(
            node_types_description=node_types_description,
            edge_types_description=edge_types_description,
            existing_nodes=existing_nodes,
            conversation_context=conversation_context,
            participant_response=participant_response,
        )

        # Build messages array
        messages = [
            {"role": "system", "content": graph_extraction["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ]

        # Get function calling schema
        function_schema = graph_extraction["function_calling_schema"]

        return messages, function_schema

    def get_function_schema(self) -> dict:
        """
        Get function calling schema for graph extraction.

        Returns:
            Dict: Function calling schema
        """
        return self._prompts["graph_extraction"]["function_calling_schema"]
