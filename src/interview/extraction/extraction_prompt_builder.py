"""
Schema-aware prompt builder for concept extraction.
Updated to use the new v0.2 schema format and extraction prompts.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
from src.core.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


class ExtractionPromptBuilder:
    """Builds LLM prompts for concept extraction using the new schema context."""
    
    def __init__(self, schema_path: str = "schemas/means_end_chain_v0.2.yaml", 
                 prompts_path: str = "prompts/extraction_prompts.yaml"):
        """Initialize with schema and prompts paths."""
        self.schema_loader = SchemaLoader(Path(schema_path))
        self.prompts_path = Path(prompts_path)
        self.extraction_prompts = self._load_extraction_prompts()
        logger.info(f"Initialized ExtractionPromptBuilder with schema: {schema_path}, prompts: {prompts_path}")
    
    def _load_extraction_prompts(self) -> dict:
        """Load extraction prompts from YAML file. Fails if file not found or malformed."""
        if not self.prompts_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Extraction prompts file not found: {self.prompts_path}. "
                f"System cannot operate without properly configured prompts."
            )

        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
                if not prompts:
                    raise ValueError(f"Extraction prompts file is empty: {self.prompts_path}")
                return prompts
        except yaml.YAMLError as e:
            raise ValueError(
                f"CRITICAL: Malformed extraction prompts file: {self.prompts_path}. "
                f"YAML parsing error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL: Failed to load extraction prompts from {self.prompts_path}: {e}"
            ) from e
    
    def _create_base_prompt(self) -> str:
        """Create base system prompt using the new extraction prompts format."""
        extraction_config = self.extraction_prompts.get("graph_extraction", {})
        system_prompt = extraction_config.get("system_prompt", "")
        
        # Add schema context to system prompt
        node_types_desc = self.schema_loader.format_node_types_for_prompt()
        edge_types_desc = self.schema_loader.format_edge_types_for_prompt()
        
        return f"""{system_prompt}

SCHEMA CONTEXT:
NODE TYPES:
{node_types_desc}

EDGE TYPES:
{edge_types_desc}

Remember: Use ONLY these defined types and follow the extraction principles."""
    
    # Remove the old _format_node_types and _format_edge_types methods
    # as they're now handled by the SchemaLoader
    
    def build_prompt(self, response: str, history: List[dict], 
                    existing_nodes: List[str]) -> Tuple[List[dict], dict]:
        """Build extraction prompt with context using new format."""
        # Build conversation context (last 2 turns)
        conversation_context = self._format_conversation_history(history)
        existing_nodes_context = self._format_existing_nodes(existing_nodes)
        
        # Get user prompt template from extraction prompts
        extraction_config = self.extraction_prompts.get("graph_extraction", {})
        user_template = extraction_config.get("user_prompt_template", "")
        
        # Fill in the template with actual context
        node_types_desc = self.schema_loader.format_node_types_for_prompt()
        edge_types_desc = self.schema_loader.format_edge_types_for_prompt()
        
        user_prompt = user_template.format(
            node_types_description=node_types_desc,
            edge_types_description=edge_types_desc,
            existing_nodes=existing_nodes_context,
            conversation_context=conversation_context,
            participant_response=response
        )
        
        messages = [
            {"role": "system", "content": self._create_base_prompt()},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use the function calling schema from the extraction prompts
        function_schema = extraction_config.get("function_calling_schema", {})
        
        logger.debug(f"Built extraction prompt for response: {response[:50]}...")
        return messages, function_schema
    
    def _format_conversation_history(self, history: List[dict]) -> str:
        """Format recent conversation history."""
        if not history:
            return "(start of interview)"
        
        # Use last 2 turns for context
        recent = history[-2:] if len(history) > 1 else history
        lines = []
        for msg in recent:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _format_existing_nodes(self, existing_nodes: List[str]) -> str:
        """Format existing nodes for context."""
        if not existing_nodes:
            return "(no existing concepts)"
        
        # Limit to avoid prompt bloat while maintaining sufficient context
        # Increased from 10 to 50 to reduce duplicate extractions in large graphs
        max_nodes = 50
        if len(existing_nodes) > max_nodes:
            nodes = existing_nodes[:max_nodes-1] + [f"... and {len(existing_nodes) - max_nodes + 1} more"]
        else:
            nodes = existing_nodes
        
        return "\n".join(f"- {node}" for node in nodes)