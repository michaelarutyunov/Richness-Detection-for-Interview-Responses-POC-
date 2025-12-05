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
        """Load extraction prompts from YAML file."""
        if not self.prompts_path.exists():
            logger.warning(f"Extraction prompts file not found: {self.prompts_path}, using defaults")
            return self._get_default_extraction_prompts()
        
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading extraction prompts: {e}, using defaults")
            return self._get_default_extraction_prompts()
    
    def _get_default_extraction_prompts(self) -> dict:
        """Default extraction prompts when file is not available."""
        return {
            "graph_extraction": {
                "system_prompt": """You extract graph information from interview responses.
You do not generate opinions, summaries, or explanations.
You detect specific concepts (nodes) and relationships (edges) from text.

Your behavior is fully driven by the Schema Context provided to you.
Use ONLY node types, edge types, and examples defined in the schema.

EXTRACTION PRINCIPLES:
1. Extract only what is present in the response
2. Support every extraction with a direct quote
3. Do not hallucinate - skip uncertain extractions
4. Merge with existing nodes when possible

Return JSON with nodes_added and edges_added.""",
                "user_prompt_template": """# SCHEMA CONTEXT
{node_types_description}
{edge_types_description}

# EXISTING GRAPH NODES
{existing_nodes}

# RECENT CONVERSATION
{conversation_context}

# PARTICIPANT RESPONSE
\"{participant_response}\"

# EXTRACTION TASK
Extract concepts and relationships using the schema examples as guidance.

Return JSON:
{"nodes_added": [...], "edges_added": [...]}""",
                "function_calling_schema": {
                    "name": "extract_graph_delta",
                    "description": "Extract nodes and edges from participant response",
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
                                        "quote": {"type": "string"}
                                    }
                                }
                            },
                            "edges_added": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["type", "source", "target", "quote", "confidence"],
                                    "properties": {
                                        "type": {"type": "string"},
                                        "source": {"type": "string"},
                                        "target": {"type": "string"},
                                        "quote": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    
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
        
        # Limit to avoid prompt bloat
        max_nodes = 10
        if len(existing_nodes) > max_nodes:
            nodes = existing_nodes[:max_nodes-1] + [f"... and {len(existing_nodes) - max_nodes + 1} more"]
        else:
            nodes = existing_nodes
        
        return "\n".join(f"- {node}" for node in nodes)