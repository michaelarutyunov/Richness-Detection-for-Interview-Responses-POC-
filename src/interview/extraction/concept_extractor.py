"""
Concept Extractor for initial concept analysis and seed node extraction.
Streamlined version for bootstrapping interviews with starting concepts.
"""

import logging
from typing import List, Dict, Any, Optional

from src.core.extraction_models import ExtractedNode, GraphDelta
from src.interview.extraction.extraction_prompt_builder import ExtractionPromptBuilder
from src.interview.extraction.extraction_validator import ExtractionValidator
from src.llm.client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Extracts seed concepts from initial product/concept descriptions."""
    
    def __init__(self, llm_client: BaseLLMClient, prompt_builder: ExtractionPromptBuilder,
                 validator: ExtractionValidator):
        """Initialize with dependencies."""
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.validator = validator
        logger.info("ConceptExtractor initialized")
    
    async def extract_seed_concepts(self, concept_description: str, max_concepts: int = 5) -> List[ExtractedNode]:
        """
        Extract initial seed concepts from product/concept description.
        
        Args:
            concept_description: Initial product or concept description
            max_concepts: Maximum number of concepts to extract
            
        Returns:
            List of extracted seed nodes for graph initialization
        """
        logger.info(f"Extracting seed concepts from: {concept_description[:100]}...")
        
        try:
            # Build concept extraction prompt
            messages = self._build_concept_extraction_prompt(concept_description, max_concepts)
            
            # Single LLM call for concept extraction
            response = await self.llm.generate_completion(messages)
            
            if not response or not response.content:
                logger.warning("No content in concept extraction response")
                return []
            
            # Parse extraction from response
            extraction = self._parse_concept_extraction(response.content)
            
            # Validate extracted concepts
            validated = self.validator.validate_extraction(extraction, concept_description)
            
            # Convert to ExtractedNode objects with Pydantic validation
            nodes = []
            for node_data in validated.get("nodes", []):
                if len(nodes) >= max_concepts:
                    break

                try:
                    # Use Pydantic validation for type safety
                    node = ExtractedNode.model_validate(node_data)
                    nodes.append(node)
                except Exception as e:
                    logger.warning(f"Node failed Pydantic validation, skipping: {e}")
                    logger.debug(f"Invalid node data: {node_data}")
                    continue
            
            logger.info(f"Extracted {len(nodes)} seed concepts")
            return nodes
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []
    
    def _build_concept_extraction_prompt(self, concept_description: str, max_concepts: int) -> List[dict]:
        """Build prompt for initial concept extraction."""
        system_prompt = """You are a concept extraction system for marketing research.
Extract the key concepts mentioned in product descriptions.
Focus on concrete features, benefits, and user outcomes.
Return a JSON array of concepts with type and description."""
        
        user_prompt = f"""# PRODUCT DESCRIPTION
{concept_description}

# EXTRACTION TASK
Extract up to {max_concepts} key concepts from this description.
Focus on:
- Concrete product features (attributes)
- User benefits (functional consequences) 
- Emotional/social impacts (psychosocial consequences)
- Core values addressed

Return JSON array:
[
  {{
    "type": "attribute|functional_consequence|psychosocial_consequence|value",
    "label": "concept_name",
    "description": "brief description"
  }}
]"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _parse_concept_extraction(self, content: str) -> dict:
        """Parse concept extraction from LLM response."""
        try:
            # Try to extract JSON array from response
            import json
            
            # Look for JSON array in response
            start_idx = content.find("[")
            end_idx = content.rfind("]")
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx + 1]
                concepts = json.loads(json_str)
                
                # Convert to extraction format
                nodes = []
                for concept in concepts:
                    if isinstance(concept, dict):
                        # Schema-agnostic: skip nodes with missing required fields
                        node_type = concept.get("type")
                        node_label = concept.get("label", "")

                        if not node_type:
                            logger.warning(f"Skipping node with missing type: label='{node_label}'")
                            continue

                        if not node_label:
                            logger.warning(f"Skipping node with missing label: type='{node_type}'")
                            continue

                        nodes.append({
                            "type": node_type,
                            "label": node_label,
                            "quote": concept.get("description", "")
                        })
                
                return {"nodes": nodes, "edges": []}
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse concept extraction JSON: {e}")
            logger.debug(f"Failed content (first 500 chars): {content[:500]}")

        # Fallback: return empty extraction
        return {"nodes": [], "edges": []}
    
    def _analyze_concept_category(self, concept_description: str) -> str:
        """Analyze concept description to suggest category."""
        concept_lower = concept_description.lower()
        
        # Simple keyword matching for category suggestion
        if any(word in concept_lower for word in ["food", "meal", "coffee", "drink", "restaurant"]):
            return "food and beverage"
        elif any(word in concept_lower for word in ["fitness", "gym", "workout", "exercise", "training"]):
            return "fitness equipment"
        elif any(word in concept_lower for word in ["technology", "device", "app", "software", "smart"]):
            return "technology and devices"
        elif any(word in concept_lower for word in ["transport", "car", "bike", "travel", "vehicle"]):
            return "transportation"
        elif any(word in concept_lower for word in ["health", "medical", "wellness", "therapy"]):
            return "health and wellness"
        else:
            return "general products and services"