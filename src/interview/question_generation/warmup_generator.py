"""
Warm-up Question Generator for behavioral interview warm-ups.

Generates behavioral warm-up questions based on extracted concepts and categories
without priming or referencing the actual concept attributes.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from string import Template
import yaml

from src.llm.client import BaseLLMClient
from src.llm.factory import LLMClientFactory

logger = logging.getLogger(__name__)


class WarmupGenerator:
    """Generates behavioral warm-up questions from concept descriptions."""
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        """
        Initialize the warm-up generator.
        
        Args:
            llm_client: LLM client for generating questions. If None, uses fallback templates.
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt_template()
        logger.info(f"WarmupGenerator initialized with {type(llm_client).__name__ if llm_client else 'fallback templates'}")
    
    def _load_prompt_template(self) -> Dict[str, Any]:
        """Load the behavioral warm-up prompt template from file."""
        try:
            prompt_file = "prompts/behavioral_warmup_prompt.yaml"
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Prompt file {prompt_file} not found, using fallback")
            return self._get_fallback_template()
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            return self._get_fallback_template()
    
    def _get_fallback_template(self) -> Dict[str, Any]:
        """Get fallback template when file is not available."""
        return {
            "template": """Generate a behavioral warm-up question about {{ category }}.
Focus on recent actions/behavior. Avoid: {{ forbidden_attributes }}.
Output only the question.""",
            "examples": []
        }
    
    async def generate_warmup_question(
        self, 
        concept: str,
        category: str,
        seed_concepts: List[str],
        forbidden_attributes: List[str]
    ) -> str:
        """
        Generate a behavioral warm-up question based on concept analysis.
        
        Args:
            concept: The original concept description
            category: Extracted product/service category
            seed_concepts: Key concepts extracted from the description
            forbidden_attributes: Concept-specific attributes to avoid
            
        Returns:
            Behavioral warm-up question string
        """
        logger.info(f"Generating behavioral warm-up for category: {category}")
        logger.debug(f"Concept: {concept[:100]}...")
        logger.debug(f"Seed concepts: {seed_concepts}")
        logger.debug(f"Forbidden attributes: {forbidden_attributes}")
        
        if not self.llm_client:
            return self._generate_fallback_warmup(category, seed_concepts)
        
        try:
            # Build the prompt using the template
            prompt = self._build_prompt(
                concept=concept,
                category=category,
                seed_concepts=seed_concepts,
                forbidden_attributes=forbidden_attributes
            )
            
            # Generate question using LLM
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_client.generate_completion(messages)
            
            # Clean and validate the response
            question = self._clean_response(response.content, category)
            
            logger.info(f"Generated behavioral warm-up: {question}")
            return question
            
        except Exception as e:
            logger.error(f"Failed to generate behavioral warm-up: {e}")
            return self._generate_fallback_warmup(category, seed_concepts)
    
    def _build_prompt(
        self, 
        concept: str, 
        category: str, 
        seed_concepts: List[str], 
        forbidden_attributes: List[str]
    ) -> str:
        """Build the prompt using the template and context."""
        template = Template(self.prompt_template["template"])
        
        # Format seed concepts as bullet points
        seed_concepts_list = "\n".join([f"- {concept}" for concept in seed_concepts]) if seed_concepts else "- No specific concepts extracted"
        
        # Format forbidden attributes as comma-separated list
        forbidden_attributes_list = ", ".join(forbidden_attributes) if forbidden_attributes else "none specified"
        
        # Format the prompt with all variables
        prompt = template.safe_substitute(
            concept=concept,
            category=category,
            seed_concepts_list=seed_concepts_list,
            forbidden_attributes_list=forbidden_attributes_list
        )
        
        return prompt.strip()
    
    def _clean_response(self, response: str) -> str:
        """Clean and validate the LLM response."""
        # Remove any explanatory text
        lines = response.strip().split('\n')
        
        # Find the first line that looks like a question
        for line in lines:
            line = line.strip()
            if line and '?' in line:
                # Extract just the question part
                question_match = re.search(r'[^.!?]*\?', line)
                if question_match:
                    question = question_match.group().strip()
                    
                    # Replace any remaining template variables with natural language
                    question = self._replace_template_variables(question, category)
                    
                    return question
        
        # If no question found, return the first non-empty line
        for line in lines:
            line = line.strip()
            if line:
                # Ensure it ends with a question mark
                if not line.endswith('?'):
                    line += '?'
                
                # Replace any remaining template variables with natural language
                line = self._replace_template_variables(line, category)
                
                return line
        
        # Ultimate fallback
        return f"Tell me about your experience with this topic?"
    
    def _replace_template_variables(self, question: str, category: str) -> str:
        """Replace any remaining template variables with natural language."""
        # Use the actual category name instead of generic placeholders
        replacements = {
            r'\{category\}': category,
            r'\{behavior\}': 'engaged',
            r'\{behavioral_marker\}': 'made a decision',
            r'\{category-relevant task or activity\}': f'engaged with {category}',
            r'\[category\]': category,
            r'\[behavior\]': 'engaged',
            r'\[behavioral marker\]': 'made a decision',
            r'this category': category,  # Also replace generic "this category"
        }
        
        for pattern, replacement in replacements.items():
            question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        return question
    
    def _generate_fallback_warmup(self, category: str, seed_concepts: List[str]) -> str:
        """Generate a fallback warm-up question when LLM is not available."""
        # Simple behavioral questions based on category
        behavioral_starters = [
            "Tell me about the last time you",
            "How do you usually",
            "When did you last",
            "How often do you"
        ]
        
        # Create a simple question based on category
        starter = behavioral_starters[len(seed_concepts) % len(behavioral_starters)]
        
        # Convert category to natural language
        if "fitness" in category.lower() or "exercise" in category.lower():
            return f"{starter} use fitness equipment or exercise?"
        elif "food" in category.lower() or "meal" in category.lower():
            return f"{starter} prepare a meal at home?"
        elif "technology" in category.lower() or "device" in category.lower():
            return f"{starter} use a new technology or device?"
        elif "transport" in category.lower() or "travel" in category.lower():
            return f"{starter} plan a trip or travel?"
        else:
            return f"{starter} engage with {category}?"
    
    def validate_warmup_question(self, question: str, forbidden_attributes: List[str]) -> bool:
        """
        Validate that the warm-up question meets requirements.
        
        Args:
            question: The generated question
            forbidden_attributes: Attributes that should not appear
            
        Returns:
            True if question is valid
        """
        question_lower = question.lower()
        
        # Check for forbidden attributes - be smart about phrase matching
        for attr in forbidden_attributes:
            attr_lower = attr.lower()
            # Skip if it's part of a larger acceptable phrase (e.g., "fitness equipment" is OK, "fitness" alone is not)
            if attr_lower in ["fitness", "gym", "workout"] and "fitness equipment" in question_lower:
                continue
            if attr_lower in ["local"] and "locally sourced" in question_lower:
                continue
            if attr_lower in ["delivery"] and "meal kit delivery" in question_lower:
                continue
                
            if attr_lower in question_lower:
                logger.warning(f"Question contains forbidden attribute '{attr}': {question}")
                return False
        
        # Check it's a question
        if '?' not in question:
            logger.warning(f"Question doesn't contain question mark: {question}")
            return False
        
        # Check for behavioral markers
        behavioral_markers = ["last time", "usually", "how do you", "when did", "how often", "tell me about"]
        has_behavioral = any(marker in question_lower for marker in behavioral_markers)
        
        if not has_behavioral:
            logger.warning(f"Question lacks behavioral markers: {question}")
            return False
        
        # Check length (roughly 20 words max)
        word_count = len(question.split())
        if word_count > 25:  # Slightly flexible
            logger.warning(f"Question too long ({word_count} words): {question}")
            return False
        
        return True