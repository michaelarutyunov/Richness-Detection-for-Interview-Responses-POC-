"""
Configurable Question Generator - Uses interview configuration from YAML.
Generates questions with configurable parameters instead of hardcoded values.
"""

import logging
from typing import Optional, List, Dict, Any
from src.core.models import Tactic, GraphState, InterviewState, Node
from src.llm.client import BaseLLMClient, LLMResponse
from src.llm.factory import LLMClientFactory
from src.interview.question_generation.warmup_generator import WarmupGenerator
from src.config.interview_config_loader import InterviewConfig

logger = logging.getLogger(__name__)


class ConfigurableQuestionGenerator:
    """
    Configurable question generator that uses interview settings from YAML.
    
    This generator replaces hardcoded values with configuration-driven behavior,
    making question generation truly configurable.
    """
    
    def __init__(self, config: InterviewConfig, llm_client: Optional[BaseLLMClient] = None):
        """Initialize the configurable question generator with interview configuration.
        
        Args:
            config: Interview configuration from YAML
            llm_client: Optional LLM client (uses configuration if not provided)
        """
        self.config = config
        self.llm_client = llm_client
        self.warmup_generator = WarmupGenerator(llm_client=llm_client) if llm_client else None
        
        # Use configuration values instead of hardcoded ones
        if not self.llm_client:
            # Create LLM client from configuration
            try:
                self.llm_client = LLMClientFactory.create_client(
                    provider=self.config.llm.default_provider,
                    api_key=os.getenv(f"{self.config.llm.default_provider.upper()}_API_KEY"),
                    model=self.config.llm.models.get(self.config.llm.default_provider),
                    temperature=self.config.llm.question_temperature,
                    max_tokens=self.config.llm.max_tokens
                )
                self.warmup_generator = WarmupGenerator(llm_client=self.llm_client)
            except Exception as e:
                logger.warning(f"Could not create LLM client from config: {e}")
                self.llm_client = None
        
        logger.info(f"ConfigurableQuestionGenerator initialized with {type(self.llm_client).__name__ if self.llm_client else 'fallback templates'}")
    
    async def generate_question(
        self,
        tactic: Tactic,
        graph_state: GraphState,
        interview_state: InterviewState,
        context_node: Optional[Node] = None
    ) -> str:
        """Generate a natural interview question using configuration values.
        
        Args:
            tactic: Selected tactic with templates and metadata
            graph_state: Current knowledge graph
            interview_state: Current interview state
            context_node: Specific node to focus question on
            
        Returns:
            Generated interview question using configuration
        """
        logger.info(f"Generating question with tactic: {tactic.id} using configuration")
        
        # If no LLM client, use configurable fallback templates
        if not self.llm_client:
            logger.debug("No LLM client available, using configurable template fallback")
            return self._generate_from_template_with_config(tactic, graph_state, context_node)
        
        try:
            # Build comprehensive context using configuration values
            context = self._build_generation_context_with_config(
                tactic, graph_state, interview_state, context_node
            )
            
            # Create prompt for LLM using configuration
            messages = self._create_generation_prompt_with_config(context)
            
            # Generate question using LLM with configuration
            response = await self.llm_client.generate_completion(messages)
            
            # Post-process the generated question using configuration
            question = self._post_process_question_with_config(response.content, context)
            
            logger.info(f"LLM generated question with config: {question}")
            return question
            
        except Exception as e:
            logger.error(f"LLM question generation failed with config: {e}")
            logger.info("Falling back to configurable template generation")
            return self._generate_from_template_with_config(tactic, graph_state, context_node)
    
    def _build_generation_context_with_config(self, tactic: Tactic, graph_state: GraphState, 
                                            interview_state: InterviewState, context_node: Optional[Node]) -> Dict[str, Any]:
        """Build comprehensive context using configuration values."""
        context = {
            "tactic": tactic,
            "graph_state": graph_state,
            "interview_state": interview_state,
            "context_node": context_node,
            "recent_questions": interview_state.question_history[-self.config.tactic_selection.recent_questions_count:],
            "graph_needs": {
                "target_depth": self.config.graph_needs.target_depth,
                "isolation_threshold": self.config.graph_needs.isolation_threshold
            },
            "extraction": {
                "confidence_threshold": self.config.extraction.confidence_threshold,
                "validation_stages": self.config.extraction.validation_stages
            }
        }
        
        # Add context node information if available
        if context_node:
            context["context_node"] = context_node
            context["node_visit_score"] = self._calculate_node_visit_score(context_node, interview_state)
            context["node_recency_score"] = self._calculate_node_recency_score(context_node, interview_state)
        
        return context
    
    def _create_generation_prompt_with_config(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create prompt for LLM using configuration values."""
        tactic = context["tactic"]
        graph_state = context["graph_state"]
        interview_state = context["interview_state"]
        
        # Build system prompt with configuration context
        system_prompt = f"""
        You are conducting a research interview about {tactic.name}.
        
        Context:
        - Target graph depth: {context['graph_needs']['target_depth']}
        - Isolation threshold: {context['graph_needs']['isolation_threshold']}
        - Extraction confidence: {context['extraction']['confidence_threshold']}
        - Recent questions to avoid: {len(context['recent_questions'])}
        
        Generate a natural, conversational question that:
        1. Addresses the {tactic.name} strategy
        2. Considers the current graph state ({graph_state.get_node_count()} nodes, {graph_state.get_edge_count()} edges)
        3. Avoids recent questions and repetition
        4. Uses temperature {self.config.llm.question_temperature} for natural conversation
        5. Stays within {self.config.llm.max_tokens} tokens
        """
        
        # Build user prompt with context
        user_prompt = f"""
        Current interview turn: {interview_state.turn_number}
        Selected tactic: {tactic.name}
        Graph context: {len(graph_state.nodes)} nodes, {len(graph_state.edges)} edges
        Recent questions: {context['recent_questions']}
        
        Generate a natural follow-up question that explores {tactic.name}.
        """
        
        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    
    def _post_process_question_with_config(self, question: str, context: Dict[str, Any]) -> str:
        """Post-process generated question using configuration values."""
        # Apply length limit from configuration
        if len(question) > self.config.question_generation.max_question_length:
            question = question[:self.config.question_generation.max_question_length - 3] + "..."
        
        # Clean up and format
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
        
        return question
    
    def _generate_from_template_with_config(self, tactic: Tactic, graph_state: GraphState, 
                                          context_node: Optional[Node]) -> str:
        """Generate question from template using configuration values."""
        logger.debug("Using configurable template fallback")
        
        # Use tactic templates with configuration-aware context
        if tactic.templates:
            # Select template based on configuration weights
            template = self._select_template_with_config(tactic, graph_state, context_node)
            
            # Interpolate with configuration-aware values
            return self._interpolate_template_with_config(template, graph_state, context_node)
        
        # Fallback to basic question
        return "Can you tell me more about that?"
    
    def _select_template_with_config(self, tactic: Tactic, graph_state: GraphState, 
                                   context_node: Optional[Node]) -> str:
        """Select template using configuration-aware logic."""
        if not tactic.templates:
            return "Can you tell me more about that?"
        
        # Simple selection - can be enhanced with configuration-aware logic
        # For now, use first template or context-aware selection
        if context_node and len(tactic.templates) > 1:
            # Select template based on node context and configuration
            return tactic.templates[0]  # Simplified for now
        
        return tactic.templates[0] if tactic.templates else "Can you tell me more about that?"
    
    def _interpolate_template_with_config(self, template: str, graph_state: GraphState, 
                                        context_node: Optional[Node]) -> str:
        """Interpolate template with configuration-aware values."""
        # Simple interpolation - can be enhanced with configuration-aware logic
        # For now, basic interpolation
        if context_node:
            return template.replace("{node}", context_node.label)
        
        return template
    
    def _calculate_node_visit_score(self, node: Node, interview_state: InterviewState) -> float:
        """Calculate node visit score using configuration weights."""
        # Use configuration weights for visit vs recency scoring
        visit_weight = self.config.question_generation.context_weights['visit_score']
        recency_weight = self.config.question_generation.context_weights['recency_score']
        
        # Simple implementation - can be enhanced
        return node.visit_count * visit_weight
    
    def _calculate_node_recency_score(self, node: Node, interview_state: InterviewState) -> float:
        """Calculate node recency score using configuration weights."""
        # Use configuration weights for recency scoring
        recency_weight = self.config.question_generation.context_weights['recency_score']
        
        # Simple implementation - can be enhanced
        return 1.0 / (node.last_visit_turn + 1) * recency_weight
    
    async def generate_behavioral_warmup(
        self,
        concept: str,
        category: str,
        seed_concepts: List[str],
        forbidden_attributes: List[str]
    ) -> str:
        """Generate a behavioral warm-up question using configuration values."""
        if not self.warmup_generator:
            return f"What are your thoughts about {concept}?"
        
        try:
            # Use warmup generator with configuration
            return await self.warmup_generator.generate_warmup(
                concept=concept,
                category=category,
                seed_concepts=seed_concepts,
                forbidden_attributes=forbidden_attributes,
                temperature=self.config.llm.question_temperature,
                max_tokens=self.config.llm.max_tokens
            )
        except Exception as e:
            logger.error(f"Warmup generation failed with config: {e}")
            return f"What are your thoughts about {concept}?"
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of question generation configuration."""
        return {
            "question_generation": {
                "temperature": self.config.question_generation.temperature,
                "max_tokens": self.config.question_generation.max_tokens,
                "max_question_length": self.config.question_generation.max_question_length,
                "context_weights": self.config.question_generation.context_weights
            },
            "llm": {
                "default_provider": self.config.llm.default_provider,
                "question_temperature": self.config.llm.question_temperature
            }
        }