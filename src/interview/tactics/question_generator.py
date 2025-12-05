"""
LLM-based Question Generator - Creates natural interview questions using language models.
"""

import logging
from typing import Optional, List, Dict, Any
from src.core.models import SchemaTactic, GraphState, InterviewState, Node
from src.llm.client import BaseLLMClient, LLMResponse
from src.llm.factory import LLMClientFactory
from src.interview.question_generation.warmup_generator import WarmupGenerator


logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Generates natural interview questions using LLM providers.
    
    Replaces the basic template interpolation with sophisticated
    language model-based question generation that considers:
    - Graph context and current node
    - Selected tactic and strategy
    - Interview state and history
    - Natural language patterns
    """
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        """Initialize the question generator with LLM client."""
        self.llm_client = llm_client
        self.warmup_generator = WarmupGenerator(llm_client=llm_client) if llm_client else None
        self._last_response = None  # Store last LLM response for token tracking
        logger.info(f"QuestionGenerator initialized with {type(llm_client).__name__ if llm_client else 'fallback templates'}")
    
    async def generate_question(
        self,
        tactic: SchemaTactic,
        graph_state: GraphState,
        interview_state: InterviewState,
        context_node: Optional[Node] = None
    ) -> str:
        """
        Generate a natural interview question using LLM or fallback templates.
        
        Args:
            tactic: Selected tactic with templates and metadata
            graph_state: Current knowledge graph
            interview_state: Current interview state
            context_node: Specific node to focus question on
            
        Returns:
            Generated interview question
        """
        logger.info(f"Generating question with tactic: {tactic.id}")
        
        # If no LLM client, use fallback template interpolation
        if not self.llm_client:
            logger.debug("No LLM client available, using template fallback")
            return self._generate_from_template(tactic, graph_state, context_node)
        
        try:
            # Build comprehensive context for LLM
            context = self._build_generation_context(
                tactic, graph_state, interview_state, context_node
            )
            
            # Create prompt for LLM
            messages = self._create_generation_prompt(context)
            
            # Generate question using LLM
            response = await self.llm_client.generate_completion(messages)
            
            # Store response for token tracking
            self._last_response = response
            
            # Post-process the generated question
            question = self._post_process_question(response.content, context)
            
            logger.info(f"LLM generated question: {question}")
            return question
            
        except Exception as e:
            logger.error(f"LLM question generation failed: {e}")
            logger.info("Falling back to template generation")
            return self._generate_from_template(tactic, graph_state, context_node)
    
    async def generate_behavioral_warmup(
        self,
        concept: str,
        category: str,
        seed_concepts: List[str],
        forbidden_attributes: List[str]
    ) -> str:
        """
        Generate a behavioral warm-up question based on concept analysis.
        
        Args:
            concept: Original concept description
            category: Extracted product/service category
            seed_concepts: Key concepts from concept analysis
            forbidden_attributes: Concept-specific attributes to avoid
            
        Returns:
            Behavioral warm-up question
        """
        if not self.warmup_generator:
            logger.warning("No warmup generator available, using fallback")
            return f"Tell me about your experience with {category}?"
        
        try:
            warmup_question = await self.warmup_generator.generate_warmup_question(
                concept=concept,
                category=category,
                seed_concepts=seed_concepts,
                forbidden_attributes=forbidden_attributes
            )
            
            # Validate the generated question
            if self.warmup_generator.validate_warmup_question(warmup_question, forbidden_attributes):
                logger.info(f"Generated valid behavioral warm-up: {warmup_question}")
                return warmup_question
            else:
                logger.warning("Generated warm-up failed validation, using fallback")
                return f"Tell me about the last time you engaged with {category}?"
                
        except Exception as e:
            logger.error(f"Failed to generate behavioral warm-up: {e}")
            return f"Tell me about your experience with {category}?"
    
    def _build_generation_context(
        self,
        tactic: SchemaTactic,
        graph_state: GraphState,
        interview_state: InterviewState,
        context_node: Optional[Node]
    ) -> Dict[str, Any]:
        """Build comprehensive context for question generation."""
        context = {
            "tactic_id": tactic.id,
            "tactic_name": tactic.name,
            "tactic_description": tactic.description,
            "interview_turn": interview_state.turn_number,


            "graph_nodes": graph_state.get_node_count(),
            "graph_edges": graph_state.get_edge_count(),
            "recent_questions": interview_state.question_history[-3:],  # Last 3 questions
            "tactic_usage": dict(interview_state.tactic_usage),
        }
        
        # Add context node information if available
        if context_node:
            context["context_node"] = {
                "id": context_node.id,
                "label": context_node.label,
                "type": context_node.type,
                "visit_count": context_node.visit_count,
                "source_quotes": context_node.source_quotes[-2:],  # Recent quotes
            }
        else:
            # Use a relevant node from the graph
            relevant_node = self._select_relevant_node(graph_state, interview_state)
            if relevant_node:
                context["context_node"] = {
                    "id": relevant_node.id,
                    "label": relevant_node.label,
                    "type": relevant_node.type,
                    "visit_count": relevant_node.visit_count,
                    "source_quotes": relevant_node.source_quotes[-2:],
                }
        
        # Add graph structure information
        isolated_nodes = graph_state.get_isolated_nodes()
        context["isolated_nodes"] = [node.label for node in isolated_nodes[:3]]  # Top 3
        context["graph_density"] = round(graph_state.get_density(), 2)
        context["average_depth"] = round(graph_state.get_average_depth(), 1)
        
        logger.debug(f"Built generation context: {context}")
        return context
    
    def _select_relevant_node(self, graph_state: GraphState, interview_state: InterviewState) -> Optional[Node]:
        """Select the most relevant node for question context."""
        if not graph_state.nodes:
            return None
        
        # Priority: least visited nodes first, then by recency
        candidates = []
        for node in graph_state.nodes.values():
            # Score based on visit count (prefer less visited)
            visit_score = 1.0 / (node.visit_count + 1)
            
            # Score based on recency (prefer recently created)
            recency_score = 1.0 / (interview_state.turn_number - node.creation_turn + 1)
            
            # Combined score
            total_score = visit_score * 0.7 + recency_score * 0.3
            candidates.append((node, total_score))
        
        # Return highest scoring node
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _create_generation_prompt(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create a detailed prompt for the LLM."""
        # Build system prompt
        system_prompt = f"""You are an expert qualitative researcher conducting an in-depth interview. 
Your goal is to ask natural, engaging questions that help explore the participant's experiences and perspectives.

Context:
- Interview turn: {context['interview_turn']}


- Selected tactic: {context['tactic_name']} ({context['tactic_description']})

Guidelines:
1. Ask ONE clear, specific question
2. Make it sound natural and conversational
3. Relate to the participant's previous responses when possible
4. Avoid repetition of recent questions
5. Use the provided context to make questions relevant
6. End with a question mark
7. Keep questions open-ended when appropriate"""
        
        # Build user prompt with specific context
        user_prompt_parts = []
        
        # Add graph context
        if context.get("context_node"):
            node = context["context_node"]
            user_prompt_parts.append(f"Focus on exploring: '{node['label']}' (type: {node['type']})")
            
            if node["source_quotes"]:
                user_prompt_parts.append(f"Previous mentions: {', '.join(node['source_quotes'])}")
        
        # Add graph structure context
        if context.get("isolated_nodes"):
            user_prompt_parts.append(f"Related concepts to connect: {', '.join(context['isolated_nodes'])}")
        
        # Add interview context
        if context.get("recent_questions"):
            user_prompt_parts.append(f"Recent questions to avoid repeating: {', '.join(context['recent_questions'])}")
        
        # Add tactic-specific guidance
        if context["tactic_id"] == "emotional_contrast":
            user_prompt_parts.append("Explore any conflicting or contrasting feelings about this topic.")
        elif context["tactic_id"] == "relationship_dynamics":
            user_prompt_parts.append("Focus on how other people or relationships factor into this.")
        elif context["tactic_id"] == "sensory_details":
            user_prompt_parts.append("Ask about specific details they noticed or experienced.")
        elif context["tactic_id"] == "before_after":
            user_prompt_parts.append("Explore how things changed or compare before vs after.")
        elif context["tactic_id"] == "emotional_turning_point":
            user_prompt_parts.append("Ask about key moments when their perspective changed.")
        elif context["tactic_id"] == "vulnerability":
            user_prompt_parts.append("Gently explore challenging or difficult aspects.")
        
        user_prompt = "\n".join(user_prompt_parts)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        logger.debug(f"Generation prompt: {user_prompt}")
        return messages
    
    def _post_process_question(self, question: str, context: Dict[str, Any]) -> str:
        """Post-process the generated question."""
        # Clean up the question
        question = question.strip()
        
        # Ensure it ends with a question mark
        if not question.endswith("?"):
            question += "?"
        
        # Remove any system prompts or extra content
        lines = question.split("\n")
        question_lines = [line.strip() for line in lines if line.strip() and not line.startswith("Here")]
        
        # Take the first substantial line that looks like a question
        for line in question_lines:
            if len(line) > 10 and "?" in line:
                question = line.split("?")[0] + "?"
                break
        
        # Ensure it's not too long
        if len(question) > 200:
            question = question[:200] + "?"
        
        logger.debug(f"Post-processed question: {question}")
        return question
    
    def _generate_from_template(self, tactic: SchemaTactic, graph_state: GraphState, context_node: Optional[Node]) -> str:
        """Generate question using basic template interpolation (fallback)."""
        logger.debug(f"Using template fallback for tactic: {tactic.id}")
        
        if not tactic.templates:
            return "Can you tell me more about that?"
        
        # Select template (cycle through available templates)
        template_index = len(graph_state.nodes) % len(tactic.templates)
        template = tactic.templates[template_index]
        
        # Simple interpolation
        question = template
        
        # Replace [concept] with a relevant concept
        if "[concept]" in question:
            if context_node:
                concept = context_node.label
            elif graph_state.nodes:
                # Use a random node
                import random
                concept = random.choice(list(graph_state.nodes.values())).label
            else:
                concept = "this topic"
            
            question = question.replace("[concept]", concept)
        
        # Replace [emotion] if present
        if "[emotion]" in question:
            emotions = ["feeling", "experience", "reaction", "response"]
            question = question.replace("[emotion]", emotions[template_index % len(emotions)])
        
        # Replace [person] if present
        if "[person]" in question:
            people = ["someone", "the person involved", "they", "that individual"]
            question = question.replace("[person]", people[template_index % len(people)])
        
        # Ensure question format
        question = question.strip()
        if not question.endswith("?"):
            question += "?"
        
        logger.debug(f"Template-generated question: {question}")
        return question


class QuestionGenerationConfig:
    """Configuration for question generation."""
    
    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        fallback_to_templates: bool = True,
        include_context_node: bool = True,
        max_question_length: int = 200
    ):
        """Initialize question generation configuration."""
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_to_templates = fallback_to_templates
        self.include_context_node = include_context_node
        self.max_question_length = max_question_length