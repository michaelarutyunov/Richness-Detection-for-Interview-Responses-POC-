"""
Question Generator for interview system.

Generates natural, conversational questions using templates and LLM.
"""

import logging
import random
import re
from pathlib import Path

import yaml

from src.core.interview_graph import InterviewGraph
from src.interview.opportunity_ranker import QuestionStrategy, RankedOpportunity
from src.interview.question_deduplicator import QuestionDeduplicator
from src.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generates interview questions from opportunities."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        templates_path: str = "prompts/question_templates.yaml",
        use_llm: bool = True,
        enable_repetition_detection: bool = True,
        word_overlap_threshold: float = 0.6,
        semantic_similarity_threshold: float = 0.75,
        history_window: int = 5,
        max_regeneration_attempts: int = 3,
    ):
        """
        Initialize question generator.

        Args:
            llm_client: Optional LLM client for natural question generation
            templates_path: Path to question templates YAML
            use_llm: Whether to use LLM for generation (vs templates only)
            enable_repetition_detection: Whether to detect and avoid repetitive questions
            word_overlap_threshold: Jaccard similarity threshold for word overlap (0.6 default)
            semantic_similarity_threshold: Threshold for semantic similarity (0.75 default)
            history_window: Number of recent questions to check for repetition (5 default)
            max_regeneration_attempts: Max attempts to regenerate if repetitive (3 default)
        """
        self.llm = llm_client
        self.use_llm = use_llm and llm_client is not None
        self._templates = self._load_templates(templates_path)
        self._question_history = []  # Track recent questions

        # Deduplication settings
        self.enable_repetition_detection = enable_repetition_detection
        self.max_regeneration_attempts = max_regeneration_attempts

        # Create deduplicator if enabled
        self.deduplicator = None
        if enable_repetition_detection:
            self.deduplicator = QuestionDeduplicator(
                word_overlap_threshold=word_overlap_threshold,
                semantic_similarity_threshold=semantic_similarity_threshold,
                history_window=history_window,
            )

    def _load_templates(self, templates_path: str) -> dict:
        """Load question templates from YAML."""
        path = Path(templates_path)
        if not path.exists():
            logger.warning(f"Templates file not found: {templates_path}")
            return self._get_default_templates()

        with open(path, encoding="utf-8") as f:
            templates = yaml.safe_load(f)

        logger.info(f"Loaded question templates from {templates_path}")
        return templates

    def _get_default_templates(self) -> dict:
        """Get fallback templates if file not found."""
        return {
            "templates": {
                "dig_deeper": ["Tell me more about {node}."],
                "connect_concepts": ["How do {node_a} and {node_b} relate?"],
                "introduce_topic": ["What are your thoughts on {topic}?"],
                "fallback": ["What else comes to mind?"],
            }
        }

    async def generate_question(
        self,
        opportunity: RankedOpportunity,
        graph: InterviewGraph,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate question for ranked opportunity with deduplication retry logic.

        Args:
            opportunity: Ranked opportunity to explore
            graph: Current interview graph
            conversation_history: Recent conversation for context

        Returns:
            str: Natural interview question
        """
        conversation_history = conversation_history or []

        # Deduplication retry loop
        for attempt in range(self.max_regeneration_attempts):
            # Try LLM generation first if enabled
            question = None
            if self.use_llm:
                try:
                    # Add anti-repetition context for attempts 2+
                    anti_repetition_context = None
                    if attempt > 0 and self.deduplicator:
                        anti_repetition_context = (
                            "IMPORTANT: The previous question was too similar to recent questions. "
                            "Generate a DIFFERENT question with a fresh angle or perspective."
                        )

                    question = await self._generate_with_llm(
                        opportunity, graph, conversation_history, anti_repetition_context
                    )
                    if question:
                        question = self._post_process_question(question)
                except Exception as e:
                    logger.warning(f"LLM question generation failed: {e}. Using templates.")

            # Fallback to templates if LLM didn't produce a question
            if not question:
                question = self._generate_from_template(opportunity, graph)
                question = self._post_process_question(question)

            # Check for repetition if deduplication enabled
            if self.deduplicator and self._question_history:
                is_repetitive, reason, similarity = self.deduplicator.is_repetitive(
                    question, self._question_history
                )

                if is_repetitive:
                    logger.info(
                        f"Attempt {attempt + 1}: Question repetitive ({reason}, "
                        f"similarity={similarity:.2f}). Regenerating..."
                    )
                    # If this is the last attempt, add variety phrase
                    if attempt == self.max_regeneration_attempts - 1:
                        question = self._add_variety_phrase(question)
                        logger.info("Max attempts reached. Adding variety phrase to force uniqueness.")
                        break
                    # Otherwise, continue to next attempt
                    continue

            # Question is not repetitive (or deduplication disabled), use it
            break

        # Add to history and return
        self._question_history.append(question)
        return question

    async def _generate_with_llm(
        self,
        opportunity: RankedOpportunity,
        graph: InterviewGraph,
        conversation_history: list[dict[str, str]],
        anti_repetition_context: str | None = None,
    ) -> str | None:
        """
        Generate question using LLM.

        Args:
            opportunity: Ranked opportunity to explore
            graph: Current interview graph
            conversation_history: Recent conversation for context
            anti_repetition_context: Optional instruction to avoid repetition

        Returns:
            Optional[str]: Generated question or None if failed
        """
        if not self.llm:
            return None

        # Build prompt
        llm_config = self._templates.get("llm_question_generation", {})
        system_prompt = llm_config.get("system_prompt", "")

        # Add anti-repetition instruction if provided
        if anti_repetition_context:
            system_prompt = f"{system_prompt}\n\n{anti_repetition_context}"

        # Format user prompt
        user_template = llm_config.get("user_prompt_template", "")

        # Get action description
        action_desc = self._get_action_description(opportunity)

        # Format target info
        target_info = f"{opportunity.node_label} ({opportunity.node_type})"

        # Format recent conversation (expanded from 3 to 6 turns)
        recent_conv = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history[-6:]
        )

        last_question = (
            conversation_history[-2]["content"]
            if len(conversation_history) >= 2 and conversation_history[-2]["role"] == "assistant"
            else "None"
        )

        user_prompt = user_template.format(
            opportunity_action=action_desc,
            target_info=target_info,
            recent_conversation=recent_conv if recent_conv else "(start of interview)",
            last_question=last_question,
        )

        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.llm.generate_with_retry(messages, max_retries=1)

        if response.content:
            return response.content.strip().strip('"').strip("'")

        return None

    def _generate_from_template(self, opportunity: RankedOpportunity, graph: InterviewGraph) -> str:
        """Generate question from template."""
        strategy = opportunity.strategy
        templates_dict = self._templates.get("templates", {})

        if strategy == QuestionStrategy.DIG_DEEPER:
            templates = templates_dict.get("dig_deeper", [])
            if templates:
                template = random.choice(templates)
                return template.replace("{node}", opportunity.node_label)

        elif strategy == QuestionStrategy.CONNECT_CONCEPTS:
            templates = templates_dict.get("connect_concepts", [])
            if templates:
                # Find a related node
                related_node = self._find_related_node(opportunity.node_id, graph)
                if related_node:
                    template = random.choice(templates)
                    return template.replace("{node_a}", opportunity.node_label).replace(
                        "{node_b}", related_node
                    )
                # Fallback to dig deeper if no related node
                return self._generate_from_template(
                    RankedOpportunity(
                        node_id=opportunity.node_id,
                        node_label=opportunity.node_label,
                        node_type=opportunity.node_type,
                        strategy=QuestionStrategy.DIG_DEEPER,
                        priority_score=opportunity.priority_score,
                        rationale=opportunity.rationale,
                        metadata=opportunity.metadata,
                    ),
                    graph,
                )

        elif strategy == QuestionStrategy.INTRODUCE_TOPIC:
            templates = templates_dict.get("introduce_topic", [])
            if templates:
                template = random.choice(templates)
                return template.replace("{topic}", opportunity.node_label)

        # Fallback
        fallback_templates = templates_dict.get("fallback", [])
        if fallback_templates:
            return random.choice(fallback_templates)

        return "What else would you like to share?"

    def _get_action_description(self, opportunity: RankedOpportunity) -> str:
        """Get human-readable action description."""
        if opportunity.strategy == QuestionStrategy.DIG_DEEPER:
            return f"probe deeper into '{opportunity.node_label}'"
        elif opportunity.strategy == QuestionStrategy.CONNECT_CONCEPTS:
            return f"explore connections from '{opportunity.node_label}'"
        elif opportunity.strategy == QuestionStrategy.INTRODUCE_TOPIC:
            return f"introduce the topic of '{opportunity.node_label}'"
        return "continue the conversation"

    def _find_related_node(self, node_id: str, graph: InterviewGraph) -> str | None:
        """Find a related node for connecting questions."""
        # Try successors first
        successors = list(graph.graph.successors(node_id))
        if successors:
            related_id = successors[0]
            return graph.graph.nodes[related_id]["data"].label

        # Try predecessors
        predecessors = list(graph.graph.predecessors(node_id))
        if predecessors:
            related_id = predecessors[0]
            return graph.graph.nodes[related_id]["data"].label

        return None

    def _post_process_question(self, question: str) -> str:
        """
        Post-process generated question.

        - Apply quality checks
        - Fix common issues
        - Ensure proper punctuation
        """
        question = question.strip()

        # Ensure ends with question mark
        if not question.endswith("?"):
            question += "?"

        # Auto-fix "Why" questions
        auto_fixes = self._templates.get("quality_checks", {}).get("auto_fixes", [])
        for fix in auto_fixes:
            pattern = fix.get("from", "")
            replacement = fix.get("to", "")
            if pattern and replacement:
                question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)

        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]

        return question

    def _add_variety_phrase(self, question: str) -> str:
        """
        Add variety phrase to make question unique when max attempts reached.

        Args:
            question: Base question

        Returns:
            str: Question with variety phrase prepended
        """
        variety_phrases = [
            "Building on that, ",
            "Following up, ",
            "I'm curious, ",
            "Let me ask differently: ",
            "From another angle, ",
            "Taking a step back, ",
        ]

        phrase = random.choice(variety_phrases)
        # Lowercase first letter of question when prepending
        if question:
            question = question[0].lower() + question[1:]

        return phrase + question

    def get_opening_question(self) -> str:
        """Get opening question to start interview."""
        return "What do you like most about this product?"

    def get_closing_question(self) -> str:
        """Get closing question to end interview."""
        return "Is there anything else you'd like to share before we wrap up?"
