"""
Question generation for interview agent.
Produces natural questions based on strategy and context.
"""

import logging
import re
import json
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field

from core.graph import Graph, Node
from core.history import History
from core.state import Momentum
from decision.strategy import Strategy, StrategySelector, FocusTarget
from utils.llm_manager import LLMManager, TaskType

logger = logging.getLogger(__name__)


class GeneratedQuestion(BaseModel):
    """Result of question generation."""
    question: str = Field(description="The generated question")
    strategy_id: str = Field(description="Strategy that guided generation")
    tactic_used: Optional[str] = Field(default=None, description="Tactic if reported")
    rationale: str = Field(default="", description="Why this question was chosen")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    llm_response: Optional[Any] = Field(default=None, description="LLMResponse object for token tracking")


class QuestionGenerator:
    """
    Generates interview questions based on strategy and context.
    Uses LLM for natural language generation.
    Includes deduplication to prevent repetitive questions.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        strategy_selector: StrategySelector,
        similarity_threshold: float = 0.85
    ):
        self.llm = llm_manager
        self.strategy_selector = strategy_selector
        self.similarity_threshold = similarity_threshold
    
    def generate(
        self,
        strategy: Strategy,
        focus: FocusTarget,
        graph: Graph,
        history: History,
        momentum: Momentum,
        max_retries: int = 2
    ) -> GeneratedQuestion:
        """
        Generate a natural question fulfilling the strategy intent.
        Includes deduplication to prevent repetitive questions.

        Args:
            strategy: Selected strategy
            focus: What to target
            graph: Current knowledge graph
            history: Conversation history
            momentum: Current engagement level
            max_retries: Maximum retries if duplicate detected

        Returns:
            GeneratedQuestion with question text and metadata
        """
        # Log suggested tactics for this strategy
        if strategy.suggested_tactics:
            logger.info(f"[Tactic] Suggested for '{strategy.id}': {strategy.suggested_tactics}")
        else:
            logger.debug(f"[Tactic] No tactics suggested for '{strategy.id}'")

        # Check plausibility for connect_isolate strategy
        plausibility_llm_response = None
        if strategy.id == "connect_isolate" and focus.node:
            candidates = self._get_connection_candidates(graph, focus.node)
            plausible, plausibility_llm_response = self.assess_connection_plausibility(focus.node, candidates)
            viable = [(n, s) for n, s in plausible if s > 0.5]

            if not viable:
                logger.info(f"[Plausibility] No plausible connections for '{focus.node.label}'")
                return GeneratedQuestion(
                    question="",
                    strategy_id=strategy.id,
                    metadata={"no_plausible_connections": True, "isolated_node_id": focus.node.id},
                    llm_response=plausibility_llm_response
                )

        # Get recent questions for deduplication
        recent_questions = history.get_recent_questions(n=6)

        system_prompt = self._build_system_prompt(strategy, momentum)
        user_prompt = self._build_user_prompt(strategy, focus, graph, history)

        question = None
        for attempt in range(max_retries + 1):
            llm_response = self.llm.complete(
                task=TaskType.QUESTION_GENERATION,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7 + (attempt * 0.1)  # Increase temperature on retry
            )

            if not llm_response.success:
                # Fallback question
                return GeneratedQuestion(
                    question="Can you tell me more about that?",
                    strategy_id=strategy.id,
                    rationale=f"Fallback due to error: {llm_response.error}",
                    llm_response=llm_response
                )

            # Extract question from response
            question = self._extract_question(llm_response.content)

            # Check for duplicate
            if not self._is_duplicate(question, recent_questions):
                break

            logger.warning(f"[Dedup] Attempt {attempt + 1}: duplicate detected, retrying...")

            # Add anti-repetition hint for retry
            if attempt < max_retries:
                user_prompt = self._add_dedup_hint(user_prompt, question)

        if self._is_duplicate(question, recent_questions):
            logger.error(f"[Dedup] Failed after {max_retries} retries, using last attempt")

        result = GeneratedQuestion(
            question=question,
            strategy_id=strategy.id,
            rationale=strategy.intent,
            llm_response=llm_response
        )
        # Store plausibility check response if it exists
        if plausibility_llm_response:
            result.metadata["plausibility_llm_response"] = plausibility_llm_response
        return result
    
    def generate_clarification(
        self,
        ambiguity_reason: str,
        response: str,
        history: History
    ) -> GeneratedQuestion:
        """
        Generate a clarification question when response isn't extractable.
        
        Args:
            ambiguity_reason: Why clarification is needed
            response: The unclear response
            history: Conversation history
            
        Returns:
            GeneratedQuestion for clarification
        """
        system_prompt = """You are a skilled qualitative interviewer.

The respondent's answer was too vague or unclear to work with. You need to ask a clarification question that:
1. Acknowledges what they said
2. Gently asks them to be more specific
3. Doesn't suggest answers or lead them
4. Feels natural and conversational

Keep the question short and simple. One question only.

Output only the question - no preamble, explanation, or quotes around it."""

        user_prompt = f"""Recent conversation:
{history.format_for_prompt(n=2)}

Their response that needs clarification:
"{response}"

Why it needs clarification: {ambiguity_reason}

Generate a natural clarification question."""

        llm_response = self.llm.complete(
            task=TaskType.QUESTION_GENERATION,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.6
        )
        
        if not llm_response.success:
            return GeneratedQuestion(
                question="Could you help me understand what you mean by that?",
                strategy_id="clarification",
                rationale="Fallback clarification",
                llm_response=llm_response
            )

        question = self._extract_question(llm_response.content)

        return GeneratedQuestion(
            question=question,
            strategy_id="clarification",
            rationale=f"Clarifying: {ambiguity_reason}",
            llm_response=llm_response
        )
    
    def generate_opening(
        self,
        concept_text: str,
        history: History
    ) -> GeneratedQuestion:
        """
        Generate the opening question that introduces the concept.
        
        Args:
            concept_text: The stimulus concept
            history: Conversation history (usually empty)
            
        Returns:
            Opening question
        """
        system_prompt = """You are starting a qualitative interview about a product concept.

        Your opening question should:
        1. Present the concept naturally (don't read it verbatim)
        2. Invite their initial, unfiltered reaction
        3. Be open-ended - not leading toward any particular response
        4. Feel conversational, not formal

        The goal is to see what they spontaneously notice and react to.

        Output only the question - no preamble, explanation, or quotes around it."""

        user_prompt = f"""Here is the concept to discuss:

        ---
        {concept_text}
        ---

        Generate an opening question that invites their initial reaction to this concept."""

        llm_response = self.llm.complete(
            task=TaskType.QUESTION_GENERATION,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7
        )
        
        if not llm_response.success:
            return GeneratedQuestion(
                question="I'd like you to read this concept and share your initial thoughts. What stands out to you?",
                strategy_id="opening",
                rationale="Fallback opening",
                llm_response=llm_response
            )

        question = self._extract_question(llm_response.content)

        return GeneratedQuestion(
            question=question,
            strategy_id="opening",
            rationale="Opening question for concept exposure",
            llm_response=llm_response
        )

    def assess_connection_plausibility(
        self,
        isolated_node: Node,
        candidate_nodes: List[Node]
    ) -> Tuple[List[Tuple[Node, float]], Optional[Any]]:
        """
        Assess which nodes could plausibly connect to the isolated node.
        Returns tuple of (list of (node, plausibility_score) sorted by score descending, llm_response).
        """
        if not candidate_nodes:
            return [], None

        system_prompt = """You are assessing whether concepts from an interview could plausibly be connected.

Two concepts are plausibly connected if:
- They share a common context or domain
- One could causally influence the other
- They represent different aspects of the same experience
- A respondent might naturally link them in their thinking

Two concepts are NOT plausibly connected if:
- They address completely different topics
- Connecting them would require a forced or artificial link
- They exist in separate experiential domains

Return JSON: {"connections": [{"label": "...", "plausibility": 0.0-1.0, "reasoning": "..."}]}"""

        candidates_text = "\n".join([
            f"- \"{n.label}\" (type: {n.node_type})"
            for n in candidate_nodes[:10]
        ])

        user_prompt = f"""Isolated concept: "{isolated_node.label}" (type: {isolated_node.node_type})

Candidate concepts to potentially connect:
{candidates_text}

For each candidate, assess plausibility of a meaningful connection."""

        response = self.llm.complete(
            task=TaskType.PLAUSIBILITY_CHECK,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2
        )

        if not response.success:
            return [(n, 0.5) for n in candidate_nodes], response

        try:
            data = json.loads(response.content)
            label_to_node = {n.label: n for n in candidate_nodes}
            results = []
            for conn in data.get("connections", []):
                label = conn.get("label")
                plausibility = conn.get("plausibility", 0.5)
                if label in label_to_node:
                    results.append((label_to_node[label], plausibility))
            return sorted(results, key=lambda x: x[1], reverse=True), response
        except Exception:
            return [(n, 0.5) for n in candidate_nodes], response

    def _get_connection_candidates(self, graph: Graph, isolated_node: Node) -> List[Node]:
        """Get candidate nodes for connecting an isolated node."""
        return [
            n for n in graph.nodes.values()
            if n.id != isolated_node.id and not n.is_ambiguous
        ]

    def _build_system_prompt(self, strategy: Strategy, momentum: Momentum) -> str:
        """Build system prompt for question generation."""
        # Get tactic descriptions
        tactics_text = self.strategy_selector.format_tactics_for_prompt(strategy)
        
        # Momentum guidance
        momentum_guidance = ""
        if momentum.level == "high":
            momentum_guidance = """
            The respondent is highly engaged. Follow their energy - you can probe deeper 
            and ask more challenging questions. They're in flow."""
        elif momentum.level == "low":
            momentum_guidance = """
            The respondent seems less engaged. Keep your question simple and 
            approachable. Consider acknowledging what they've shared before asking more."""
        
        return f"""You are a skilled qualitative interviewer conducting a concept evaluation.

            ## Your Intent
            {strategy.intent}

            ## Guidance
            {strategy.llm_guidance}

            ## Suggested Approaches
            {tactics_text}

            {momentum_guidance}

            ## Rules
            1. Generate ONE question only
            2. Keep it natural and conversational
            3. Don't lead or suggest answers
            4. Don't be repetitive with recent questions
            5. Match the tone to their engagement level
            6. Use their language when referencing what they've said

            Output only the question - no preamble, explanation, or quotes around it."""

    def _build_user_prompt(
        self,
        strategy: Strategy,
        focus: FocusTarget,
        graph: Graph,
        history: History
    ) -> str:
        """Build user prompt for question generation."""
        parts = []
        
        # Conversation context
        parts.append("## Recent Conversation")
        parts.append(history.format_for_prompt(n=4))
        parts.append("")
        
        # Graph context
        parts.append("## Current Understanding")
        parts.append(graph.summary())
        parts.append("")
        
        # Focus
        parts.append("## Focus for This Question")
        parts.append(focus.describe())
        parts.append("")
        
        parts.append("Generate the next question.")
        
        return "\n".join(parts)
    
    def _extract_question(self, content: str) -> str:
        """Extract clean question from LLM response."""
        # Remove any markdown or extra formatting
        question = content.strip()

        # Remove quotes if present
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        if question.startswith("'") and question.endswith("'"):
            question = question[1:-1]

        # Remove "Question:" prefix if present
        prefixes = ["Question:", "Q:", "Next question:"]
        for prefix in prefixes:
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()

        return question

    def _is_duplicate(self, candidate: str, recent_questions: List[str]) -> bool:
        """
        Check if candidate question is duplicate of recent questions.

        Args:
            candidate: The candidate question to check
            recent_questions: List of recent questions

        Returns:
            True if duplicate detected
        """
        if not candidate or not recent_questions:
            return False

        candidate_norm = self._normalize(candidate)

        for recent in recent_questions:
            recent_norm = self._normalize(recent)

            # Exact match check
            if candidate_norm == recent_norm:
                logger.debug(f"[Dedup] Exact match detected")
                return True

            # Semantic similarity check (Jaccard)
            similarity = self._similarity(candidate_norm, recent_norm)
            if similarity > self.similarity_threshold:
                logger.debug(f"[Dedup] Similar question detected ({similarity:.0%})")
                return True

        return False

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove punctuation and lowercase
        normalized = re.sub(r'["\'\?\.!,;:\-]', '', text.lower().strip())
        # Collapse whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _similarity(self, a: str, b: str) -> float:
        """Calculate Jaccard similarity between two normalized strings."""
        words_a = set(a.split())
        words_b = set(b.split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)

    def _add_dedup_hint(self, user_prompt: str, duplicate_question: str) -> str:
        """Add anti-repetition hint to user prompt for retry."""
        hint = f"\n\n## IMPORTANT: DO NOT ask this question (too similar to recent):\n\"{duplicate_question}\"\n\nGenerate a DIFFERENT question that approaches the topic from a fresh angle."
        return user_prompt + hint
