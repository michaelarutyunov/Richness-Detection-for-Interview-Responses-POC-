"""
Interview Manager - Orchestrates the complete interview flow.

Coordinates opportunity ranking, question generation, and response processing.
"""

import logging

from src.core.interview_graph import InterviewGraph
from src.core.schema_manager import SchemaManager
from src.interview.opportunity_ranker import OpportunityRanker
from src.interview.prompt_builder import PromptBuilder
from src.interview.question_generator import QuestionGenerator
from src.interview.response_processor import ResponseProcessor
from src.interview.validator import Validator
from src.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class InterviewManager:
    """Manages complete interview flow."""

    def __init__(
        self,
        schema_manager: SchemaManager,
        extraction_client: BaseLLMClient,
        question_client: BaseLLMClient | None = None,
        min_richness: float = 10.0,
        max_turns: int = 20,
    ):
        """
        Initialize interview manager.

        Args:
            schema_manager: Schema manager for validation
            extraction_client: LLM client for graph extraction
            question_client: LLM client for question generation (optional)
            min_richness: Minimum richness threshold to stop
            max_turns: Maximum interview turns
        """
        self.schema = schema_manager
        self.graph = InterviewGraph(schema_manager)

        # Initialize components
        self.ranker = OpportunityRanker(self.graph)
        self.question_gen = QuestionGenerator(llm_client=question_client)

        # Response processing pipeline
        prompt_builder = PromptBuilder()
        validator = Validator(schema_manager)
        self.response_processor = ResponseProcessor(
            llm_client=extraction_client,
            prompt_builder=prompt_builder,
            validator=validator,
        )

        # Interview state
        self.conversation_history = []
        self.turn_number = 0
        self.min_richness = min_richness
        self.max_turns = max_turns
        self._interview_started = False

    async def start_interview(self) -> str:
        """
        Start interview with opening question.

        Returns:
            str: Opening question
        """
        if self._interview_started:
            logger.warning("Interview already started")
            return "We're already talking! What else would you like to share?"

        self._interview_started = True
        self.turn_number = 0

        opening_question = self.question_gen.get_opening_question()

        self.conversation_history.append({"role": "assistant", "content": opening_question})

        logger.info("Interview started")
        return opening_question

    async def process_response(self, participant_response: str) -> str:
        """
        Process participant response and generate next question.

        Args:
            participant_response: Participant's response text

        Returns:
            str: Next interview question
        """
        if not self._interview_started:
            logger.warning("Interview not started yet")
            return await self.start_interview()

        self.turn_number += 1

        # Add response to history
        self.conversation_history.append({"role": "user", "content": participant_response})

        # Extract graph delta from response
        delta = await self.response_processor.process_response(
            participant_response=participant_response,
            conversation_history=self.conversation_history,
            existing_graph=self.graph,
            turn_number=self.turn_number,
        )

        # Apply delta to graph
        nodes_added, edges_added = self.graph.apply_delta(delta, self.turn_number)

        logger.info(
            f"Turn {self.turn_number}: Extracted {nodes_added} nodes, "
            f"{edges_added} edges (richness: {delta.richness_score:.2f})"
        )

        # Check if should continue
        if not self.should_continue():
            question = await self._get_closing_question()
            self.conversation_history.append({"role": "assistant", "content": question})
            return question

        # Select next opportunity
        opportunities = self.ranker.rank_opportunities(max_opportunities=5)

        if not opportunities:
            # No opportunities - wrap up
            logger.info("No opportunities remaining")
            question = await self._get_closing_question()
            self.conversation_history.append({"role": "assistant", "content": question})
            return question

        # Take best opportunity
        best_opportunity = opportunities[0]

        # Update focus
        self.ranker.update_focus(best_opportunity.node_id)

        # Generate question
        question = await self.question_gen.generate_question(
            opportunity=best_opportunity,
            graph=self.graph,
            conversation_history=self.conversation_history,
        )

        logger.info(f"Next question (strategy: {best_opportunity.strategy}): {question[:50]}...")

        # Add to history
        self.conversation_history.append({"role": "assistant", "content": question})

        return question

    def should_continue(self) -> bool:
        """
        Determine if interview should continue.

        Returns:
            bool: True if should continue
        """
        return self.ranker.should_continue(
            current_turn=self.turn_number,
            min_richness=self.min_richness,
            max_turns=self.max_turns,
        )

    async def _get_closing_question(self) -> str:
        """Get closing question."""
        return self.question_gen.get_closing_question()

    def get_summary(self) -> dict:
        """
        Get interview summary.

        Returns:
            Dict with interview statistics
        """
        ranker_summary = self.ranker.get_summary()

        return {
            **ranker_summary,
            "turns": self.turn_number,
            "questions_asked": len(
                [m for m in self.conversation_history if m["role"] == "assistant"]
            ),
            "is_complete": not self.should_continue(),
        }

    def export_graph(self, path: str):
        """
        Export interview graph to file.

        Args:
            path: Output file path
        """
        self.graph.export_graphml(path)
        logger.info(f"Exported interview graph to {path}")

    def get_conversation_transcript(self) -> list[dict[str, str]]:
        """Get full conversation transcript."""
        return self.conversation_history.copy()
