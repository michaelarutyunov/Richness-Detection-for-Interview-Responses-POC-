"""
Interview Manager - Orchestrates the complete interview flow.

Coordinates opportunity ranking, question generation, and response processing.
"""

import logging
import time
from datetime import datetime

from src.core.data_models import (
    InterviewPhase,
    InterviewState,
    QuestionMethod,
    TurnLog,
)
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
        min_richness: float = 25.0,  # Increased from 10.0 to achieve 10-15 turn interviews
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
        self.turn_logs = []  # Store turn-by-turn logs for extended reporting
        self.turn_number = 0
        self.min_richness = min_richness
        self.max_turns = max_turns
        self._interview_started = False
        self.session_id = ""  # Will be set when interview starts
        self._start_time: datetime | None = None  # Track when interview started

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
        self._start_time = datetime.now()
        if not self.session_id:  # Only generate if not already set by UI
            self.session_id = self._start_time.strftime("%Y%m%d_%H%M%S")
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

        # Start timing
        turn_start_time = time.time()

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

        processing_time = time.time() - turn_start_time

        logger.info(
            f"Turn {self.turn_number}: Extracted {nodes_added} nodes, "
            f"{edges_added} edges (richness: {delta.richness_score:.2f})"
        )

        # Check if should continue
        should_terminate = not self.should_continue()
        if should_terminate:
            question = await self._get_closing_question()
            self.conversation_history.append({"role": "assistant", "content": question})

            # Create turn log for termination
            turn_log = TurnLog(
                session_id=self.session_id,
                turn_number=self.turn_number,
                timestamp=datetime.now(),
                schema_version="1.0",
                participant_response=participant_response,
                participant_response_length=len(participant_response),
                graph_delta=delta,
                processing_time_seconds=processing_time,
                interview_state=self._build_interview_state(should_terminate=True),
                question_generated=question,
                question_method=QuestionMethod.TEMPLATE,  # Closing questions use templates
                question_generation_time_seconds=0.0,
                errors=delta.extraction_metadata.get("validation_errors", []),
                warnings=delta.extraction_metadata.get("validation_warnings", []),
            )
            self.turn_logs.append(turn_log)

            return question

        # Select next opportunity
        # Build interview state to get current phase
        current_state = self._build_interview_state()
        opportunities = self.ranker.rank_opportunities(
            max_opportunities=5,
            current_turn=self.turn_number,
            interview_phase=current_state.phase
        )

        if not opportunities:
            # No opportunities - wrap up
            logger.info("No opportunities remaining")
            question = await self._get_closing_question()
            self.conversation_history.append({"role": "assistant", "content": question})

            # Create turn log
            turn_log = TurnLog(
                session_id=self.session_id,
                turn_number=self.turn_number,
                timestamp=datetime.now(),
                schema_version="1.0",
                participant_response=participant_response,
                participant_response_length=len(participant_response),
                graph_delta=delta,
                processing_time_seconds=processing_time,
                interview_state=self._build_interview_state(should_terminate=True),
                question_generated=question,
                question_method=QuestionMethod.TEMPLATE,  # Closing questions use templates
                question_generation_time_seconds=0.0,
                errors=delta.extraction_metadata.get("validation_errors", []),
                warnings=delta.extraction_metadata.get("validation_warnings", []),
            )
            self.turn_logs.append(turn_log)

            return question

        # Take best opportunity
        best_opportunity = opportunities[0]

        # Mark the node as visited
        self.graph.visit_node(best_opportunity.node_id, self.turn_number)

        # Update focus
        self.ranker.update_focus(best_opportunity.node_id)

        # Generate question
        question_start_time = time.time()
        question = await self.question_gen.generate_question(
            opportunity=best_opportunity,
            graph=self.graph,
            conversation_history=self.conversation_history,
        )
        question_gen_time = time.time() - question_start_time

        logger.info(f"Next question (strategy: {best_opportunity.strategy}): {question[:50]}...")

        # Add to history
        self.conversation_history.append({"role": "assistant", "content": question})

        # Build interview state snapshot
        interview_state = self._build_interview_state(
            best_opportunity=best_opportunity, should_terminate=False
        )

        # Create turn log with proper state and timing
        turn_log = TurnLog(
            session_id=self.session_id,
            turn_number=self.turn_number,
            timestamp=datetime.now(),
            schema_version="1.0",
            participant_response=participant_response,
            participant_response_length=len(participant_response),
            graph_delta=delta,
            processing_time_seconds=processing_time,
            interview_state=interview_state,
            question_generated=question,
            question_method=QuestionMethod.LLM,  # Generated via LLM
            question_generation_time_seconds=question_gen_time,
            reasoning_trace=self.question_gen.last_reasoning_trace,  # Capture K2-thinking trace
            errors=delta.extraction_metadata.get("validation_errors", []),
            warnings=delta.extraction_metadata.get("validation_warnings", []),
        )

        self.turn_logs.append(turn_log)

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

    def _build_interview_state(
        self, best_opportunity=None, should_terminate: bool = False
    ) -> InterviewState:
        """
        Build current interview state snapshot.

        Args:
            best_opportunity: Current best opportunity (if any)
            should_terminate: Whether interview should terminate

        Returns:
            InterviewState: Current state snapshot
        """
        # Calculate graph metrics
        node_count = self.graph.node_count
        edge_count = self.graph.edge_count
        coverage = self.graph.calculate_coverage()
        coverage_overall = coverage["overall"]  # Extract float from dict

        # Get focus stack from ranker
        focus_stack = (
            self.ranker._focus_stack.copy() if hasattr(self.ranker, "_focus_stack") else []
        )

        # Determine phase based on coverage and richness
        richness = self.graph.calculate_richness()
        if coverage_overall < 0.3:
            phase = InterviewPhase.COVERAGE
        elif richness < self.min_richness * 0.7:
            phase = InterviewPhase.DEPTH
        elif coverage_overall < 0.8:
            phase = InterviewPhase.CONNECTION
        else:
            phase = InterviewPhase.WRAP_UP

        return InterviewState(
            session_id=self.session_id,
            turn_number=self.turn_number,
            phase=phase,
            graph_node_count=node_count,
            graph_edge_count=edge_count,
            cumulative_richness=richness,
            coverage_pct=coverage_overall,
            avg_node_depth=0.0,  # TODO: Calculate if needed
            top_opportunity=best_opportunity,
            focus_stack=focus_stack,
            dead_end_nodes=[],  # TODO: Track if needed
            should_terminate=should_terminate,
            termination_reason="Target richness reached" if should_terminate else None,
            started_at=self._start_time or datetime.now(),
            last_response_at=datetime.now(),
        )

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

    def get_turn_logs(self) -> list[TurnLog]:
        """Get all turn logs for extended reporting."""
        return self.turn_logs
