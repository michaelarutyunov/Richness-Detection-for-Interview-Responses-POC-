"""
GraphDrivenOrchestrator - Main orchestration pipeline for graph-driven interviews.
"""

import logging
from typing import Optional, List, Dict, Any
from src.core.models import GraphState, InterviewState, SchemaTactic
from src.interview.core.graph_needs_detector import GraphNeedsDetector
from src.interview.core.strategy_selector import StrategySelector
from src.interview.tactics.selector import SchemaDrivenTacticSelector
from src.interview.tactics.question_generator import QuestionGenerator
from src.interview.extraction.graph_extraction_orchestrator import GraphExtractionOrchestrator


logger = logging.getLogger(__name__)


class GraphDrivenOrchestrator:
    """
    Main orchestrator that implements the graph-driven interview pipeline.
    
    The orchestrator follows this flow:
    GraphState → GraphNeedsDetector → StrategySelector → TacticSelector → QuestionGenerator
    
    This replaces phase-driven logic with graph-state-driven question selection.
    """
    
    def __init__(self, extraction_orchestrator: GraphExtractionOrchestrator,
                 needs_detector: GraphNeedsDetector = None,
                 strategy_selector: StrategySelector = None,
                 tactic_selector: SchemaDrivenTacticSelector = None,
                 question_generator: QuestionGenerator = None):
        """Initialize the orchestrator with its components.

        Args:
            extraction_orchestrator: Required graph extraction orchestrator
            needs_detector: Optional graph needs detector (defaults to new instance)
            strategy_selector: Optional strategy selector (defaults to new instance)
            tactic_selector: Optional tactic selector (defaults to new instance)
            question_generator: Optional question generator (defaults to new instance)

        Raises:
            ValueError: If extraction_orchestrator is None
        """
        if not extraction_orchestrator:
            raise ValueError("extraction_orchestrator is required for graph-driven interviews")

        self.extraction_orchestrator = extraction_orchestrator
        self.needs_detector = needs_detector or GraphNeedsDetector()
        self.strategy_selector = strategy_selector or StrategySelector()
        self.tactic_selector = tactic_selector or SchemaDrivenTacticSelector()
        self.question_generator = question_generator or QuestionGenerator()

        logger.info("GraphDrivenOrchestrator initialized")
        logger.debug("Components: needs_detector=%s, strategy_selector=%s, tactic_selector=%s, extraction_orchestrator=%s",
                    type(self.needs_detector).__name__,
                    type(self.strategy_selector).__name__,
                    type(self.tactic_selector).__name__,
                    type(self.extraction_orchestrator).__name__)
    
    async def next_question(self, graph_state: GraphState, interview_state: InterviewState,
                     available_tactics: list) -> Optional[str]:
        """
        Generate the next question based on graph state and interview state.
        
        Args:
            graph_state: Current knowledge graph
            interview_state: Current interview state
            available_tactics: List of available tactics
            
        Returns:
            Generated question or None if generation fails
        """
        logger.info("Generating next question (turn %s)", interview_state.turn_number)
        logger.debug("Graph state: %s nodes, %s edges", 
                    graph_state.get_node_count(), graph_state.get_edge_count())
        
        try:
            # Step 1: Detect graph needs
            logger.debug("Step 1: Detecting productive graph needs")
            needs = self.needs_detector.detect_productive_needs(graph_state)
            
            if not needs:
                logger.warning("No needs detected, falling back to default behavior")
                return self._get_fallback_question(interview_state)
            
            # Step 2: Select strategy with dead-end fallback
            logger.debug("Step 2: Selecting strategy from %s needs with dead-end protection", len(needs))
            strategy = self.strategy_selector.select(needs, graph_state, interview_state)
            logger.info("Selected strategy: %s", strategy.value)
            
            # Step 3: Select tactic
            logger.debug("Step 3: Selecting tactic for strategy: %s", strategy.value)
            tactic = self.tactic_selector.select(strategy, interview_state, available_tactics)
            
            if not tactic:
                logger.warning("No valid tactic found for strategy: %s", strategy.value)
                return self._get_fallback_question(interview_state)
            
            logger.info("Selected tactic: %s", tactic.id)
            
            # Track depth activity for dead-end protection
            self._track_depth_activity(interview_state, tactic)
            
            # Step 4: Generate question using LLM or templates
            logger.debug("Step 4: Generating question from tactic: %s", tactic.id)
            question = await self.question_generator.generate_question(
                tactic, graph_state, interview_state
            )
            
            # Track token usage if LLM was used and returned usage data
            if self.question_generator.llm_client and hasattr(self.question_generator, '_last_response'):
                last_response = getattr(self.question_generator, '_last_response', None)
                if last_response and last_response.usage:
                    interview_state.add_token_usage(
                        prompt_tokens=last_response.usage.get('prompt_tokens', 0),
                        completion_tokens=last_response.usage.get('completion_tokens', 0),
                        total_tokens=last_response.usage.get('total_tokens', 0)
                    )
                    logger.debug(f"Tracked token usage: {last_response.usage}")
            
            if not question:
                logger.warning("Question generation failed for tactic: %s", tactic.id)
                return self._get_fallback_question(interview_state)
            
            # Log the complete decision chain
            self._log_decision_chain(needs, strategy, tactic, question, graph_state, interview_state)
            
            logger.info("Successfully generated question: %s", question)
            return question
            
        except Exception as e:
            logger.error("Error in orchestrator pipeline: %s", str(e), exc_info=True)
            return self._get_fallback_question(interview_state)
    
    def _log_decision_chain(self, needs: List, strategy: Any, tactic: Any, 
                           question: str, graph_state: Any, interview_state: Any) -> None:
        """Log the complete decision chain for debugging and analysis."""
        logger.debug("=== DECISION CHAIN ===")
        logger.debug("Detected needs: %s", [str(need) for need in needs])
        logger.debug("Selected strategy: %s", strategy.value if hasattr(strategy, 'value') else str(strategy))
        logger.debug("Selected tactic: %s", tactic.id)
        logger.debug("Generated question: %s", question)
        logger.debug("Graph state: %s nodes, %s edges", 
                    graph_state.get_node_count(), graph_state.get_edge_count())
        logger.debug("Interview state: turn %s", interview_state.turn_number)
        logger.debug("======================")
    

    
    def _get_fallback_question(self, interview_state: InterviewState) -> str:
        """Get a fallback question when orchestration fails."""
        logger.info("Using fallback question for turn %s", interview_state.turn_number)
        
        # Simple fallback questions based on turn number
        fallback_questions = [
            "Can you tell me more about that?",
            "What else comes to mind when you think about this?",
            "How did that experience affect you?",
            "What was most important about that?",
            "Can you describe that in more detail?"
        ]
        
        # Cycle through fallback questions
        question_index = interview_state.turn_number % len(fallback_questions)
        return fallback_questions[question_index]
    
    async def process_response(self, response_text: str, conversation_history: List[Dict[str, str]], 
                             graph_state: GraphState, interview_state: InterviewState) -> Optional[str]:
        """
        Process participant response and generate next question.
        
        This method integrates concept extraction with the existing orchestration pipeline.
        
        Args:
            response_text: Participant's response text
            conversation_history: Recent conversation turns
            graph_state: Current knowledge graph state (will be updated)
            interview_state: Current interview state
            
        Returns:
            Generated next question or None if processing fails
        """
        # extraction_orchestrator is now required, so this check is no longer needed
        # If we reach here, extraction_orchestrator is guaranteed to exist
        
        try:
            # Step 1: Extract concepts from response and update graph
            logger.debug("Step 1: Extracting concepts from response")
            extraction_delta = await self.extraction_orchestrator.process_participant_response(
                response_text=response_text,
                conversation_history=conversation_history,
                current_graph=graph_state,
                interview_state=interview_state
            )
            
            # Log extraction results
            if not extraction_delta.is_empty():
                logger.info(f"Extracted and applied: {extraction_delta.get_summary()}")
                logger.debug(f"Applied nodes: {len(extraction_delta.nodes_added)}, "
                           f"Applied edges: {len(extraction_delta.edges_added)}")
            else:
                logger.debug("No concepts extracted from response")
            
            # Step 2: Generate next question based on updated graph
            logger.debug("Step 2: Generating next question based on updated graph")
            
            # Get available tactics (load if not already loaded)
            from src.interview.tactics.loader import SchemaDrivenTacticLoader
            tactic_loader = SchemaDrivenTacticLoader()
            available_tactics = tactic_loader.load_tactics()
            
            # Generate next question using updated graph state
            next_question = await self.next_question(
                graph_state=graph_state,
                interview_state=interview_state,
                available_tactics=available_tactics
            )
            
            return next_question
            
        except Exception as e:
            logger.error(f"Error processing response with extraction: {e}", exc_info=True)
            # Fallback to regular orchestration
            return await self.next_question(graph_state, interview_state, [])
    
    def get_orchestrator_state(self) -> Dict:
        """Get current state of the orchestrator components."""
        return {
            "needs_detector": type(self.needs_detector).__name__,
            "strategy_selector": type(self.strategy_selector).__name__,
            "tactic_selector": type(self.tactic_selector).__name__,
            "components_initialized": True
        }
    
    def _track_depth_activity(self, interview_state: InterviewState, tactic: SchemaTactic) -> None:
        """Track when depth exploration tactics are used for dead-end protection."""
        depth_tactics = ["value_ladder", "emotional_probe", "causal_value_link"]
        
        if tactic.id in depth_tactics:
            interview_state.last_depth_turn = interview_state.turn_number
            logger.debug("Depth tactic detected: %s, updated last_depth_turn to %s", 
                        tactic.id, interview_state.turn_number)
    
    def validate_components(self) -> bool:
        """Validate that all orchestrator components are properly initialized."""
        try:
            assert self.needs_detector is not None, "Needs detector not initialized"
            assert self.strategy_selector is not None, "Strategy selector not initialized"
            assert self.tactic_selector is not None, "SchemaTactic selector not initialized"
            
            logger.debug("All orchestrator components validated successfully")
            return True
            
        except AssertionError as e:
            logger.error("Orchestrator component validation failed: %s", str(e))
            return False
    
    def __str__(self) -> str:
        """String representation of the orchestrator."""
        return (f"GraphDrivenOrchestrator(needs_detector={type(self.needs_detector).__name__}, "
                f"strategy_selector={type(self.strategy_selector).__name__}, "
                f"tactic_selector={type(self.tactic_selector).__name__})")