"""
Configurable Graph-Driven Orchestrator - Uses interview configuration from YAML.
Replaces hardcoded values with configuration-driven behavior.
"""

import logging
from typing import Optional, List, Dict, Any
from src.core.models import GraphState, InterviewState, SchemaTactic
from src.interview.core.configurable_graph_needs_detector import ConfigurableGraphNeedsDetector
from src.interview.core.strategy_selector import StrategySelector
from src.interview.tactics.selector import SchemaDrivenTacticSelector
from src.interview.tactics.configurable_question_generator import ConfigurableQuestionGenerator
from src.interview.extraction.graph_extraction_orchestrator import GraphExtractionOrchestrator
from src.config.interview_config_loader import InterviewConfig, InterviewConfigLoader

logger = logging.getLogger(__name__)


class ConfigurableGraphDrivenOrchestrator:
    """
    Configurable orchestrator that uses interview settings from YAML configuration.
    
    This orchestrator replaces hardcoded values with configuration-driven behavior,
    making the system truly configurable without code changes.
    """
    
    def __init__(self, 
                 extraction_orchestrator: GraphExtractionOrchestrator,
                 config_loader: InterviewConfigLoader,
                 needs_detector: ConfigurableGraphNeedsDetector = None,
                 strategy_selector: StrategySelector = None,
                 tactic_selector: SchemaDrivenTacticSelector = None,
                 question_generator: ConfigurableQuestionGenerator = None):
        """Initialize the configurable orchestrator with configuration loader.

        Args:
            extraction_orchestrator: Required graph extraction orchestrator
            config_loader: Interview configuration loader
            needs_detector: Optional configurable graph needs detector (uses config if not provided)
            strategy_selector: Optional strategy selector (uses config if not provided)
            tactic_selector: Optional tactic selector (uses config if not provided)
            question_generator: Optional question generator (uses config if not provided)

        Raises:
            ValueError: If extraction_orchestrator or config_loader is None
        """
        if not extraction_orchestrator:
            raise ValueError("extraction_orchestrator is required for configurable interviews")
        if not config_loader:
            raise ValueError("config_loader is required for configurable interviews")

        self.extraction_orchestrator = extraction_orchestrator
        self.config_loader = config_loader
        
        # Load configuration once during initialization
        self.config = config_loader.load_config()
        
        # Initialize components with configuration
        self.needs_detector = needs_detector or self._create_configured_needs_detector()
        self.strategy_selector = strategy_selector or self._create_configured_strategy_selector()
        self.tactic_selector = tactic_selector or self._create_configured_tactic_selector()
        self.question_generator = question_generator or self._create_configured_question_generator()

        logger.info("ConfigurableGraphDrivenOrchestrator initialized with configuration")
        logger.info("Configuration loaded: interview_flow=%s, graph_needs=%s, llm=%s",
                   type(self.config.interview_flow).__name__,
                   type(self.config.graph_needs).__name__,
                   type(self.config.llm).__name__)
    
    def _create_configured_needs_detector(self) -> ConfigurableGraphNeedsDetector:
        """Create ConfigurableGraphNeedsDetector with configuration values."""
        return ConfigurableGraphNeedsDetector(config=self.config)
    
    def _create_configured_strategy_selector(self) -> StrategySelector:
        """Create StrategySelector with configuration values."""
        return StrategySelector(
            dead_end_threshold=self.config.graph_needs.dead_end_threshold,
            config=self.config,
            needs_detector=self.needs_detector
        )
    
    def _create_configured_tactic_selector(self) -> SchemaDrivenTacticSelector:
        """Create TacticSelector with configuration values."""
        return SchemaDrivenTacticSelector(
            usage_penalty_weight=self.config.tactic_selection.usage_penalty_weight,
            recency_penalty_weight=self.config.tactic_selection.recency_penalty_weight,
            recency_penalty_cap=self.config.tactic_selection.recency_penalty_cap,
            recent_tactics_count=self.config.tactic_selection.recent_tactics_count,
            recent_questions_count=self.config.tactic_selection.recent_questions_count
        )
    
    def _create_configured_question_generator(self) -> ConfigurableQuestionGenerator:
        """Create ConfigurableQuestionGenerator with configuration values."""
        return ConfigurableQuestionGenerator(config=self.config)
    
    async def next_question(self, graph_state: GraphState, interview_state: InterviewState,
                     available_tactics: list) -> Optional[str]:
        """Generate the next question using configuration-driven logic.
        
        Args:
            graph_state: Current knowledge graph
            interview_state: Current interview state
            available_tactics: List of available tactics
            
        Returns:
            Generated question or None if generation fails
        """
        logger.info("Generating next question with configuration (turn %s)", interview_state.turn_number)
        logger.debug("Graph state: %s nodes, %s edges", 
                    graph_state.get_node_count(), graph_state.get_edge_count())
        
        try:
            # Check turn limits from configuration
            if interview_state.turn_number >= self.config.interview_flow.max_turns:
                logger.info("Maximum turns reached (%s), ending interview", self.config.interview_flow.max_turns)
                return None
            
            if interview_state.turn_number < self.config.interview_flow.min_turns:
                logger.debug("Minimum turns not reached (%s), continuing interview", self.config.interview_flow.min_turns)
            
            # Step 1: Detect graph needs with configured parameters
            logger.debug("Step 1: Detecting graph needs with configuration")
            needs = self.needs_detector.detect_productive_needs(graph_state)
            
            if not needs:
                logger.warning("No needs detected, using fallback")
                return self._get_configured_fallback_question(interview_state)
            
            # Step 2: Select strategy with configured parameters
            logger.debug("Step 2: Selecting strategy with configuration")
            strategy = self.strategy_selector.select(needs, graph_state, interview_state)
            logger.info("Selected strategy: %s (configured)", strategy.value)
            
            # Step 3: Select tactic with configured parameters
            logger.debug("Step 3: Selecting tactic with configuration")
            from src.interview.tactics.loader import SchemaDrivenTacticLoader
            tactic_loader = SchemaDrivenTacticLoader()
            available_tactics = tactic_loader.load_tactics()
            tactic = self.tactic_selector.select(strategy, interview_state, available_tactics)
            
            if not tactic:
                logger.warning("No valid tactic found, using fallback")
                return self._get_configured_fallback_question(interview_state)
            
            # Step 4: Generate question with configured parameters
            logger.debug("Step 4: Generating question with configuration")
            question, tokens_used = await self.question_generator.generate_question(
                tactic=tactic,
                graph_state=graph_state,
                interview_state=interview_state
            )
            
            # Track token usage from question generation
            interview_state.add_token_usage(
                prompt_tokens=0,  # Question generation doesn't separate prompt/completion
                completion_tokens=0,
                total_tokens=tokens_used
            )
            logger.debug(f"Tracked {tokens_used} tokens from question generation")
            
            if question:
                logger.info("Generated question successfully with configuration")
                return question
            else:
                logger.warning("Question generation failed, using fallback")
                return self._get_configured_fallback_question(interview_state)
                
        except Exception as e:
            logger.error(f"Interview orchestration failed with configuration: {e}", exc_info=True)
            return self._get_configured_fallback_question(interview_state)
    
    async def process_response(self, response_text: str, 
                             conversation_history: List[Dict[str, str]],
                             graph_state: GraphState,
                             interview_state: InterviewState) -> Optional[str]:
        """
        Process participant response and generate the next question.
        
        Args:
            response_text: Participant's response text
            conversation_history: Recent conversation turns
            graph_state: Current knowledge graph state
            interview_state: Current interview state
            
        Returns:
            Generated next question or None if generation fails
        """
        logger.info(f"Processing response for turn {interview_state.turn_number}")
        
        try:
            # Process response through extraction if available
            if self.extraction_orchestrator:
                logger.debug("Processing response through concept extraction")
                delta, extraction_result = await self.extraction_orchestrator.process_participant_response(
                    response_text=response_text,
                    conversation_history=conversation_history,
                    current_graph=graph_state,
                    interview_state=interview_state
                )
                
                # Track token usage from extraction
                if extraction_result and extraction_result.metadata:
                    interview_state.add_token_usage(
                        prompt_tokens=0,  # Extraction doesn't separate prompt/completion
                        completion_tokens=0,
                        total_tokens=extraction_result.metadata.tokens_used
                    )
                    logger.debug(f"Tracked {extraction_result.metadata.tokens_used} tokens from extraction")
                
                if not delta.is_empty():
                    logger.info(f"Extracted {len(delta.nodes_added)} new concepts from response")
                else:
                    logger.debug("No new concepts extracted from response")
            else:
                logger.debug("No extraction orchestrator available, skipping concept extraction")
            
            # Generate next question using updated graph state
            logger.debug("Generating next question after response processing")
            from src.interview.tactics.loader import SchemaDrivenTacticLoader
            tactic_loader = SchemaDrivenTacticLoader()
            available_tactics = tactic_loader.load_tactics()
            
            next_question = await self.next_question(
                graph_state=graph_state,
                interview_state=interview_state,
                available_tactics=available_tactics
            )
            
            if next_question:
                logger.info(f"Generated next question: {next_question}")
                return next_question
            else:
                logger.warning("Failed to generate next question, using fallback")
                return self._get_configured_fallback_question(interview_state)
                
        except Exception as e:
            logger.error(f"Response processing failed: {e}", exc_info=True)
            return self._get_configured_fallback_question(interview_state)
    
    def _get_configured_fallback_question(self, interview_state: InterviewState) -> str:
        """Get fallback question based on configuration."""
        if self.config.interview_flow.enable_fallback and self.config.interview_flow.fallback_questions:
            # Use configured fallback questions
            import random
            return random.choice(self.config.interview_flow.fallback_questions)
        else:
            # Use hardcoded fallback (for safety)
            return "Can you tell me more about that?"
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current interview configuration."""
        return {
            "interview_flow": {
                "max_turns": self.config.interview_flow.max_turns,
                "min_turns": self.config.interview_flow.min_turns,
                "enable_fallback": self.config.interview_flow.enable_fallback,
                "fallback_questions_count": len(self.config.interview_flow.fallback_questions)
            },
            "graph_needs": {
                "target_depth": self.config.graph_needs.target_depth,
                "dead_end_threshold": self.config.graph_needs.dead_end_threshold,
                "strategy_weights": self.config.graph_needs.strategy_weights
            },
            "extraction": {
                "confidence_threshold": self.config.extraction.confidence_threshold,
                "validation_stages": self.config.extraction.validation_stages
            },
            "llm": {
                "default_provider": self.config.llm.default_provider,
                "extraction_temperature": self.config.llm.extraction_temperature,
                "question_temperature": self.config.llm.question_temperature
            }
        }
    
    def get_orchestrator_state(self) -> Dict:
        """Get current state of the orchestrator components."""
        return {
            "needs_detector": type(self.needs_detector).__name__,
            "strategy_selector": type(self.strategy_selector).__name__,
            "tactic_selector": type(self.tactic_selector).__name__,
            "components_initialized": True
        }
    
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
        return (f"ConfigurableGraphDrivenOrchestrator(needs_detector={type(self.needs_detector).__name__}, "
                f"strategy_selector={type(self.strategy_selector).__name__}, "
                f"tactic_selector={type(self.tactic_selector).__name__})")