"""
Configurable Graph-Driven Orchestrator - Uses interview configuration from YAML.
Replaces hardcoded values with configuration-driven behavior.
"""

import logging
from typing import Optional, List, Dict, Any
from src.core.models import GraphState, InterviewState, Tactic
from src.interview.core.graph_needs_detector import GraphNeedsDetector
from src.interview.core.strategy_selector import StrategySelector
from src.interview.tactics.selector import SchemaDrivenTacticSelector
from src.interview.tactics.question_generator import QuestionGenerator
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
                 needs_detector: GraphNeedsDetector = None,
                 strategy_selector: StrategySelector = None,
                 tactic_selector: SchemaDrivenTacticSelector = None,
                 question_generator: QuestionGenerator = None):
        """Initialize the configurable orchestrator with configuration loader.

        Args:
            extraction_orchestrator: Required graph extraction orchestrator
            config_loader: Interview configuration loader
            needs_detector: Optional graph needs detector (uses config if not provided)
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
    
    def _create_configured_needs_detector(self) -> GraphNeedsDetector:
        """Create GraphNeedsDetector with configuration values."""
        return GraphNeedsDetector(
            min_nodes_for_seed_expansion=self.config.graph_needs.min_nodes_for_seed_expansion,
            isolation_threshold=self.config.graph_needs.isolation_threshold,
            depth_completion_threshold=self.config.graph_needs.depth_completion_threshold,
            target_depth=self.config.graph_needs.target_depth,
            dead_end_threshold=self.config.graph_needs.dead_end_threshold,
            dead_end_probe_count=self.config.graph_needs.dead_end_probe_count
        )
    
    def _create_configured_strategy_selector(self) -> StrategySelector:
        """Create StrategySelector with configuration values."""
        return StrategySelector(
            dead_end_threshold=self.config.graph_needs.dead_end_threshold
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
    
    def _create_configured_question_generator(self) -> QuestionGenerator:
        """Create QuestionGenerator with configuration values."""
        return QuestionGenerator(
            temperature=self.config.question_generation.temperature,
            max_tokens=self.config.question_generation.max_tokens,
            max_question_length=self.config.question_generation.max_question_length,
            context_weights=self.config.question_generation.context_weights
        )
    
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
            tactic = self.tactic_selector.select(strategy, interview_state, available_tactics)
            
            if not tactic:
                logger.warning("No valid tactic found, using fallback")
                return self._get_configured_fallback_question(interview_state)
            
            # Step 4: Generate question with configured parameters
            logger.debug("Step 4: Generating question with configuration")
            question = await self.question_generator.generate_question(
                graph_state, interview_state, tactic, available_tactics
            )
            
            if question:
                logger.info("Generated question successfully with configuration")
                return question
            else:
                logger.warning("Question generation failed, using fallback")
                return self._get_configured_fallback_question(interview_state)
                
        except Exception as e:
            logger.error(f"Interview orchestration failed with configuration: {e}", exc_info=True)
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