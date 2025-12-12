"""
Interview orchestration.
Ties all components together in the interview loop.
"""

import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

from core.graph import Graph
from core.schema import Schema
from core.history import History, Turn
from core.state import GraphState, CoverageState, Momentum, NodeFocusTracker, EdgeFocusTracker
from decision.strategy import StrategySelector
from decision.extraction import Extractor, ExtractionResult
from generation.generator import QuestionGenerator, GeneratedQuestion
from utils.llm_manager import LLMManager
from utils.logger import InterviewLogger
from utils.concept_parser import ConceptParser, ParsedConcept, load_concept

logger = logging.getLogger(__name__)


class InterviewConfig(BaseModel):
    """Configuration for interview session."""
    schema_path: str = Field(description="Path to methodology schema YAML")
    logic_path: str = Field(description="Path to interview_logic.yaml")
    llm_config_path: str = Field(description="Path to llm_config.yaml")
    concept_text: str = Field(description="Stimulus concept text")
    concept_name: str = Field(default="Unnamed Concept", description="Concept name")
    element_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Manual element configuration for coverage tracking"
    )
    max_turns: int = Field(default=20, description="Maximum interview turns")
    min_coverage_before_expansion: bool = Field(
        default=True,
        description="Require coverage before free expansion"
    )


class InterviewState(BaseModel):
    """Current state of the interview."""
    turn_count: int = 0
    is_complete: bool = False
    completion_reason: Optional[str] = None
    last_question: Optional[str] = None
    last_strategy: Optional[str] = None


class InterviewController:
    """
    Main orchestrator for interview sessions.
    Manages the interview loop from opening to closing.
    """
    
    def __init__(
        self,
        graph: Graph,
        schema: Schema,
        coverage_state: CoverageState,
        history: History,
        strategy_selector: StrategySelector,
        extractor: Extractor,
        generator: QuestionGenerator,
        config: InterviewConfig,
        logger: Optional[InterviewLogger] = None
    ):
        self.graph = graph
        self.schema = schema
        self.coverage_state = coverage_state
        self.history = history
        self.strategy_selector = strategy_selector
        self.extractor = extractor
        self.generator = generator
        self.config = config
        self.state = InterviewState()
        self._concept_text = config.concept_text

        # Node focus tracking to prevent stuck loops
        self.node_focus_tracker = NodeFocusTracker()

        # Edge focus tracking to prevent stuck loops on invalid edges
        self.edge_focus_tracker = EdgeFocusTracker()

        # Momentum tracking for fatigue detection (persistent across turns)
        self._momentum_tracker = Momentum.default()

        # Tracking for extended logging
        self._coverage_evolution: list = []
        self._graph_growth: list = []

        # Set up logging
        self.logger = logger or InterviewLogger()
        self.logger.session_start(config.concept_name, schema.name)

        # Token usage tracking
        self.token_usage: Dict[str, Dict[str, Union[int, float]]] = {}

    def enable_arbitration(self, config: Optional[Dict] = None) -> None:
        """
        Enable utility-based strategy arbitration.

        Args:
            config: Optional configuration dict for scorers. If None, uses defaults.
                   Format: {"scorers": {"redundancy": {"weight": 1.0}, ...}}
        """
        from decision.arbitration import ArbitrationEngine

        if config:
            engine = ArbitrationEngine.from_config(config)
        else:
            engine = ArbitrationEngine.create_default()

        self.strategy_selector.set_arbitration_engine(engine)
        self.logger.info("[Controller] Arbitration mode enabled")

    def _record_token_usage(self, response) -> None:
        """
        Record token usage from an LLM response.

        Args:
            response: LLMResponse object from an LLM call
        """
        from utils.llm_manager import LLMResponse

        # Type check for safety
        if not isinstance(response, LLMResponse):
            return

        model_key = f"{response.provider}:{response.model}"

        if model_key not in self.token_usage:
            self.token_usage[model_key] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_cost_usd': 0.0
            }

        self.token_usage[model_key]['input_tokens'] += response.input_tokens
        self.token_usage[model_key]['output_tokens'] += response.output_tokens

        if response.cost_usd:
            self.token_usage[model_key]['total_cost_usd'] += response.cost_usd

    @classmethod
    def initialize(
        cls,
        concept_text: str,
        schema_path: str,
        logic_path: str,
        llm_config_path: str,
        element_config: Optional[Dict[str, Any]] = None,
        concept_name: str = "Unnamed Concept",
        max_turns: int = 20,
        session_id: Optional[str] = None
    ) -> "InterviewController":
        """
        Set up a fresh interview session.
        
        Args:
            concept_text: The stimulus concept
            schema_path: Path to methodology schema
            logic_path: Path to interview_logic.yaml
            llm_config_path: Path to llm_config.yaml
            element_config: Optional manual element definitions
            concept_name: Name of the concept
            max_turns: Maximum interview turns
            session_id: Optional session ID for logging
            
        Returns:
            Configured InterviewController
        """
        # Set up logger first
        logger = InterviewLogger(session_id)
        logger.info(f"Initializing interview session")
        
        # Load configurations
        logger.info(f"Loading schema from {schema_path}")
        schema = Schema.load(schema_path)
        
        logger.info(f"Loading interview logic from {logic_path}")
        strategy_selector = StrategySelector.load(logic_path)
        
        logger.info(f"Loading LLM config from {llm_config_path}")
        llm_manager = LLMManager.from_config_file(llm_config_path)
        llm_manager.log_health_check()

        # Initialize state objects
        graph = Graph()
        history = History()
        coverage_state = CoverageState.initialize(concept_text, element_config)

        # Log concept and reference elements for debugging
        logger.info("=" * 60)
        logger.info("CONCEPT TEXT:")
        logger.info("-" * 40)
        for line in concept_text.strip().split('\n'):
            logger.info(f"  {line}")
        logger.info("-" * 40)

        logger.info(f"Coverage tracking {len(coverage_state.reference_elements)} elements")
        if coverage_state.reference_elements:
            logger.info("REFERENCE ELEMENTS FOR COVERAGE:")
            for elem_id, elem in coverage_state.reference_elements.items():
                content_preview = elem.content[:100] + "..." if len(elem.content) > 100 else elem.content
                logger.info(f"  - {elem_id}: {content_preview}")
        logger.info("=" * 60)
        
        # Create components
        extractor = Extractor(schema, coverage_state, llm_manager)
        generator = QuestionGenerator(llm_manager, strategy_selector)
        
        # Create config
        config = InterviewConfig(
            schema_path=schema_path,
            logic_path=logic_path,
            llm_config_path=llm_config_path,
            concept_text=concept_text,
            concept_name=concept_name,
            element_config=element_config,
            max_turns=max_turns
        )
        
        return cls(
            graph=graph,
            schema=schema,
            coverage_state=coverage_state,
            history=history,
            strategy_selector=strategy_selector,
            extractor=extractor,
            generator=generator,
            config=config,
            logger=logger
        )
    
    @classmethod
    def from_concept_file(
        cls,
        concept_path: str,
        schema_path: str,
        logic_path: str,
        llm_config_path: str,
        max_turns: int = 20,
        session_id: Optional[str] = None
    ) -> "InterviewController":
        """
        Initialize from a concept file with automatic parsing.
        
        Args:
            concept_path: Path to concept markdown file
            schema_path: Path to methodology schema
            logic_path: Path to interview_logic.yaml
            llm_config_path: Path to llm_config.yaml
            max_turns: Maximum interview turns
            session_id: Optional session ID for logging
            
        Returns:
            Configured InterviewController
        """
        # Load LLM for concept parsing
        llm_manager = LLMManager.from_config_file(llm_config_path)
        
        # Parse concept
        parser = ConceptParser(llm_manager)
        concept = parser.parse_file(concept_path)
        
        # Get element config from parsed concept
        element_config = concept.get_element_config()
        
        return cls.initialize(
            concept_text=concept.description,
            schema_path=schema_path,
            logic_path=logic_path,
            llm_config_path=llm_config_path,
            element_config=element_config,
            concept_name=concept.name,
            max_turns=max_turns,
            session_id=session_id
        )
    
    def generate_opening(self) -> str:
        """
        Generate the opening question that introduces the concept.
        
        Returns:
            Opening question text
        """
        self.logger.turn_start(0)
        
        generated = self.generator.generate_opening(
            self._concept_text,
            self.history
        )
        if generated.llm_response:
            self._record_token_usage(generated.llm_response)

        self.state.last_question = generated.question
        self.state.last_strategy = "opening"

        self.logger.question_generated(generated.question, "opening")
        
        return generated.question
    
    def process_response(self, response_text: str) -> str:
        """
        Process a respondent's response and generate the next question.
        
        This is the main interview loop step:
        1. Check extractability
        2. Extract nodes/edges (or clarify)
        3. Update graph and state
        4. Select strategy
        5. Generate next question
        6. Record turn
        
        Args:
            response_text: Respondent's answer
            
        Returns:
            Next question to ask
        """
        self.logger.turn_start(self.state.turn_count + 1)
        self.logger.response_received(response_text)
        
        # Check if interview should end
        if self.should_close():
            self.state.is_complete = True
            self.logger.info(f"Interview closing: {self.state.completion_reason}")
            closing = self._generate_closing()
            self._log_session_end()
            return closing
        
        # 1. Check extractability
        extractable, reason, llm_resp = self.extractor.assess_extractability(
            response_text,
            self.history
        )
        if llm_resp:
            self._record_token_usage(llm_resp)
        
        if not extractable:
            self.logger.extraction_result(0, 0, False)
            
            # Generate clarification question
            generated = self.generator.generate_clarification(
                reason or "Response unclear",
                response_text,
                self.history
            )
            if generated.llm_response:
                self._record_token_usage(generated.llm_response)

            # Record turn (no extraction)
            self._record_turn(
                question=generated.question,
                response=response_text,
                extraction=None,
                strategy_id="clarification",
                focus_node_id=None
            )
            
            self.logger.question_generated(generated.question, "clarification")
            
            return generated.question
        
        # 2. Extract and assess momentum
        extraction = self.extractor.extract(response_text, self.graph, self.history)
        if extraction.llm_response:
            self._record_token_usage(extraction.llm_response)

        momentum = self.extractor.assess_momentum(response_text, self.history)
        if momentum.llm_response:
            self._record_token_usage(momentum.llm_response)

        # Update momentum tracker for fatigue detection
        self._momentum_tracker.level = momentum.level
        self._momentum_tracker.indicators = momentum.indicators
        self._momentum_tracker.record(self.state.turn_count + 1)

        self.logger.extraction_result(
            len(extraction.nodes),
            len(extraction.edges),
            extraction.is_extractable
        )
        self.logger.momentum_assessed(momentum.level, momentum.indicators)

        # Extended logging: extraction details
        self.logger.extraction_details(
            nodes=[n.label for n in extraction.nodes],
            edges=[(e.source_id, e.target_id, e.relation_type) for e in extraction.edges],
            reactions=extraction.element_reactions,
            element_mappings=extraction.node_element_mappings
        )

        if self._momentum_tracker.is_fatigued():
            logger.warning(f"[Fatigue] Sustained low engagement detected ({self._momentum_tracker.consecutive_low_count} consecutive)")
        
        # 3. Update graph
        for node in extraction.nodes:
            self.graph.add_node(node)
        for edge in extraction.edges:
            self.graph.add_edge(edge)

        # Track coverage before update
        gaps_before = len(self.coverage_state.get_gaps())
        mappings_before = dict(self.coverage_state.element_node_mappings)

        # Update coverage state
        self.coverage_state.update(self.graph, extraction.node_element_mappings)

        # Record reactions for coverage
        for element_id, reaction in extraction.element_reactions.items():
            self.coverage_state.record_reaction(element_id, reaction)

        # Log coverage changes
        gaps_after = len(self.coverage_state.get_gaps())
        if extraction.node_element_mappings:
            for node_id, element_id in extraction.node_element_mappings.items():
                node = self.graph.get_node(node_id)
                node_label = node.label if node else node_id
                logger.info(f"[Coverage Update] element '{element_id}' covered by node '{node_label}'")

        if gaps_before != gaps_after:
            remaining_gaps = [g.element_id for g in self.coverage_state.get_gaps()]
            logger.info(f"[Coverage] Gaps changed: {gaps_before} -> {gaps_after}, remaining: {remaining_gaps}")

        self.logger.coverage_update(
            gaps_after,
            self.coverage_state.is_satisfied()
        )
        
        # Compute graph state
        graph_state = GraphState.compute(
            self.graph, 
            self.schema, 
            self.history.turns
        )
        
        self.logger.graph_state(
            len(graph_state.isolated_nodes),
            len(graph_state.ambiguous_nodes),
            len(graph_state.terminal_nodes)
        )

        # Track metrics for extended logging
        self._coverage_evolution.append(gaps_after)
        self._graph_growth.append((len(self.graph.nodes), len(self.graph.edges)))

        # Extended logging: graph evolution
        isolation_ratio = self.graph.compute_isolation_ratio() if hasattr(self.graph, 'compute_isolation_ratio') else (
            len(graph_state.isolated_nodes) / len(self.graph.nodes) if self.graph.nodes else 0.0
        )
        self.logger.graph_evolution(
            turn=self.state.turn_count + 1,
            nodes_total=len(self.graph.nodes),
            edges_total=len(self.graph.edges),
            nodes_added=len(extraction.nodes),
            edges_added=len(extraction.edges),
            isolation_ratio=isolation_ratio,
            coverage_gaps=gaps_after
        )

        # 4. Select strategy (with node and edge focus tracking)
        strategy, focus = self.strategy_selector.select(
            self.graph,
            graph_state,
            self.coverage_state,
            self._momentum_tracker,
            self.node_focus_tracker,
            self.edge_focus_tracker,
            self.history  # Pass history for arbitration
        )

        # Record node focus to prevent stuck loops
        if focus.node:
            self.node_focus_tracker.record_focus(focus.node.id)
            logger.debug(f"[NodeFocus] Recorded focus on '{focus.node.label}' (count: {self.node_focus_tracker.get_focus_count(focus.node.id)})")

        # Record edge focus for resolve_schema_tension
        if strategy.id == "resolve_schema_tension" and focus.node_pair:
            # Find the edge between this pair
            n1, n2 = focus.node_pair
            edge = self.graph.get_edge_between(n1.id, n2.id)
            if not edge:
                edge = self.graph.get_edge_between(n2.id, n1.id)
            if edge:
                self.edge_focus_tracker.record_focus(edge.id)
                logger.debug(f"[EdgeFocus] Recorded focus on edge '{edge.id}' (count: {self.edge_focus_tracker.get_focus_count(edge.id)})")

        self.logger.strategy_selected(strategy.id, focus.describe())

        # Extended logging: strategy reasoning
        self.logger.strategy_reasoning(
            strategy_id=strategy.id,
            intent=strategy.intent,
            focus_description=focus.describe()
        )

        # 5. Generate question
        generated = self.generator.generate(
            strategy,
            focus,
            self.graph,
            self.history,
            momentum
        )
        if generated.llm_response:
            self._record_token_usage(generated.llm_response)
        # Track plausibility check if it was performed
        if "plausibility_llm_response" in generated.metadata:
            self._record_token_usage(generated.metadata["plausibility_llm_response"])

        # Handle no plausible connections
        if not generated.question and generated.metadata.get("no_plausible_connections"):
            node_id = generated.metadata.get("isolated_node_id")
            if node_id:
                # Force exhaustion of this node for connect_isolate
                self.node_focus_tracker.exhausted_nodes.append(node_id)
            # Re-select strategy (will skip exhausted node)
            strategy, focus = self.strategy_selector.select(
                self.graph, graph_state, self.coverage_state,
                self._momentum_tracker, self.node_focus_tracker, self.edge_focus_tracker,
                self.history
            )
            generated = self.generator.generate(strategy, focus, self.graph, self.history, momentum)
            if generated.llm_response:
                self._record_token_usage(generated.llm_response)
            # Track plausibility check if it was performed in the regeneration
            if "plausibility_llm_response" in generated.metadata:
                self._record_token_usage(generated.metadata["plausibility_llm_response"])

        # 6. Record turn
        focus_node_id = focus.node.id if focus.node else None
        self._record_turn(
            question=generated.question,
            response=response_text,
            extraction=extraction,
            strategy_id=strategy.id,
            focus_node_id=focus_node_id,
            momentum=momentum
        )
        
        self.state.last_question = generated.question
        self.state.last_strategy = strategy.id

        # Only log non-empty questions
        if generated.question:
            self.logger.question_generated(generated.question, strategy.id)
        else:
            logger.warning(f"[Question] Empty question generated for strategy '{strategy.id}' - this should not happen after empty question handling")

        # Log turn summary and graph growth
        logger.info(
            f"[Turn {self.state.turn_count}] Response: {len(response_text)} chars -> "
            f"+{len(extraction.nodes)} nodes, +{len(extraction.edges)} edges | "
            f"Strategy: {strategy.id} | Coverage: {gaps_after} gaps"
        )
        logger.info(
            f"[Graph Growth] Turn {self.state.turn_count}: "
            f"nodes={len(self.graph.nodes)}, edges={len(self.graph.edges)}"
        )

        return generated.question
    
    def should_close(self) -> bool:
        """
        Determine if the interview should end.

        Conditions:
        - Max turns reached
        - Fatigue detected with adequate coverage
        - Coverage complete + expansion exhausted + low momentum
        """
        # Max turns
        if self.state.turn_count >= self.config.max_turns:
            self.state.completion_reason = "max_turns_reached"
            return True

        # Fatigue-based early close (with minimum coverage)
        if self.state.turn_count >= 5 and self._momentum_tracker.is_fatigued():
            gaps = len(self.coverage_state.get_non_exhausted_gaps())
            total_elements = len(self.coverage_state.reference_elements)
            if total_elements > 0:
                coverage_ratio = 1 - (gaps / total_elements)
                if coverage_ratio >= 0.6:  # At least 60% coverage
                    self.state.completion_reason = "fatigue_detected"
                    logger.info(f"[Close] Fatigue detected with {coverage_ratio:.0%} coverage - ending early")
                    return True

        # Early exit conditions (optional)
        if self.state.turn_count >= 5:  # Minimum turns before considering early exit
            graph_state = GraphState.compute(
                self.graph,
                self.schema,
                self.history.turns
            )

            # All coverage satisfied
            coverage_complete = self.coverage_state.is_satisfied()

            # No structural issues
            no_issues = not graph_state.has_structural_issues()

            # Nothing left to explore
            nothing_to_explore = (
                len(graph_state.unexplored_nodes) == 0 and
                len(graph_state.terminal_nodes) > 0  # Have reached some endpoints
            )

            if coverage_complete and no_issues and nothing_to_explore:
                self.state.completion_reason = "exploration_complete"
                return True

        return False
    
    def _generate_closing(self) -> str:
        """Generate closing statement."""
        return "Thank you for sharing your thoughts. We've covered a lot of ground. Is there anything else you'd like to add about this concept?"
    
    def _record_turn(
        self,
        question: str,
        response: str,
        extraction: Optional[ExtractionResult],
        strategy_id: str,
        focus_node_id: Optional[str],
        momentum: Optional[Momentum] = None
    ) -> None:
        """Record a turn in history."""
        self.state.turn_count += 1
        
        turn = Turn(
            turn_number=self.state.turn_count,
            question=question,
            response=response,
            extracted_nodes=[n.id for n in extraction.nodes] if extraction else [],
            extracted_edges=[
                (e.source_id, e.target_id) for e in extraction.edges
            ] if extraction else [],
            strategy_used=strategy_id,
            timestamp=datetime.now(),
            metadata={
                "focus_node_id": focus_node_id,
                "momentum": momentum.level if momentum else None,
                "momentum_indicators": momentum.indicators if momentum else []
            }
        )
        
        self.history.add_turn(turn)
    
    def _log_session_end(self) -> None:
        """Log session end with extended summary."""
        summary = self.get_interview_summary()
        self.logger.session_end(summary)

        # Extended session summary
        graph_state = GraphState.compute(
            self.graph,
            self.schema,
            self.history.turns
        )
        isolation_ratio = (
            len(graph_state.isolated_nodes) / len(self.graph.nodes)
            if self.graph.nodes else 0.0
        )

        self.logger.session_summary_extended(
            total_turns=self.state.turn_count,
            strategy_counts=self.history.get_strategy_counts(),
            coverage_evolution=self._coverage_evolution,
            graph_growth=self._graph_growth,
            final_isolation_ratio=isolation_ratio,
            completion_reason=self.state.completion_reason or "unknown"
        )
    
    def get_interview_summary(self) -> Dict[str, Any]:
        """
        Get summary of the interview session.
        
        Returns:
            Dictionary with interview metrics and state
        """
        graph_state = GraphState.compute(
            self.graph, 
            self.schema, 
            self.history.turns
        )
        
        return {
            "turns": self.state.turn_count,
            "is_complete": self.state.is_complete,
            "completion_reason": self.state.completion_reason,
            "graph": {
                "nodes": len(self.graph.nodes),
                "edges": len(self.graph.edges),
                "isolated_nodes": len(graph_state.isolated_nodes),
                "terminal_nodes": len(graph_state.terminal_nodes),
            },
            "coverage": {
                "satisfied": self.coverage_state.is_satisfied(),
                "gaps": len(self.coverage_state.get_gaps()),
                "elements_addressed": len([
                    e for e in self.coverage_state.reference_elements
                    if self.coverage_state.element_node_mappings.get(e)
                ])
            },
            "strategies_used": self.history.get_strategy_counts()
        }
    
    def get_final_graph(self) -> Graph:
        """Export the final knowledge graph."""
        return self.graph
    
    def get_transcript(self) -> str:
        """Get full interview transcript."""
        return self.history.format_full_transcript()
    
    def export_session(self) -> Dict[str, Any]:
        """
        Export complete session data for analysis.
        
        Returns:
            Dictionary with all session data
        """
        return {
            "config": self.config.model_dump(),
            "summary": self.get_interview_summary(),
            "graph": self.graph.to_dict(),
            "history": self.history.to_dict(),
            "coverage": {
                "elements": {
                    elem_id: self.coverage_state.get_element_status(elem_id)
                    for elem_id in self.coverage_state.reference_elements
                },
                "gaps": [gap.model_dump() for gap in self.coverage_state.get_gaps()]
            },
            "schema_validation": self.schema.validate_graph(self.graph),
            "token_usage": self.token_usage,
            "total_cost_usd": sum(
                model_data['total_cost_usd']
                for model_data in self.token_usage.values()
            )
        }
    
    def close(self) -> Dict[str, Any]:
        """
        Explicitly close the interview and return summary.
        
        Returns:
            Interview summary
        """
        self.state.is_complete = True
        if not self.state.completion_reason:
            self.state.completion_reason = "manual_close"
        
        self._log_session_end()
        return self.get_interview_summary()
