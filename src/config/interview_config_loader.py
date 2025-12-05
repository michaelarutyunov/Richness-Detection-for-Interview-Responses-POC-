"""
Interview Configuration Loader - Loads interview settings from YAML.
Provides clean separation between configuration and codebase.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class InterviewFlowConfig:
    """Interview flow control configuration."""
    max_turns: int
    min_turns: int
    enable_fallback: bool
    fallback_questions: List[str]


@dataclass
class GraphNeedsConfig:
    """Graph needs detection configuration."""
    min_nodes_for_seed_expansion: int
    isolation_threshold: float
    depth_completion_threshold: float
    target_depth: int
    dead_end_threshold: float
    dead_end_probe_count: int
    strategy_weights: Dict[str, float]


@dataclass
class ExtractionConfig:
    """Extraction and validation configuration."""
    confidence_threshold: float
    validation_stages: int
    max_retries: int
    max_history_turns: int
    existing_nodes_limit: int


@dataclass
class TacticSelectionConfig:
    """Tactic and question selection configuration."""
    usage_penalty_weight: float
    recency_penalty_weight: float
    recency_penalty_cap: float
    recent_tactics_count: int
    recent_questions_count: int


@dataclass
class QuestionGenerationConfig:
    """Question generation configuration."""
    temperature: float
    max_tokens: int
    max_question_length: int
    context_weights: Dict[str, float]


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    level: str
    log_decisions: bool
    log_graph_state: bool
    log_tactic_selection: bool


@dataclass
class LLMConfig:
    """LLM provider and generation configuration."""
    default_provider: str
    models: Dict[str, str]
    extraction_temperature: float
    question_temperature: float
    max_tokens: int
    request_timeout: int
    max_retries: int
    retry_delay: int
    requests_per_minute: int
    tokens_per_minute: int


@dataclass
class PathsConfig:
    """File paths configuration."""
    schema_file: str
    extraction_prompts_file: str
    strategy_tactic_map_file: str
    log_directory: str


@dataclass
class UIConfig:
    """UI and server configuration."""
    server_name: str
    server_port: int
    share: bool
    show_error: bool


@dataclass
class AdvancedConfig:
    """Advanced features configuration (for future use)."""
    enable_focus_stack: bool
    enable_dead_end_detection: bool
    enable_coreference_resolution: bool
    enable_graph_visualization: bool


@dataclass
class InterviewConfig:
    """Complete interview configuration."""
    interview_flow: InterviewFlowConfig
    graph_needs: GraphNeedsConfig
    extraction: ExtractionConfig
    tactic_selection: TacticSelectionConfig
    question_generation: QuestionGenerationConfig
    logging: LoggingConfig
    llm: LLMConfig
    paths: PathsConfig
    ui: UIConfig
    advanced: AdvancedConfig


class InterviewConfigLoader:
    """Loads and validates interview configuration from YAML."""
    
    def __init__(self, config_path: str = "configs/interview_config.yaml"):
        """Initialize the config loader.
        
        Args:
            config_path: Path to the interview configuration file
        """
        self.config_path = Path(config_path)
        self._config_data = None
        
    def load_config(self) -> InterviewConfig:
        """Load and validate interview configuration.
        
        Returns:
            InterviewConfig with validated settings
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        logger.info(f"Loading interview configuration from {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Interview config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML config: {e}")
        
        return self._validate_and_create_config()
    
    def _validate_and_create_config(self) -> InterviewConfig:
        """Validate configuration and create InterviewConfig object."""
        try:
            # Interview flow
            interview_flow_data = self._config_data.get('interview_flow', {})
            interview_flow = InterviewFlowConfig(
                max_turns=interview_flow_data.get('max_turns', 20),
                min_turns=interview_flow_data.get('min_turns', 5),
                enable_fallback=interview_flow_data.get('enable_fallback', False),
                fallback_questions=interview_flow_data.get('fallback_questions', [
                    "Can you tell me more about that?",
                    "What else comes to mind?",
                    "How does that make you feel?"
                ])
            )
            
            # Graph needs
            graph_needs_data = self._config_data.get('graph_needs', {})
            strategy_weights = graph_needs_data.get('strategy_weights', {
                'seed_expansion': 0.9,
                'bridge_building': 0.7,
                'depth_completion': 0.6
            })
            graph_needs = GraphNeedsConfig(
                min_nodes_for_seed_expansion=graph_needs_data.get('min_nodes_for_seed_expansion', 4),
                isolation_threshold=graph_needs_data.get('isolation_threshold', 0.1),
                depth_completion_threshold=graph_needs_data.get('depth_completion_threshold', 0.3),
                target_depth=graph_needs_data.get('target_depth', 5),
                dead_end_threshold=graph_needs_data.get('dead_end_threshold', 0.6),
                dead_end_probe_count=graph_needs_data.get('dead_end_probe_count', 3),
                strategy_weights=strategy_weights
            )
            
            # Extraction
            extraction_data = self._config_data.get('extraction', {})
            extraction = ExtractionConfig(
                confidence_threshold=extraction_data.get('confidence_threshold', 0.6),
                validation_stages=extraction_data.get('validation_stages', 2),
                max_retries=extraction_data.get('max_retries', 2),
                max_history_turns=extraction_data.get('max_history_turns', 3),
                existing_nodes_limit=extraction_data.get('existing_nodes_limit', 10)
            )
            
            # Tactic selection
            tactic_selection_data = self._config_data.get('tactic_selection', {})
            tactic_selection = TacticSelectionConfig(
                usage_penalty_weight=tactic_selection_data.get('usage_penalty_weight', 0.7),
                recency_penalty_weight=tactic_selection_data.get('recency_penalty_weight', 0.15),
                recency_penalty_cap=tactic_selection_data.get('recency_penalty_cap', 0.5),
                recent_tactics_count=tactic_selection_data.get('recent_tactics_count', 3),
                recent_questions_count=tactic_selection_data.get('recent_questions_count', 3)
            )
            
            # Question generation
            question_gen_data = self._config_data.get('question_generation', {})
            context_weights = question_gen_data.get('context_weights', {
                'visit_score': 0.7,
                'recency_score': 0.3
            })
            question_generation = QuestionGenerationConfig(
                temperature=question_gen_data.get('temperature', 0.7),
                max_tokens=question_gen_data.get('max_tokens', 150),
                max_question_length=question_gen_data.get('max_question_length', 200),
                context_weights=context_weights
            )
            
            # Logging
            logging_data = self._config_data.get('logging', {})
            logging = LoggingConfig(
                level=logging_data.get('level', 'INFO'),
                log_decisions=logging_data.get('log_decisions', True),
                log_graph_state=logging_data.get('log_graph_state', True),
                log_tactic_selection=logging_data.get('log_tactic_selection', True)
            )
            
            # LLM
            llm_data = self._config_data.get('llm', {})
            llm = LLMConfig(
                default_provider=llm_data.get('default_provider', 'kimi'),
                models=llm_data.get('models', {
                    'anthropic': 'claude-3-5-sonnet-20241022',
                    'openai': 'gpt-4o',
                    'kimi': 'kimi-k2-turbo-preview',
                    'deepseek': 'deepseek-chat'
                }),
                extraction_temperature=llm_data.get('extraction_temperature', 0.3),
                question_temperature=llm_data.get('question_temperature', 0.7),
                max_tokens=llm_data.get('max_tokens', 500),
                request_timeout=llm_data.get('request_timeout', 30),
                max_retries=llm_data.get('max_retries', 2),
                retry_delay=llm_data.get('retry_delay', 1),
                requests_per_minute=llm_data.get('requests_per_minute', 60),
                tokens_per_minute=llm_data.get('tokens_per_minute', 100000)
            )
            
            # Paths
            paths_data = self._config_data.get('paths', {})
            paths = PathsConfig(
                schema_file=paths_data.get('schema_file', 'schemas/means_end_chain_v0.2.yaml'),
                extraction_prompts_file=paths_data.get('extraction_prompts_file', 'extraction_prompts.yaml'),
                strategy_tactic_map_file=paths_data.get('strategy_tactic_map_file', 'strategy_tactic_map.yaml'),
                log_directory=paths_data.get('log_directory', 'data/interviews')
            )
            
            # UI
            ui_data = self._config_data.get('ui', {})
            ui = UIConfig(
                server_name=ui_data.get('server_name', '0.0.0.0'),
                server_port=ui_data.get('server_port', 7860),
                share=ui_data.get('share', False),
                show_error=ui_data.get('show_error', True)
            )
            
            # Advanced
            advanced_data = self._config_data.get('advanced', {})
            advanced = AdvancedConfig(
                enable_focus_stack=advanced_data.get('enable_focus_stack', True),
                enable_dead_end_detection=advanced_data.get('enable_dead_end_detection', True),
                enable_coreference_resolution=advanced_data.get('enable_coreference_resolution', True),
                enable_graph_visualization=advanced_data.get('enable_graph_visualization', False)
            )
            
            return InterviewConfig(
                interview_flow=interview_flow,
                graph_needs=graph_needs,
                extraction=extraction,
                tactic_selection=tactic_selection,
                question_generation=question_generation,
                logging=logging,
                llm=llm,
                paths=paths,
                ui=ui,
                advanced=advanced
            )
            
        except Exception as e:
            logger.error(f"Failed to create InterviewConfig: {e}")
            raise ValueError(f"Invalid interview configuration: {e}")
    
    def get_available_sections(self) -> List[str]:
        """Get list of available configuration sections."""
        if not self._config_data:
            return []
        return list(self._config_data.keys())
    
    def validate_section(self, section_name: str) -> bool:
        """Validate that a configuration section exists."""
        try:
            if not self._config_data:
                # Try to load config without validation (just YAML parsing)
                if self.config_path.exists():
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        self._config_data = yaml.safe_load(f)
                else:
                    return False
            
            return section_name in self._config_data
            
        except Exception:
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration without loading full config."""
        if not self._config_data:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "available_sections": self.get_available_sections(),
            "interview_flow": {
                "max_turns": self._config_data.get('interview_flow', {}).get('max_turns'),
                "min_turns": self._config_data.get('interview_flow', {}).get('min_turns')
            },
            "graph_needs": {
                "default_provider": self._config_data.get('llm', {}).get('default_provider'),
                "target_depth": self._config_data.get('graph_needs', {}).get('target_depth')
            }
        }