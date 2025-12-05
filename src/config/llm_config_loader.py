"""
LLM Configuration Loader - Uses optimized three-section architecture.
Single configuration file with provider-agnostic extraction specs.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class LLMProviderConfig:
    """Configuration for a specific LLM provider."""
    api_key: str
    base_url: Optional[str]
    request_timeout: int
    graph_extraction_model: str
    question_generation_model: str


@dataclass
class ExtractionSpecs:
    """Provider-agnostic extraction specifications."""
    graph_extraction_temperature: float
    graph_extraction_max_tokens: int
    graph_extraction_timeout: int
    question_generation_temperature: float
    question_generation_max_tokens: int
    question_generation_timeout: int


@dataclass
class DualLLMConfig:
    """Configuration for dual LLM setup."""
    graph_extraction_provider: str
    question_generation_provider: str
    extraction_specs: ExtractionSpecs
    graph_extraction_config: LLMProviderConfig
    question_generation_config: LLMProviderConfig
    retry_config: Dict[str, Any]
    rate_limits: Dict[str, Any]


class LLMConfigLoader:
    """Loads and validates optimized LLM configuration from YAML."""
    
    def __init__(self, config_path: str = "configs/llm_config.yaml"):
        """Initialize the config loader.
        
        Args:
            config_path: Path to the LLM configuration file
        """
        self.config_path = Path(config_path)
        self._config_data = None
        
    def load_config(self) -> DualLLMConfig:
        """Load and validate LLM configuration.
        
        Returns:
            DualLLMConfig with validated settings
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        logger.info(f"Loading LLM configuration from {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"LLM config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML config: {e}")
        
        return self._validate_and_create_config()
    
    def _validate_and_create_config(self) -> DualLLMConfig:
        """Validate configuration and create DualLLMConfig object."""
        # Section 1: Validate model selection
        graph_provider = self._config_data.get('graph_extraction_model')
        question_provider = self._config_data.get('question_generation_model')
        
        if not graph_provider or not question_provider:
            raise ValueError("Both graph_extraction_model and question_generation_model must be specified")
        
        providers = self._config_data.get('providers', {})
        
        if graph_provider not in providers:
            raise ValueError(f"Graph extraction provider '{graph_provider}' not found in providers section")
        
        if question_provider not in providers:
            raise ValueError(f"Question generation provider '{question_provider}' not found in providers section")
        
        # Section 2: Get provider-agnostic extraction specs
        extraction_specs_data = self._config_data.get('extraction_specs', {})
        extraction_specs = self._create_extraction_specs(extraction_specs_data)
        
        # Section 3: Create provider configurations with merged specs
        graph_provider_data = providers[graph_provider]
        graph_config = self._create_provider_config(
            graph_provider, 
            graph_provider_data, 
            extraction_specs,
            is_graph_extraction=True
        )
        
        question_provider_data = providers[question_provider]
        question_config = self._create_provider_config(
            question_provider, 
            question_provider_data, 
            extraction_specs,
            is_graph_extraction=False
        )
        
        # Section 4: Get global settings
        retry_config = self._config_data.get('retry_config', {
            'max_retries': 2,
            'initial_delay_seconds': 1,
            'backoff_multiplier': 2,
            'max_delay_seconds': 30
        })
        
        rate_limits = self._config_data.get('rate_limits', {
            'requests_per_minute': 60,
            'tokens_per_minute': 100000
        })
        
        logger.info(f"Loaded dual LLM config: extraction={graph_provider}, generation={question_provider}")
        
        return DualLLMConfig(
            graph_extraction_provider=graph_provider,
            question_generation_provider=question_provider,
            extraction_specs=extraction_specs,
            graph_extraction_config=graph_config,
            question_generation_config=question_config,
            retry_config=retry_config,
            rate_limits=rate_limits
        )
    
    def _create_extraction_specs(self, specs_data: Dict[str, Any]) -> ExtractionSpecs:
        """Create provider-agnostic extraction specifications."""
        graph_specs = specs_data.get('graph_extraction', {})
        question_specs = specs_data.get('question_generation', {})
        
        return ExtractionSpecs(
            graph_extraction_temperature=graph_specs.get('temperature', 0.3),
            graph_extraction_max_tokens=graph_specs.get('max_tokens', 1000),
            graph_extraction_timeout=graph_specs.get('timeout_seconds', 15),
            question_generation_temperature=question_specs.get('temperature', 0.7),
            question_generation_max_tokens=question_specs.get('max_tokens', 300),
            question_generation_timeout=question_specs.get('timeout_seconds', 20)
        )
    
    def _create_provider_config(
        self, 
        provider_name: str, 
        provider_data: Dict[str, Any], 
        extraction_specs: ExtractionSpecs,
        is_graph_extraction: bool
    ) -> LLMProviderConfig:
        """Create provider configuration with merged specs."""
        # Get API key from environment
        api_key_env = provider_data.get('api_key_env')
        if not api_key_env:
            raise ValueError(f"Provider '{provider_name}' missing api_key_env")
        
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key environment variable '{api_key_env}' not set for provider '{provider_name}'")
        
        # Get models from provider data
        models = provider_data.get('models', {})
        graph_model = models.get('graph_extraction', 'unknown')
        question_model = models.get('question_generation', 'unknown')
        
        return LLMProviderConfig(
            api_key=api_key,
            base_url=provider_data.get('base_url'),
            request_timeout=provider_data.get('request_timeout', 30),
            graph_extraction_model=graph_model,
            question_generation_model=question_model
        )
    
    def get_available_providers(self) -> list:
        """Get list of available providers from config."""
        if not self._config_data:
            return []
        return list(self._config_data.get('providers', {}).keys())
    
    def validate_provider(self, provider_name: str) -> bool:
        """Validate that a provider is properly configured."""
        try:
            # Ensure we have config data loaded
            if not self._config_data:
                # Try to load config without validation (just YAML parsing)
                if self.config_path.exists():
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        self._config_data = yaml.safe_load(f)
                else:
                    return False
            
            providers = self._config_data.get('providers', {})
            if provider_name not in providers:
                return False
            
            provider_data = providers[provider_name]
            api_key_env = provider_data.get('api_key_env')
            if not api_key_env:
                return False
            
            api_key = os.getenv(api_key_env)
            return api_key is not None and len(api_key) > 0
            
        except Exception:
            return False
    
    def get_extraction_specs(self) -> Optional[ExtractionSpecs]:
        """Get provider-agnostic extraction specifications."""
        if not self._config_data:
            return None
        
        extraction_specs_data = self._config_data.get('extraction_specs', {})
        return self._create_extraction_specs(extraction_specs_data)
    
    def get_current_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration without loading full config."""
        if not self._config_data:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "graph_extraction_model": self._config_data.get('graph_extraction_model'),
            "question_generation_model": self._config_data.get('question_generation_model'),
            "available_providers": self.get_available_providers(),
            "extraction_specs": {
                "graph_extraction": self._config_data.get('extraction_specs', {}).get('graph_extraction', {}),
                "question_generation": self._config_data.get('extraction_specs', {}).get('question_generation', {})
            }
        }