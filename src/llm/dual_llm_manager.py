"""
Dual LLM Manager - Uses optimized three-section configuration.
Manages two LLM clients for different tasks with provider-agnostic specs.
"""

import logging
from typing import Optional, Dict, Any, List
from src.llm.client import BaseLLMClient, LLMResponse
from src.llm.factory import LLMClientFactory
from src.config.llm_config_loader import LLMConfigLoader, DualLLMConfig, LLMProviderConfig, ExtractionSpecs

logger = logging.getLogger(__name__)


class DualLLMManager:
    """Manages two LLM clients using optimized three-section configuration."""
    
    def __init__(self, config_loader: Optional[LLMConfigLoader] = None):
        """Initialize the dual LLM manager.
        
        Args:
            config_loader: Optional config loader (creates default if not provided)
        """
        self.config_loader = config_loader or LLMConfigLoader()
        self._config: Optional[DualLLMConfig] = None
        self._graph_extraction_client: Optional[BaseLLMClient] = None
        self._question_generation_client: Optional[BaseLLMClient] = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the dual LLM manager with optimized configuration.
        
        Returns:
            True if initialization successful
            
        Raises:
            RuntimeError: If configuration is invalid or clients cannot be created
        """
        try:
            logger.info("Initializing dual LLM manager with optimized configuration")
            
            # Load optimized configuration
            self._config = self.config_loader.load_config()
            
            # Get extraction specs for parameter application
            extraction_specs = self._config.extraction_specs
            
            # Create graph extraction client with specs from config
            graph_config = self._config.graph_extraction_config
            graph_specs = extraction_specs if extraction_specs else self._get_default_specs()
            
            # Only pass base_url to providers that support it (Kimi and DeepSeek)
            client_kwargs = {}
            if self._config.graph_extraction_provider in ['kimi', 'deepseek'] and graph_config.base_url:
                client_kwargs['base_url'] = graph_config.base_url
            
            self._graph_extraction_client = LLMClientFactory.create_client(
                provider=self._config.graph_extraction_provider,
                api_key=graph_config.api_key,
                model=graph_config.graph_extraction_model,
                temperature=graph_specs.graph_extraction_temperature,
                max_tokens=graph_specs.graph_extraction_max_tokens,
                **client_kwargs
            )
            
            # Create question generation client with specs from config
            question_config = self._config.question_generation_config
            question_specs = extraction_specs if extraction_specs else self._get_default_specs()
            
            # Only pass base_url to providers that support it (Kimi and DeepSeek)
            client_kwargs = {}
            if self._config.question_generation_provider in ['kimi', 'deepseek'] and question_config.base_url:
                client_kwargs['base_url'] = question_config.base_url
            
            self._question_generation_client = LLMClientFactory.create_client(
                provider=self._config.question_generation_provider,
                api_key=question_config.api_key,
                model=question_config.question_generation_model,
                temperature=question_specs.question_generation_temperature,
                max_tokens=question_specs.question_generation_max_tokens,
                **client_kwargs
            )
            
            self._initialized = True
            logger.info(
                f"Dual LLM manager initialized: "
                f"extraction={self._config.graph_extraction_provider} "
                f"({self._config.graph_extraction_config.graph_extraction_model}), "
                f"generation={self._config.question_generation_provider} "
                f"({self._config.question_generation_config.question_generation_model})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize dual LLM manager: {e}")
            raise RuntimeError(f"Dual LLM initialization failed: {e}")
    
    def _get_default_specs(self) -> ExtractionSpecs:
        """Get default extraction specs if config loading failed."""
        return ExtractionSpecs(
            graph_extraction_temperature=0.3,
            graph_extraction_max_tokens=1000,
            graph_extraction_timeout=15,
            question_generation_temperature=0.7,
            question_generation_max_tokens=300,
            question_generation_timeout=20
        )
    
    def is_initialized(self) -> bool:
        """Check if the manager is properly initialized."""
        return self._initialized and self._graph_extraction_client is not None and self._question_generation_client is not None
    
    async def generate_graph_extraction(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate completion for graph extraction task using config specs.
        
        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            
        Returns:
            LLM response
            
        Raises:
            RuntimeError: If not initialized or generation fails
        """
        if not self.is_initialized():
            raise RuntimeError("Dual LLM manager not initialized")
        
        try:
            logger.debug(f"Generating graph extraction with {self._config.graph_extraction_provider}")
            logger.debug(f"Using specs: temp={self._config.extraction_specs.graph_extraction_temperature}, "
                        f"tokens={self._config.extraction_specs.graph_extraction_max_tokens}, "
                        f"timeout={self._config.extraction_specs.graph_extraction_timeout}")
            
            response = await self._graph_extraction_client.generate_completion(messages, system_prompt)
            
            logger.debug(f"Graph extraction completed: {response.tokens_used} tokens, {response.latency_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            raise RuntimeError(f"Graph extraction failed: {e}")
    
    async def generate_question_generation(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate completion for question generation task using config specs.
        
        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            
        Returns:
            LLM response
            
        Raises:
            RuntimeError: If not initialized or generation fails
        """
        if not self.is_initialized():
            raise RuntimeError("Dual LLM manager not initialized")
        
        try:
            logger.debug(f"Generating question with {self._config.question_generation_provider}")
            logger.debug(f"Using specs: temp={self._config.extraction_specs.question_generation_temperature}, "
                        f"tokens={self._config.extraction_specs.question_generation_max_tokens}, "
                        f"timeout={self._config.extraction_specs.question_generation_timeout}")
            
            response = await self._question_generation_client.generate_completion(messages, system_prompt)
            
            logger.debug(f"Question generation completed: {response.tokens_used} tokens, {response.latency_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            raise RuntimeError(f"Question generation failed: {e}")
    
    async def generate_graph_extraction_with_function_call(self, messages: List[Dict[str, str]], function_schema: Dict[str, Any], system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate completion with function calling for graph extraction using config specs.
        
        Args:
            messages: Conversation messages
            function_schema: Function calling schema
            system_prompt: Optional system prompt
            
        Returns:
            LLM response with function call
            
        Raises:
            RuntimeError: If not initialized or generation fails
        """
        if not self.is_initialized():
            raise RuntimeError("Dual LLM manager not initialized")
        
        try:
            logger.debug(f"Generating graph extraction with function calling using {self._config.graph_extraction_provider}")
            logger.debug(f"Using specs: temp={self._config.extraction_specs.graph_extraction_temperature}, "
                        f"tokens={self._config.extraction_specs.graph_extraction_max_tokens}, "
                        f"timeout={self._config.extraction_specs.graph_extraction_timeout}")
            
            response = await self._graph_extraction_client.generate_completion_with_function_call(messages, function_schema, system_prompt)
            
            logger.debug(f"Graph extraction with function calling completed: {response.tokens_used} tokens, {response.latency_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Graph extraction with function calling failed: {e}")
            raise RuntimeError(f"Graph extraction with function calling failed: {e}")
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about current providers and their specs."""
        if not self.is_initialized():
            return {"status": "not_initialized"}
        
        extraction_specs = self._config.extraction_specs
        
        return {
            "graph_extraction_provider": self._config.graph_extraction_provider,
            "graph_extraction_model": self._config.graph_extraction_config.graph_extraction_model,
            "graph_extraction_specs": {
                "temperature": extraction_specs.graph_extraction_temperature,
                "max_tokens": extraction_specs.graph_extraction_max_tokens,
                "timeout": extraction_specs.graph_extraction_timeout
            },
            "question_generation_provider": self._config.question_generation_provider,
            "question_generation_model": self._config.question_generation_config.question_generation_model,
            "question_generation_specs": {
                "temperature": extraction_specs.question_generation_temperature,
                "max_tokens": extraction_specs.question_generation_max_tokens,
                "timeout": extraction_specs.question_generation_timeout
            },
            "status": "initialized"
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get detailed summary of current configuration."""
        if not self.is_initialized():
            return {"initialized": False}
        
        extraction_specs = self._config.extraction_specs
        
        return {
            "initialized": True,
            "model_selection": {
                "graph_extraction_model": self._config.graph_extraction_provider,
                "question_generation_model": self._config.question_generation_provider
            },
            "extraction_specs": {
                "graph_extraction": {
                    "temperature": extraction_specs.graph_extraction_temperature,
                    "max_tokens": extraction_specs.graph_extraction_max_tokens,
                    "timeout_seconds": extraction_specs.graph_extraction_timeout
                },
                "question_generation": {
                    "temperature": extraction_specs.question_generation_temperature,
                    "max_tokens": extraction_specs.question_generation_max_tokens,
                    "timeout_seconds": extraction_specs.question_generation_timeout
                }
            },
            "providers": {
                "graph_extraction": {
                    "provider": self._config.graph_extraction_provider,
                    "model": self._config.graph_extraction_config.graph_extraction_model,
                    "base_url": self._config.graph_extraction_config.base_url,
                    "request_timeout": self._config.graph_extraction_config.request_timeout
                },
                "question_generation": {
                    "provider": self._config.question_generation_provider,
                    "model": self._config.question_generation_config.question_generation_model,
                    "base_url": self._config.question_generation_config.base_url,
                    "request_timeout": self._config.question_generation_config.request_timeout
                }
            },
            "global_settings": {
                "retry_config": self._config.retry_config,
                "rate_limits": self._config.rate_limits
            }
        }
    
    async def shutdown(self):
        """Shutdown the dual LLM manager and cleanup resources."""
        logger.info("Shutting down dual LLM manager")
        self._initialized = False
        self._graph_extraction_client = None
        self._question_generation_client = None
        self._config = None