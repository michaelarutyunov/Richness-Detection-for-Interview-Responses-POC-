"""
LLM Client Factory - Creates appropriate LLM clients based on configuration.
"""

import logging
from typing import Optional, Dict, Any
from src.llm.client import BaseLLMClient, AnthropicClient, OpenAIClient, KimiClient, DeepSeekClient, LLMProvider


logger = logging.getLogger(__name__)


class LLMClientFactory:
    """Factory for creating LLM clients based on provider and configuration."""
    
    # Provider-to-client mapping
    CLIENT_MAP = {
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.KIMI: KimiClient,
        LLMProvider.DEEPSEEK: DeepSeekClient,
    }
    
    @staticmethod
    def create_client(
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client based on provider and configuration.
        
        Args:
            provider: LLM provider name (anthropic, openai, kimi)
            api_key: API key for the provider
            model: Specific model to use
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Configured LLM client instance
            
        Raises:
            ValueError: If provider is not supported
            RuntimeError: If client creation fails
        """
        logger.info(f"Creating LLM client for provider: {provider}")
        
        try:
            # Convert string to enum
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported: {list(LLMProvider)}")
        
        # Get client class
        client_class = LLMClientFactory.CLIENT_MAP.get(provider_enum)
        if not client_class:
            raise ValueError(f"No client class found for provider: {provider}")
        
        # Set default models if not specified
        if not model:
            model = LLMClientFactory._get_default_model(provider_enum)
        
        # Create client with base parameters
        base_params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Merge with provider-specific parameters
        client_params = {**base_params, **kwargs}
        
        # Add API key for providers that need it
        if provider_enum in [LLMProvider.ANTHROPIC, LLMProvider.OPENAI, LLMProvider.KIMI, LLMProvider.DEEPSEEK]:
            client_params["api_key"] = api_key
        
        try:
            # Create client instance
            client = client_class(**client_params)
            
            # Validate configuration
            if not client.validate_config():
                raise RuntimeError(f"Invalid configuration for {provider} client")
            
            logger.info(f"Successfully created {provider} client with model: {model}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create {provider} client: {e}")
            raise RuntimeError(f"Failed to create LLM client: {e}")
    
    @staticmethod
    def _get_default_model(provider: LLMProvider) -> str:
        """Get default model for provider."""
        defaults = {
            LLMProvider.KIMI: "moonshot-v1-8k",        # Default: cost-effective, good for interviews
            LLMProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",  # Premium option - Claude Sonnet 4.5
            LLMProvider.OPENAI: "gpt-4",              # Alternative option
            LLMProvider.DEEPSEEK: "deepseek-chat",    # New option - cost-effective
        }
        return defaults.get(provider, "unknown")
    
    @staticmethod
    def create_client_from_config(config: Dict[str, Any]) -> BaseLLMClient:
        """
        Create LLM client from configuration dictionary.
        
        Args:
            config: Configuration dictionary with provider, api_key, etc.
            
        Returns:
            Configured LLM client
        """
        required_fields = ["provider", "api_key"]
        
        # Validate required fields
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Extract configuration
        provider = config["provider"]
        api_key = config["api_key"]
        model = config.get("model")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 500)
        
        # Remove provider-specific fields for kwargs
        kwargs = {k: v for k, v in config.items() if k not in required_fields + ["model", "temperature", "max_tokens"]}
        
        return LLMClientFactory.create_client(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    @staticmethod
    def get_supported_providers() -> list:
        """Get list of supported LLM providers."""
        return [provider.value for provider in LLMProvider]
    
    @staticmethod
    def validate_provider_config(provider: str, api_key: str, model: Optional[str] = None) -> bool:
        """
        Validate provider configuration without creating client.
        
        Args:
            provider: Provider name
            api_key: API key
            model: Model name (optional)
            
        Returns:
            True if configuration is valid
        """
        try:
            provider_enum = LLMProvider(provider.lower())
            client_class = LLMClientFactory.CLIENT_MAP.get(provider_enum)
            
            if not client_class:
                return False
            
            # Create a temporary client for validation
            temp_config = {
                "api_key": api_key,
                "model": model or LLMClientFactory._get_default_model(provider_enum)
            }
            
            temp_client = client_class(**temp_config)
            return temp_client.validate_config()
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Convenience functions
def create_default_clients() -> Dict[str, BaseLLMClient]:
    """Create default clients for common providers (requires env vars)."""
    import os
    
    clients = {}
    
    # Kimi client (default/first choice - cost-effective)
    kimi_key = os.getenv("KIMI_API_KEY")
    if kimi_key:
        try:
            clients["kimi"] = LLMClientFactory.create_client(
                provider="kimi",
                api_key=kimi_key,
                temperature=0.3,  # Lower temperature for more focused questions
                max_tokens=150
            )
        except Exception as e:
            logger.warning(f"Could not create Kimi client: {e}")
    
    # Anthropic client (premium option)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            clients["anthropic"] = LLMClientFactory.create_client(
                provider="anthropic",
                api_key=anthropic_key,
                temperature=0.7
            )
        except Exception as e:
            logger.warning(f"Could not create Anthropic client: {e}")
    
    # OpenAI client (alternative option)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            clients["openai"] = LLMClientFactory.create_client(
                provider="openai",
                api_key=openai_key,
                temperature=0.7
            )
        except Exception as e:
            logger.warning(f"Could not create OpenAI client: {e}")
    
    # DeepSeek client (new option)
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        try:
            clients["deepseek"] = LLMClientFactory.create_client(
                provider="deepseek",
                api_key=deepseek_key,
                temperature=0.5  # Balanced temperature for DeepSeek
            )
        except Exception as e:
            logger.warning(f"Could not create DeepSeek client: {e}")

    return clients