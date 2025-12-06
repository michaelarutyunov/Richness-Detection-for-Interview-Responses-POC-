"""
Unified LLM client interface for multiple providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    KIMI = "kimi"
    DEEPSEEK = "deepseek"


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None  # token usage info
    metadata: Optional[Dict[str, Any]] = None
    function_call: Optional[Dict[str, Any]] = None  # For function calling responses
    model_used: str = ""  # Actual model used
    latency_ms: int = 0  # Response latency
    tokens_used: int = 0  # Total tokens used


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 500, provider: str = "unknown"):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 500)
            provider: Provider name (default: "unknown")
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        logger.info(f"Initialized {self.__class__.__name__} with model: {model}, provider: {provider}")
    
    @abstractmethod
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate a completion from the LLM."""
        pass
    
    async def generate_completion_with_function_call(
        self,
        messages: List[Dict[str, str]],
        function_schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion with function calling (default implementation)."""
        # Default implementation - override in subclasses that support function calling
        logger.warning(f"Function calling not implemented for {self.provider}, falling back to regular completion")
        return await self.generate_completion(messages, system_prompt)
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate client configuration (API keys, etc.)."""
        pass
    
    async def _test_connectivity(self, test_message: str = "Hi") -> bool:
        """
        Test API connectivity with a lightweight call.
        
        Args:
            test_message: Simple message to send for connectivity test
            
        Returns:
            True if API is reachable and functional
        """
        try:
            response = await self.generate_completion(
                messages=[{"role": "user", "content": test_message}],
                system_prompt=None
            )
            return response is not None and len(response.content.strip()) > 0
        except Exception as e:
            logger.error(f"API connectivity test failed for {self.provider}: {e}")
            return False
    
    def validate_config_with_connectivity(self) -> bool:
        """
        Validate configuration including API connectivity test.
        
        Returns:
            True if configuration and connectivity are valid
        """
        # First do basic config validation
        if not self.validate_config():
            return False
            
        # Test API connectivity
        try:
            import asyncio
            
            # Run connectivity test in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._test_connectivity())
                if result:
                    logger.info(f"{self.provider.title()} API connectivity validated successfully")
                    return True
                else:
                    return False
            finally:
                loop.close()
                asyncio.set_event_loop(None)
                
        except Exception as e:
            logger.error(f"{self.provider.title()} API connectivity test error: {e}")
            return False
    
    def _prepare_messages(self, messages: List[Dict[str, str]], system_prompt: Optional[str]) -> List[Dict[str, str]]:
        """Prepare messages for API call."""
        prepared_messages = []
        
        if system_prompt:
            prepared_messages.append({"role": "system", "content": system_prompt})
        
        prepared_messages.extend(messages)
        return prepared_messages


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", **kwargs):
        """Initialize Anthropic client."""
        super().__init__(model, provider=LLMProvider.ANTHROPIC, **kwargs)
        self.api_key = api_key
        
        # Import anthropic library only when needed
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            logger.warning("anthropic library not installed. Install with: pip install anthropic")
            self.client = None
    
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion using Anthropic Claude."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized. Check library installation.")
        
        try:
            # Convert messages to Anthropic format
            # Extract system message if present
            system_msg = None
            chat_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    chat_messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Build API call parameters
            kwargs = {
                "model": self.model,
                "messages": chat_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            
            if system_msg:
                kwargs["system"] = system_msg
            
            response = await self.client.messages.create(**kwargs)
            
            # Extract response content - concatenate all text blocks
            content_parts = []
            if response.content and len(response.content) > 0:
                # Anthropic response.content is a list of content blocks
                for content_block in response.content:
                    if hasattr(content_block, 'text') and content_block.type == 'text':
                        content_parts.append(content_block.text)

            content = " ".join(content_parts) if content_parts else ""

            if not content:
                logger.error("No text content in Anthropic response - possible safety filter")
                raise ValueError("LLM returned empty response - cannot proceed")
            
            return LLMResponse(
                content=content,
                provider=self.provider.value,
                model=self.model,
                model_used=self.model,
                latency_ms=0,  # Will be measured by caller
                tokens_used=0 if response.usage is None else (response.usage.input_tokens + response.usage.output_tokens)
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Failed to generate completion: {e}")
    
    async def generate_completion_with_function_call(
        self,
        messages: List[Dict[str, str]],
        function_schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion with function calling using Anthropic Claude."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized. Check library installation.")
        
        import time
        start_time = time.time()
        
        try:
            # Convert messages to Anthropic format
            system_msg = system_prompt
            chat_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    chat_messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Build API call parameters with function calling
            kwargs = {
                "model": self.model,
                "messages": chat_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "tools": [function_schema],  # Anthropic uses 'tools' for function calling
            }
            
            if system_msg:
                kwargs["system"] = system_msg
            
            response = await self.client.messages.create(**kwargs)
            
            # Extract function call or content
            function_call = None
            content = ""
            
            if response.content and len(response.content) > 0:
                for content_block in response.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == 'tool_use':
                            # Extract function call
                            function_call = {
                                "name": content_block.name,
                                "arguments": content_block.input
                            }
                        elif content_block.type == 'text':
                            content = content_block.text
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=content,
                provider=self.provider.value,
                model=self.model,
                model_used=self.model,
                latency_ms=latency_ms,
                tokens_used=0 if response.usage is None else (response.usage.input_tokens + response.usage.output_tokens),
                function_call=function_call
            )
            
        except Exception as e:
            logger.error(f"Anthropic function calling error: {e}")
            raise RuntimeError(f"Failed to generate completion with function calling: {e}")
    
    def validate_config(self) -> bool:
        """Validate Anthropic configuration (basic check - override for connectivity test)."""
        if not self.api_key:
            logger.error("Anthropic API key not provided")
            return False
        
        if not self.client:
            logger.error("Anthropic client not initialized")
            return False
        
        return True
    
    def validate_config_with_connectivity(self) -> bool:
        """Validate Anthropic configuration including API connectivity test."""
        # Use the base class method for connectivity testing
        return super().validate_config_with_connectivity()


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API client using OpenAI-compatible interface."""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: Optional[str] = None, **kwargs):
        """Initialize DeepSeek client."""
        # Extract base_url before calling super() to avoid passing it to base class
        self.base_url = base_url or "https://api.deepseek.com/v1"
        # Only pass temperature and max_tokens to base class
        base_kwargs = {k: v for k, v in kwargs.items() if k in ['temperature', 'max_tokens']}
        super().__init__(model, provider=LLMProvider.DEEPSEEK, **base_kwargs)
        self.api_key = api_key
        self.provider = LLMProvider.DEEPSEEK
        
        try:
            import openai  # DeepSeek uses OpenAI-compatible API
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
        except ImportError:
            logger.warning("openai library not installed for DeepSeek client")
            self.client = None
    
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion using DeepSeek API."""
        if not self.client:
            raise RuntimeError("DeepSeek client not initialized. Check library installation.")
        
        try:
            # Prepare messages (OpenAI-compatible format)
            deepseek_messages = self._prepare_messages(messages, system_prompt)
            
            # Make API call to DeepSeek
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=deepseek_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract response content
            content = response.choices[0].message.content if response.choices else ""

            return LLMResponse(
                content=content,
                provider=self.provider.value,
                model=self.model,
                usage={
                    "prompt_tokens": 0 if response.usage is None else response.usage.prompt_tokens,
                    "completion_tokens": 0 if response.usage is None else response.usage.completion_tokens,
                    "total_tokens": 0 if response.usage is None else response.usage.total_tokens
                }
            )

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise RuntimeError(f"Failed to generate completion: {e}")
    
    def validate_config(self) -> bool:
        """Validate DeepSeek configuration."""
        if not self.api_key:
            logger.error("DeepSeek API key not provided")
            return False
        
        if not self.client:
            logger.error("DeepSeek client not initialized")
            return False
        
        return True


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", **kwargs):
        """Initialize OpenAI client."""
        super().__init__(model, provider="openai", **kwargs)
        self.api_key = api_key
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            logger.warning("openai library not installed. Install with: pip install openai")
            self.client = None
    
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion using OpenAI GPT."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Check library installation.")
        
        try:
            # Prepare messages (OpenAI format)
            openai_messages = self._prepare_messages(messages, system_prompt)
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract response content
            content = response.choices[0].message.content if response.choices else ""

            return LLMResponse(
                content=content,
                provider=self.provider.value,
                model=self.model,
                usage={
                    "prompt_tokens": 0 if response.usage is None else response.usage.prompt_tokens,
                    "completion_tokens": 0 if response.usage is None else response.usage.completion_tokens,
                    "total_tokens": 0 if response.usage is None else response.usage.total_tokens
                }
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Failed to generate completion: {e}")
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration (basic check - override for connectivity test)."""
        if not self.api_key:
            logger.error("OpenAI API key not provided")
            return False
        
        if not self.client:
            logger.error("OpenAI client not initialized")
            return False
        
        return True
    
    def validate_config_with_connectivity(self) -> bool:
        """Validate OpenAI configuration including API connectivity test."""
        # Use the base class method for connectivity testing
        return super().validate_config_with_connectivity()


class KimiClient(BaseLLMClient):
    """Moonshot Kimi API client."""
    
    def __init__(self, api_key: str, model: str = "moonshot-v1-8k", base_url: Optional[str] = None, **kwargs):
        """Initialize Kimi client."""
        # Extract base_url before calling super() to avoid passing it to base class
        self.base_url = base_url or "https://api.moonshot.ai/v1"
        # Only pass temperature and max_tokens to base class
        base_kwargs = {k: v for k, v in kwargs.items() if k in ['temperature', 'max_tokens']}
        super().__init__(model, provider=LLMProvider.KIMI, **base_kwargs)
        self.api_key = api_key
        self.provider = LLMProvider.KIMI
        
        try:
            import openai  # Kimi uses OpenAI-compatible API
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
        except ImportError:
            logger.warning("openai library not installed for Kimi client")
            self.client = None
    
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion using Kimi (Moonshot)."""
        if not self.client:
            raise RuntimeError("Kimi client not initialized. Check library installation.")
        
        try:
            # Prepare messages (OpenAI-compatible format) - improved from working archive
            kimi_messages = self._prepare_messages(messages, system_prompt)
            
            # Make API call with timeout parameter from working archive
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=kimi_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30  # Add timeout from working archive version
            )
            
            # Extract response content - improved from working archive version
            content = ""
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    content = message.content
            
            if not content:
                logger.warning("No content found in Kimi response")
                content = "Can you tell me more about that?"  # Fallback
            
            return LLMResponse(
                content=content,
                provider=self.provider.value,
                model=self.model,
                usage={
                    "prompt_tokens": 0 if response.usage is None else response.usage.prompt_tokens,
                    "completion_tokens": 0 if response.usage is None else response.usage.completion_tokens,
                    "total_tokens": 0 if response.usage is None else response.usage.total_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Kimi API error: {e}")
            raise RuntimeError(f"Failed to generate completion: {e}")
    
    def validate_config(self) -> bool:
        """Validate Kimi configuration."""
        if not self.api_key:
            logger.error("Kimi API key not provided")
            return False
        
        if not self.client:
            logger.error("Kimi client not initialized")
            return False
        
        return True