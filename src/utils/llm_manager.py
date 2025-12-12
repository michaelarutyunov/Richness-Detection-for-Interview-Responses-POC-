"""
LLM client management.
Handles multiple providers with unified interface.
"""

import os
import time
import logging
from typing import Dict, Optional, Any, List, Union, Callable
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
import yaml
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

# Import provider SDKs - handle missing gracefully
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def _call_with_timeout(func: Callable, timeout_seconds: int, *args, **kwargs) -> Any:
    """
    Execute function with hard timeout using ThreadPoolExecutor.

    This provides application-level timeout enforcement independent of SDK timeouts,
    which may not be respected by all providers (notably Kimi API).

    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time in seconds
        *args, **kwargs: Arguments to pass to func

    Returns:
        Return value from func

    Raises:
        FuturesTimeoutError: If execution exceeds timeout
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as e:
            logger.error(f"[LLM Timeout] Hard timeout exceeded {timeout_seconds}s")
            future.cancel()
            raise TimeoutError(f"Operation exceeded {timeout_seconds}s timeout") from e


class TaskType(str, Enum):
    """Types of LLM tasks."""
    GRAPH_EXTRACTION = "graph_extraction"
    QUESTION_GENERATION = "question_generation"
    EXTRACTABILITY_CHECK = "extractability_check"
    MOMENTUM_ASSESSMENT = "momentum_assessment"
    PLAUSIBILITY_CHECK = "plausibility_check"


class LLMResponse(BaseModel):
    """Standardized response from LLM."""
    content: str = Field(description="Response text")
    model: str = Field(description="Model used")
    provider: str = Field(description="Provider used")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    latency_ms: int = Field(default=0, description="Response latency in ms")
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)
    function_call: Optional[Dict[str, Any]] = Field(default=None, description="Function call result if using tools")
    cost_input_per_1m: Optional[float] = Field(default=None, description="Cost per 1M input tokens in USD")
    cost_output_per_1m: Optional[float] = Field(default=None, description="Cost per 1M output tokens in USD")

    @property
    def tokens_used(self) -> int:
        """Backward compatibility: total tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> Optional[float]:
        """Calculate cost if pricing available."""
        if self.cost_input_per_1m and self.cost_output_per_1m:
            input_cost = (self.input_tokens / 1_000_000) * self.cost_input_per_1m
            output_cost = (self.output_tokens / 1_000_000) * self.cost_output_per_1m
            return input_cost + output_cost
        return None


class RetryConfig(BaseModel):
    """Configuration for retry logic."""
    max_retries: int = 2
    initial_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 30.0


class ModelConfig(BaseModel):
    """Configuration for a specific model assignment."""
    name: str
    request_timeout: Optional[int] = None
    cost_input: Optional[float] = None
    cost_output: Optional[float] = None


class ProviderConfig(BaseModel):
    """Configuration for a single provider."""
    api_key_env: str
    base_url: Optional[str] = None
    request_timeout: int = 30
    models: Dict[str, Union[str, ModelConfig]] = Field(default_factory=dict)
    organization: Optional[str] = None

    @classmethod
    def model_validate(cls, obj):
        """Override validation to normalize models."""
        if isinstance(obj, dict) and 'models' in obj:
            models = obj['models']
            if isinstance(models, dict):
                obj['models'] = {
                    k: ModelConfig(name=v) if isinstance(v, str) else v
                    for k, v in models.items()
                }
        return super().model_validate(obj)


class ExtractionSpec(BaseModel):
    """Specification for a task type."""
    temperature: float = 0.5
    max_tokens: int = 1000
    timeout_seconds: int = 30


class LLMConfig(BaseModel):
    """Full LLM configuration."""
    graph_extraction_model: str = "anthropic"
    question_generation_model: str = "anthropic"
    extraction_specs: Dict[str, ExtractionSpec] = Field(default_factory=dict)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    
    @classmethod
    def load(cls, path: str) -> "LLMConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Parse extraction specs
        extraction_specs = {}
        for task_name, spec_data in data.get('extraction_specs', {}).items():
            extraction_specs[task_name] = ExtractionSpec(**spec_data)
        
        # Parse providers
        providers = {}
        for provider_name, provider_data in data.get('providers', {}).items():
            providers[provider_name] = ProviderConfig(**provider_data)
        
        # Parse retry config
        retry_config = RetryConfig(**data.get('retry_config', {}))
        
        return cls(
            graph_extraction_model=data.get('graph_extraction_model', 'anthropic'),
            question_generation_model=data.get('question_generation_model', 'anthropic'),
            extraction_specs=extraction_specs,
            providers=providers,
            retry_config=retry_config
        )


class LLMManager:
    """
    Manages LLM clients for different providers.
    Provides unified interface for all LLM operations.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._clients: Dict[str, Any] = {}
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize clients for configured providers."""
        for provider_name, provider_config in self.config.providers.items():
            api_key = os.getenv(provider_config.api_key_env)
            if not api_key:
                logger.warning(f"Skipping provider {provider_name}: {provider_config.api_key_env} not set")
                continue

            if provider_name == "anthropic" and HAS_ANTHROPIC:
                self._clients[provider_name] = Anthropic(api_key=api_key)
                logger.info(f"Initialized {provider_name} client")

            elif provider_name == "openai" and HAS_OPENAI:
                kwargs = {"api_key": api_key}
                if provider_config.base_url:
                    kwargs["base_url"] = provider_config.base_url
                if provider_config.organization:
                    kwargs["organization"] = provider_config.organization
                self._clients[provider_name] = OpenAI(**kwargs)
                logger.info(f"Initialized {provider_name} client")

            elif provider_name in ("kimi", "deepseek") and HAS_OPENAI:
                # These use OpenAI-compatible API
                self._clients[provider_name] = OpenAI(
                    api_key=api_key,
                    base_url=provider_config.base_url
                )
                logger.info(f"Initialized {provider_name} client (OpenAI-compatible)")
    
    @classmethod
    def from_config_file(cls, path: str) -> "LLMManager":
        """Create manager from config file."""
        config = LLMConfig.load(path)
        return cls(config)
    
    def get_provider_for_task(self, task: TaskType) -> str:
        """Determine which provider to use for a task."""
        if task in (TaskType.GRAPH_EXTRACTION, TaskType.EXTRACTABILITY_CHECK, TaskType.MOMENTUM_ASSESSMENT, TaskType.PLAUSIBILITY_CHECK):
            return self.config.graph_extraction_model
        elif task == TaskType.QUESTION_GENERATION:
            return self.config.question_generation_model
        return self.config.graph_extraction_model  # Default
    
    def get_spec_for_task(self, task: TaskType) -> ExtractionSpec:
        """Get extraction spec for a task type."""
        # Map task types to spec names
        spec_map = {
            TaskType.GRAPH_EXTRACTION: "graph_extraction",
            TaskType.QUESTION_GENERATION: "question_generation",
            TaskType.EXTRACTABILITY_CHECK: "graph_extraction",  # Use same as extraction
            TaskType.MOMENTUM_ASSESSMENT: "graph_extraction",
            TaskType.PLAUSIBILITY_CHECK: "graph_extraction",  # Use same as extraction
        }
        spec_name = spec_map.get(task, "graph_extraction")
        return self.config.extraction_specs.get(spec_name, ExtractionSpec())
    
    def complete(
        self,
        task: TaskType,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Execute an LLM completion.

        Args:
            task: Type of task (determines provider and defaults)
            system_prompt: System message
            user_prompt: User message
            temperature: Override default temperature
            max_tokens: Override default max tokens
            tools: Optional list of tool/function schemas for function calling
            tool_choice: Optional tool choice specification

        Returns:
            LLMResponse with content and metadata
        """
        provider_name = self.get_provider_for_task(task)
        spec = self.get_spec_for_task(task)

        # Use overrides or defaults
        temp = temperature if temperature is not None else spec.temperature
        tokens = max_tokens if max_tokens is not None else spec.max_tokens

        # Get client
        client = self._clients.get(provider_name)
        if not client:
            return LLMResponse(
                content="",
                model="unknown",
                provider=provider_name,
                success=False,
                error=f"No client available for provider: {provider_name}"
            )

        # Get model configuration and resolve timeout and pricing
        provider_config = self.config.providers[provider_name]
        task_key = "graph_extraction" if task != TaskType.QUESTION_GENERATION else "question_generation"
        model_config = provider_config.models.get(task_key)

        if isinstance(model_config, str):
            model = model_config
            timeout = provider_config.request_timeout
            cost_input = None
            cost_output = None
        else:
            model = model_config.name
            timeout = model_config.request_timeout or provider_config.request_timeout
            cost_input = model_config.cost_input
            cost_output = model_config.cost_output

        # Execute with retry - use function calling if tools provided
        return self._execute_with_retry(
            client=client,
            provider_name=provider_name,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temp,
            max_tokens=tokens,
            timeout=timeout,
            cost_input=cost_input,
            cost_output=cost_output,
            task=task,
            tools=tools,
            tool_choice=tool_choice
        )
    
    def _execute_with_retry(
        self,
        client: Any,
        provider_name: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        cost_input: Optional[float],
        cost_output: Optional[float],
        task: Optional[TaskType] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Execute request with retry logic."""
        retry = self.config.retry_config
        delay = retry.initial_delay_seconds
        last_error = None
        task_name = task.value if task else "unknown"
        use_tools = tools is not None and len(tools) > 0

        for attempt in range(retry.max_retries + 1):
            try:
                start_time = time.time()
                mode = "with tools" if use_tools else "standard"
                logger.info(f"[LLM Call] {task_name} via {provider_name}/{model} ({mode})")

                if provider_name == "anthropic":
                    if use_tools:
                        response = self._call_anthropic_with_tools(
                            client, model, system_prompt, user_prompt, temperature, max_tokens, tools, timeout
                        )
                    else:
                        response = self._call_anthropic(
                            client, model, system_prompt, user_prompt, temperature, max_tokens, timeout
                        )
                else:
                    # OpenAI-compatible (openai, kimi, deepseek)
                    if use_tools:
                        response = self._call_openai_compatible_with_tools(
                            client, provider_name, model, system_prompt, user_prompt,
                            temperature, max_tokens, timeout, tools, tool_choice
                        )
                    else:
                        response = self._call_openai_compatible(
                            client, provider_name, model, system_prompt, user_prompt, temperature, max_tokens, timeout
                        )

                latency_ms = int((time.time() - start_time) * 1000)
                input_tokens = response.get("input_tokens", 0)
                output_tokens = response.get("output_tokens", 0)
                function_call = response.get("function_call")
                logger.info(f"[LLM Result] {latency_ms}ms, {input_tokens + output_tokens} tokens (in:{input_tokens}, out:{output_tokens}), success=True, has_function_call={function_call is not None}")

                return LLMResponse(
                    content=response["content"],
                    model=model,
                    provider=provider_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    success=True,
                    function_call=function_call,
                    cost_input_per_1m=cost_input,
                    cost_output_per_1m=cost_output
                )

            except Exception as e:
                last_error = str(e)
                logger.error(f"[LLM Error] {provider_name}: {last_error}")
                if attempt < retry.max_retries:
                    time.sleep(delay)
                    delay = min(delay * retry.backoff_multiplier, retry.max_delay_seconds)

        return LLMResponse(
            content="",
            model=model,
            provider=provider_name,
            success=False,
            error=f"Failed after {retry.max_retries + 1} attempts: {last_error}"
        )
    
    def _call_anthropic(
        self,
        client: "Anthropic",
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: int
    ) -> Dict:
        """Execute Anthropic API call with hard timeout enforcement."""
        def _make_api_call():
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                timeout=timeout  # SDK timeout
            )

        try:
            # Enforce hard timeout at application level
            response = _call_with_timeout(_make_api_call, timeout)
        except TimeoutError as e:
            logger.error(f"[LLM Timeout] anthropic/{model} exceeded {timeout}s (hard timeout): {e}")
            raise

        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

        return {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens}
    
    def _call_openai_compatible(
        self,
        client: "OpenAI",
        provider_name: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: int
    ) -> Dict:
        """Execute OpenAI-compatible API call with hard timeout enforcement."""
        def _make_api_call():
            return client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,  # SDK timeout (may not be enforced)
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

        try:
            # Enforce hard timeout at application level
            response = _call_with_timeout(_make_api_call, timeout)
        except TimeoutError as e:
            logger.error(f"[LLM Timeout] {provider_name}/{model} exceeded {timeout}s (hard timeout): {e}")
            raise

        content = response.choices[0].message.content if response.choices else ""
        if not content:
            logger.warning(f"Empty response from {provider_name}")
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens}

    def _call_openai_compatible_with_tools(
        self,
        client: "OpenAI",
        provider_name: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Execute OpenAI-compatible API call with function calling and hard timeout enforcement."""
        # Build tool_choice - default to requiring the first tool
        if tool_choice is None and tools:
            tool_choice = {
                "type": "function",
                "function": {"name": tools[0]["function"]["name"]}
            }

        def _make_api_call():
            return client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,  # SDK timeout (may not be enforced)
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=tools,
                tool_choice=tool_choice
            )

        try:
            # Enforce hard timeout at application level
            response = _call_with_timeout(_make_api_call, timeout)
        except TimeoutError as e:
            logger.error(f"[LLM Timeout] {provider_name}/{model} exceeded {timeout}s (hard timeout, with tools): {e}")
            raise

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Extract function call result if present
        message = response.choices[0].message if response.choices else None
        if message and message.tool_calls:
            tool_call = message.tool_calls[0]
            function_call = {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments  # JSON string
            }
            # For function calling, the content is the arguments JSON
            content = tool_call.function.arguments
            logger.debug(f"[Function Call] {tool_call.function.name} returned")
            return {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens, "function_call": function_call}
        else:
            # Fallback to regular content if no tool call
            content = message.content if message else ""
            if not content:
                logger.warning(f"No tool call or content from {provider_name}")
            return {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens}

    def _call_anthropic_with_tools(
        self,
        client: "Anthropic",
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        tools: List[Dict[str, Any]],
        timeout: int
    ) -> Dict:
        """Execute Anthropic API call with function calling (tools) and hard timeout enforcement."""
        # Convert OpenAI-style tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            if "function" in tool:
                anthropic_tools.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"].get("parameters", {})
                })

        def _make_api_call():
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                tools=anthropic_tools,
                timeout=timeout  # SDK timeout
            )

        try:
            # Enforce hard timeout at application level
            response = _call_with_timeout(_make_api_call, timeout)
        except TimeoutError as e:
            logger.error(f"[LLM Timeout] anthropic/{model} exceeded {timeout}s (hard timeout, with tools): {e}")
            raise

        content = ""
        function_call = None
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

        # Parse response content blocks
        if response.content:
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == 'tool_use':
                        function_call = {
                            "name": block.name,
                            "arguments": block.input  # Already a dict for Anthropic
                        }
                        # Convert to JSON string for consistency
                        import json
                        content = json.dumps(block.input)
                        logger.debug(f"[Function Call] {block.name} returned")
                    elif block.type == 'text':
                        content = block.text

        return {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens, "function_call": function_call}
    
    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available."""
        return provider_name in self._clients

    def list_available_providers(self) -> List[str]:
        """List all available providers."""
        return list(self._clients.keys())

    def log_health_check(self) -> None:
        """Log system readiness at session start."""
        available = self.list_available_providers()
        configured = list(self.config.providers.keys())
        missing = [p for p in configured if p not in available]

        logger.info(f"[Health] Available providers: {available}")
        if missing:
            logger.warning(f"[Health] Missing providers: {missing}")

        # Log configured models for key tasks
        extraction_provider = self.config.graph_extraction_model
        generation_provider = self.config.question_generation_model

        extraction_model = "unavailable"
        generation_model = "unavailable"

        if extraction_provider in self.config.providers:
            model_config = self.config.providers[extraction_provider].models.get("graph_extraction", "unknown")
            extraction_model = model_config.name if isinstance(model_config, ModelConfig) else model_config
        if generation_provider in self.config.providers:
            model_config = self.config.providers[generation_provider].models.get("question_generation", "unknown")
            generation_model = model_config.name if isinstance(model_config, ModelConfig) else model_config

        logger.info(f"[Health] Extraction: {extraction_provider}/{extraction_model}")
        logger.info(f"[Health] Generation: {generation_provider}/{generation_model}")
