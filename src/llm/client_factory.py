"""
Factory for creating LLM clients based on task.
"""

import logging
import os
from typing import Literal

from src.llm.anthropic_client import AnthropicClient
from src.llm.base_client import BaseLLMClient
from src.llm.config import load_model_config
from src.llm.exceptions import LLMConfigError
from src.llm.kimi_client import KimiClient

logger = logging.getLogger(__name__)

TaskType = Literal["graph_processing", "question_generation"]


class LLMClientFactory:
    """Factory for creating appropriate LLM client based on task."""

    @staticmethod
    def create_client(
        task: TaskType,
        config_path: str = "configs/model_config.yaml",
    ) -> BaseLLMClient:
        """
        Create LLM client for specified task.

        Args:
            task: Task type ("graph_processing" or "question_generation")
            config_path: Path to model configuration file

        Returns:
            BaseLLMClient: Configured client for the task

        Raises:
            LLMConfigError: Invalid configuration or missing API key
        """
        # Load configuration
        config = load_model_config(config_path)

        if task not in config.models:
            raise LLMConfigError(f"Task '{task}' not found in model configuration")

        model_config = config.models[task]

        # Get API key from environment
        api_key = os.getenv(model_config.api_key_env)
        if not api_key:
            raise LLMConfigError(
                f"API key not found: {model_config.api_key_env}. "
                f"Set environment variable for {task}."
            )

        # Get provider-specific settings
        provider_settings = config.provider_settings.get(model_config.provider)

        # Create appropriate client based on provider
        if model_config.provider == "moonshot":
            base_url = (
                provider_settings.base_url if provider_settings else "https://api.moonshot.ai/v1"
            )
            client = KimiClient(
                api_key=api_key,
                model=model_config.model,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout_seconds,
                base_url=base_url,
            )
            logger.info(
                f"Created Kimi client for {task}: {model_config.model} "
                f"(temp={model_config.temperature}, max_tokens={model_config.max_tokens})"
            )

        elif model_config.provider == "anthropic":
            client = AnthropicClient(
                api_key=api_key,
                model=model_config.model,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout_seconds,
            )
            logger.info(
                f"Created Anthropic client for {task}: {model_config.model} "
                f"(temp={model_config.temperature}, max_tokens={model_config.max_tokens})"
            )

        else:
            raise LLMConfigError(
                f"Unsupported provider: {model_config.provider}. "
                f"Supported providers: moonshot, anthropic"
            )

        return client
