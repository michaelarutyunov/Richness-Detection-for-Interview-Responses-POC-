"""
Configuration loader for LLM models.

Loads and validates model_config.yaml using Pydantic models.
"""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.llm.exceptions import LLMConfigError

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    provider: str
    model: str
    api_key_env: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    supports_function_calling: bool = True
    supports_reasoning: bool = False  # NEW: For thinking models like K2-thinking


class RetryConfig(BaseModel):
    """Retry configuration for LLM requests."""

    max_retries: int = Field(ge=0)
    initial_delay_seconds: float = Field(gt=0)
    backoff_multiplier: float = Field(gt=1.0)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)


class ProviderSettings(BaseModel):
    """Provider-specific settings."""

    base_url: str | None = None
    request_timeout: int | None = None
    max_tokens_to_sample: int | None = None
    organization: str | None = None


class LLMConfig(BaseModel):
    """Complete LLM configuration."""

    models: dict[str, ModelConfig]
    fallbacks: dict[str, list[ModelConfig]] = Field(default_factory=dict)
    retry_config: RetryConfig
    rate_limits: RateLimitConfig
    provider_settings: dict[str, ProviderSettings] = Field(default_factory=dict)


def load_model_config(config_path: str = "configs/model_config.yaml") -> LLMConfig:
    """
    Load and validate model configuration from YAML file.

    Args:
        config_path: Path to model_config.yaml

    Returns:
        LLMConfig: Validated configuration

    Raises:
        LLMConfigError: Failed to load or validate configuration
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise LLMConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Validate with Pydantic
        config = LLMConfig(**config_data)

        logger.info(
            f"Loaded LLM config with {len(config.models)} model configurations: "
            f"{', '.join(config.models.keys())}"
        )

        return config

    except yaml.YAMLError as e:
        raise LLMConfigError(f"Failed to parse YAML: {e}") from e
    except Exception as e:
        raise LLMConfigError(f"Failed to load config from {config_path}: {e}") from e


def get_model_config(task: str, config_path: str = "configs/model_config.yaml") -> ModelConfig:
    """
    Get model configuration for a specific task.

    Args:
        task: Task name ("graph_processing" or "question_generation")
        config_path: Path to model_config.yaml

    Returns:
        ModelConfig: Configuration for the task

    Raises:
        LLMConfigError: Task not found in configuration
    """
    config = load_model_config(config_path)

    if task not in config.models:
        raise LLMConfigError(f"Task '{task}' not found in model configuration")

    return config.models[task]
