"""
Custom exceptions for LLM module.
"""


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMProviderError(LLMError):
    """Exception raised when LLM provider API fails."""

    pass


class LLMTimeoutError(LLMError):
    """Exception raised when LLM request times out."""

    pass


class LLMValidationError(LLMError):
    """Exception raised when LLM response fails validation."""

    pass


class LLMConfigError(LLMError):
    """Exception raised when LLM configuration is invalid."""

    pass
