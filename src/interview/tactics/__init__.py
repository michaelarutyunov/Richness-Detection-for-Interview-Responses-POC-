"""
Interview tactics and strategies.

This module contains components for implementing different interview tactics,
including questioning strategies, response analysis approaches, and evaluation methods.
"""

from .loader import SchemaDrivenTacticLoader
from .configurable_question_generator import ConfigurableQuestionGenerator
from .selector import SchemaDrivenTacticSelector

__all__ = [
    'SchemaDrivenTacticLoader',
    'ConfigurableQuestionGenerator',
    'SchemaDrivenTacticSelector'
]

__version__ = "1.0.0"
__author__ = "Interview Analysis Team"