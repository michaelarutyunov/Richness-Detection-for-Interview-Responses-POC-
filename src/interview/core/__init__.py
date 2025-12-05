"""
Core interview processing components.

This module contains the fundamental classes and functions for interview processing,
including main processors, result handlers, and core data structures.
"""

from .graph_driven_orchestrator import GraphDrivenOrchestrator
from .graph_needs_detector import GraphNeedsDetector
from .strategy_selector import StrategySelector

__all__ = [
    'GraphDrivenOrchestrator',
    'GraphNeedsDetector', 
    'StrategySelector'
]

__version__ = "1.0.0"
__author__ = "Interview Analysis Team"