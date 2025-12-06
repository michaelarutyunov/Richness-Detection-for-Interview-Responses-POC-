"""
Core interview processing components.

This module contains the fundamental classes and functions for interview processing,
including main processors, result handlers, and core data structures.
"""

from .configurable_orchestrator import ConfigurableGraphDrivenOrchestrator
from .configurable_graph_needs_detector import ConfigurableGraphNeedsDetector
from .strategy_selector import StrategySelector

__all__ = [
    'ConfigurableGraphDrivenOrchestrator',
    'ConfigurableGraphNeedsDetector', 
    'StrategySelector'
]

__version__ = "1.0.0"
__author__ = "Interview Analysis Team"