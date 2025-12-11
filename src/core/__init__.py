"""
Core data structures for interview agent.
"""

from core.graph import Graph, Node, Edge
from core.schema import Schema, NodeTypeDefinition, EdgeTypeDefinition
from core.history import History, Turn
from core.state import (
    GraphState,
    CoverageState,
    CoverageGap,
    CoverageRequirements,
    ReferenceElement,
    Momentum
)

__all__ = [
    # Graph
    "Graph",
    "Node", 
    "Edge",
    # Schema
    "Schema",
    "NodeTypeDefinition",
    "EdgeTypeDefinition",
    # History
    "History",
    "Turn",
    # State
    "GraphState",
    "CoverageState",
    "CoverageGap",
    "CoverageRequirements",
    "ReferenceElement",
    "Momentum",
]
