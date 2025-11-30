"""
Core Pydantic data models for the AI Interview System.

Defines all data structures used across the application:
- Graph elements (Node, Edge)
- Interview states and deltas
- Opportunities and actions
- Logging and persistence
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Graph Elements
# ============================================================================


class Node(BaseModel):
    """Represents a node in the interview knowledge graph."""

    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Node type from schema (e.g., 'attribute', 'value')")
    label: str = Field(..., description="Human-readable label")
    creation_turn: int = Field(..., description="Turn number when node was created")
    visit_count: int = Field(default=0, description="Number of times this node was probed")
    last_visit_turn: int | None = Field(
        default=None, description="Turn number when node was last visited (for time-aware recency)"
    )
    source_quotes: list[str] = Field(
        default_factory=list, description="Participant quotes that led to this node"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")

    # Richness tracking
    total_richness: float = Field(
        default=0.0, description="Cumulative richness contributed by this node"
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Ensure label follows naming conventions."""
        if not v:
            raise ValueError("Label cannot be empty")
        # Convert to lowercase with underscores
        return v.lower().replace(" ", "_").replace("-", "_")


class Edge(BaseModel):
    """Represents an edge connecting two nodes in the graph."""

    id: str = Field(..., description="Unique edge identifier")
    type: str = Field(..., description="Edge type from schema (e.g., 'leads_to', 'blocks')")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    creation_turn: int = Field(..., description="Turn number when edge was created")
    source_quote: str = Field(..., description="Participant quote that established this connection")
    weight: float = Field(default=1.0, description="Edge weight/strength")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this relationship (0.0-1.0)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")


class GraphDelta(BaseModel):
    """Changes to the graph from a single participant response."""

    nodes_added: list[Node] = Field(
        default_factory=list, description="New nodes extracted from response"
    )
    edges_added: list[Edge] = Field(
        default_factory=list, description="New edges extracted from response"
    )
    richness_score: float = Field(default=0.0, description="Calculated richness of this response")
    extraction_metadata: dict[str, Any] = Field(
        default_factory=dict, description="LLM extraction metadata (model, latency, etc.)"
    )

    @property
    def is_empty(self) -> bool:
        """Check if delta contains no new information."""
        return len(self.nodes_added) == 0 and len(self.edges_added) == 0


# ============================================================================
# Interview Opportunities & Actions
# ============================================================================


class OpportunityAction(str, Enum):
    """Types of interview actions based on graph analysis."""

    DIG_DEEPER = "dig_deeper"
    CONNECT_CONCEPTS = "connect_concepts"
    INTRODUCE_TOPIC = "introduce_topic"
    SWITCH_TOPIC = "switch_topic"
    CLARIFY = "clarify"


class Opportunity(BaseModel):
    """A potential next step in the interview."""

    action: OpportunityAction = Field(..., description="Type of action to take")
    target_node_id: str | None = Field(
        default=None, description="Primary node to focus on (for dig_deeper, introduce_topic)"
    )
    target_node_ids: list[str] | None = Field(
        default=None, description="Multiple nodes (for connect_concepts)"
    )
    priority: float = Field(..., description="Priority score (higher = more important)")
    rationale: str = Field(..., description="Why this opportunity was identified")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional opportunity metadata"
    )


# ============================================================================
# Interview State
# ============================================================================


class InterviewPhase(str, Enum):
    """Current phase of the interview."""

    COVERAGE = "coverage"  # Exploring seed topics
    DEPTH = "depth"  # Going deeper on interesting areas
    CONNECTION = "connection"  # Linking concepts
    WRAP_UP = "wrap_up"  # Final questions


class InterviewState(BaseModel):
    """Current state of the interview."""

    session_id: str = Field(..., description="Unique session identifier")
    turn_number: int = Field(..., description="Current turn number")
    phase: InterviewPhase = Field(
        default=InterviewPhase.COVERAGE, description="Current interview phase"
    )

    # Graph metrics
    graph_node_count: int = Field(..., description="Total nodes in graph")
    graph_edge_count: int = Field(..., description="Total edges in graph")
    cumulative_richness: float = Field(..., description="Total richness score accumulated")
    coverage_pct: float = Field(..., description="Percentage of seed nodes explored (0.0-1.0)")
    avg_node_depth: float = Field(default=0.0, description="Average depth of nodes from seeds")

    # Interview progression
    top_opportunity: Any = Field(
        default=None, description="Highest priority opportunity (RankedOpportunity)"
    )
    focus_stack: list[str] = Field(
        default_factory=list,
        description="Stack of node IDs being explored (for conversation coherence)",
    )
    dead_end_nodes: list[str] = Field(
        default_factory=list, description="Node IDs marked as exhausted"
    )

    # Termination signals
    should_terminate: bool = Field(default=False, description="Whether interview should end")
    termination_reason: str | None = Field(
        default=None, description="Reason for termination if applicable"
    )

    # Timing
    started_at: datetime = Field(default_factory=datetime.now, description="When interview started")
    last_response_at: datetime | None = Field(
        default=None, description="When last response was received"
    )


# ============================================================================
# Logging & Persistence
# ============================================================================


class QuestionMethod(str, Enum):
    """How the question was generated."""

    TEMPLATE = "template"
    LLM = "llm"
    FALLBACK = "fallback"


class TurnLog(BaseModel):
    """Complete log of a single interview turn."""

    session_id: str = Field(..., description="Session identifier")
    turn_number: int = Field(..., description="Turn number")
    timestamp: datetime = Field(default_factory=datetime.now, description="When turn occurred")
    schema_version: str = Field(..., description="Schema version used")

    # Input
    participant_response: str = Field(..., description="What participant said")
    participant_response_length: int = Field(..., description="Character count of response")

    # Processing
    graph_delta: GraphDelta = Field(..., description="Extracted graph changes")
    processing_time_seconds: float = Field(default=0.0, description="Time to process response")

    # State
    interview_state: InterviewState = Field(..., description="State after processing")

    # Output
    question_generated: str = Field(..., description="Next question to ask")
    question_method: QuestionMethod = Field(..., description="How question was generated")
    question_generation_time_seconds: float = Field(
        default=0.0, description="Time to generate question"
    )
    reasoning_trace: str | None = Field(
        default=None, description="K2-thinking reasoning trace (if thinking model used)"
    )

    # Errors (if any)
    errors: list[str] = Field(
        default_factory=list, description="Any errors encountered during turn"
    )
    warnings: list[str] = Field(default_factory=list, description="Any warnings during turn")


class InterviewSummary(BaseModel):
    """Summary of completed interview for export."""

    session_id: str
    participant_id: str
    schema_used: str
    schema_version: str

    # Timing
    started_at: datetime
    completed_at: datetime
    total_duration_seconds: float

    # Conversation
    total_turns: int
    total_participant_words: int

    # Graph metrics
    final_node_count: int
    final_edge_count: int
    coverage_achieved: float
    avg_richness_per_turn: float

    # Quality indicators
    avg_response_length: float
    num_dead_ends: int
    num_validation_errors: int

    # Outputs
    transcript_path: str
    graph_export_path: str


# ============================================================================
# Schema Configuration Models
# ============================================================================


class NodeTypeConfig(BaseModel):
    """Configuration for a node type from schema."""

    name: str
    description: str
    richness_weight: float
    probing_prompt: str
    llm_extraction_prompt: str
    validation_regex: str | None = None


class EdgeTypeConfig(BaseModel):
    """Configuration for an edge type from schema."""

    name: str
    description: str
    valid_sources: list[str]
    valid_targets: list[str]
    richness_boost: float


class SeedNodeConfig(BaseModel):
    """Configuration for a seed node."""

    name: str
    type: str
    label: str
    introduction_prompt: str


class SchemaManifest(BaseModel):
    """Complete schema configuration loaded from YAML."""

    schema_version: str
    domain: str

    interview_config: dict[str, Any]
    richness_scoring: dict[str, Any]

    node_types: list[NodeTypeConfig]
    edge_types: list[EdgeTypeConfig]
    seed_nodes: list[SeedNodeConfig]

    schema_description: str | None = None
    author: str | None = None
    created_date: str | None = None


# ============================================================================
# LLM Response Models
# ============================================================================


class ExtractedNode(BaseModel):
    """Node extracted from LLM response."""

    type: str
    label: str
    quote: str


class ExtractedEdge(BaseModel):
    """Edge extracted from LLM response."""

    type: str
    source: str
    target: str
    quote: str
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (0.0-1.0)"
    )


class LLMExtractionResponse(BaseModel):
    """Structured response from LLM extraction."""

    nodes_added: list[ExtractedNode] = Field(default_factory=list)
    edges_added: list[ExtractedEdge] = Field(default_factory=list)
