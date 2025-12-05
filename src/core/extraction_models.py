"""
Lightweight extraction models for concept extraction pipeline.
Streamlined version of legacy models for improved performance.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import re


class ExtractedNode(BaseModel):
    """Represents a concept node extracted from interview response."""
    type: str = Field(description="Node type from schema (e.g., 'attribute', 'value')")
    label: str = Field(description="Descriptive label (lowercase_with_underscores)")
    quote: str = Field(description="Exact quote from response supporting this node")

    @field_validator('label')
    @classmethod
    def validate_label_format(cls, v: str) -> str:
        """Validate label follows lowercase_with_underscores format."""
        if not v:
            raise ValueError("Label cannot be empty")
        if not re.match(r'^[a-z_][a-z0-9_]{2,39}$', v):
            raise ValueError(
                f"Label must be lowercase_with_underscores (3-40 chars, alphanumeric + underscore): '{v}'"
            )
        return v


class ExtractedEdge(BaseModel):
    """Represents a relationship between concepts extracted from interview response."""
    type: str = Field(description="Edge type from schema (e.g., 'leads_to', 'blocks')")
    source: str = Field(description="Source node label")
    target: str = Field(description="Target node label") 
    quote: str = Field(description="Quote from response establishing this relationship")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score based on causal language")


class GraphDelta(BaseModel):
    """Represents changes to be applied to the knowledge graph."""
    nodes_added: List[ExtractedNode] = Field(default_factory=list)
    edges_added: List[ExtractedEdge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extraction metadata")
    
    def is_empty(self) -> bool:
        """Check if delta contains no changes."""
        return len(self.nodes_added) == 0 and len(self.edges_added) == 0
    
    def get_summary(self) -> str:
        """Get human-readable summary of changes."""
        return f"Added {len(self.nodes_added)} nodes, {len(self.edges_added)} edges"


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    model_used: str = Field(default="unknown")
    latency_ms: int = Field(default=0)
    tokens_used: int = Field(default=0)
    validation_errors: List[str] = Field(default_factory=list)
    extraction_timestamp: datetime = Field(default_factory=datetime.now)


class ExtractionResult(BaseModel):
    """Complete extraction result with delta and metadata."""
    delta: GraphDelta
    metadata: ExtractionMetadata
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)