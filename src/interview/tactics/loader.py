"""
Schema-driven TacticLoader - Updated to use the new v0.2 schema format.
Replaces the old file-based tactic loading with schema-integrated tactics.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.core.models import SchemaTactic
from src.core.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


class SchemaDrivenTacticLoader:
    """Loads and manages interview tactics from the unified schema."""
    
    def __init__(self, schema_loader: Optional[SchemaLoader] = None):
        """Initialize with schema loader."""
        self.schema_loader = schema_loader or SchemaLoader()
        self._tactics_cache: Dict[str, SchemaTactic] = {}
        logger.info("SchemaDrivenTacticLoader initialized")
    
    def load_tactics(self) -> List[SchemaTactic]:
        """Load all available tactics from schema."""
        logger.info("Loading tactics from schema")

        # Get tactics from schema loader - return directly without conversion
        schema_tactics = self.schema_loader.get_all_tactics()

        # Cache all tactics
        for tactic in schema_tactics:
            self._tactics_cache[tactic.id] = tactic

        logger.info(f"Loaded {len(schema_tactics)} tactics from schema")
        return schema_tactics
    
    def get_tactic_by_id(self, tactic_id: str) -> Optional[SchemaTactic]:
        """Get a specific tactic by ID."""
        if tactic_id in self._tactics_cache:
            return self._tactics_cache[tactic_id]

        # Load from schema if not cached
        schema_tactic = self.schema_loader.get_tactic(tactic_id)
        if schema_tactic:
            self._tactics_cache[tactic_id] = schema_tactic
            return schema_tactic

        return None
    
    def get_tactics_for_strategy(self, strategy_name: str) -> List[SchemaTactic]:
        """Get all tactics associated with a strategy."""
        schema_tactics = self.schema_loader.get_tactics_for_strategy(strategy_name)

        # Cache all tactics
        for tactic in schema_tactics:
            if tactic.id not in self._tactics_cache:
                self._tactics_cache[tactic.id] = tactic

        return schema_tactics

    def get_tactics_by_node_type(self, node_type: str) -> List[SchemaTactic]:
        """Get tactics that produce specific node types."""
        all_tactics = self.load_tactics()
        return [
            tactic for tactic in all_tactics
            if node_type in tactic.produces_node_types
        ]
    
    def validate_tactic_compatibility(self, tactic_id: str, interview_state: Any) -> bool:
        """Validate if a tactic is compatible with current interview state."""
        tactic = self.get_tactic_by_id(tactic_id)
        if not tactic:
            return False
        
        # Check basic constraints
        if interview_state.turn_number < tactic.min_turn:
            return False
        
        # Check usage count
        usage_count = getattr(interview_state, 'tactic_usage', {}).get(tactic_id, 0)
        if usage_count >= tactic.max_visit_count:
            return False
        

        
        return True
    
    def get_tactic_metadata(self, tactic_id: str) -> Dict[str, Any]:
        """Get detailed metadata for a tactic."""
        tactic = self.get_tactic_by_id(tactic_id)
        if not tactic:
            return {}
        
        return {
            "intent": tactic.metadata.get("intent", ""),
            "trigger": tactic.metadata.get("trigger", {}),
            "pattern": tactic.metadata.get("pattern", {}),
            "followups": tactic.metadata.get("followups", {}),
            "produces_node_types": tactic.metadata.get("produces_node_types", []),
            "valid_edge_types": tactic.metadata.get("valid_edge_types", []),
            "constraints": {
                "min_turn": tactic.min_turn,
                "max_visit_count": tactic.max_visit_count,

            }
        }