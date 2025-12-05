"""
Schema-driven TacticLoader - Updated to use the new v0.2 schema format.
Replaces the old file-based tactic loading with schema-integrated tactics.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.core.models import Tactic, SchemaTactic
from src.core.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


class SchemaDrivenTacticLoader:
    """Loads and manages interview tactics from the unified schema."""
    
    def __init__(self, schema_loader: Optional[SchemaLoader] = None):
        """Initialize with schema loader."""
        self.schema_loader = schema_loader or SchemaLoader()
        self._tactics_cache: Dict[str, Tactic] = {}
        logger.info("SchemaDrivenTacticLoader initialized")
    
    def load_tactics(self) -> List[Tactic]:
        """Load all available tactics from schema."""
        logger.info("Loading tactics from schema")
        
        # Get tactics from schema loader
        schema_tactics = self.schema_loader.get_all_tactics()
        
        # Convert SchemaTactic objects to Tactic objects (legacy compatibility)
        tactics = []
        for schema_tactic in schema_tactics:
            tactic = self._convert_schema_tactic_to_tactic(schema_tactic)
            if tactic:
                tactics.append(tactic)
                self._tactics_cache[tactic.id] = tactic
        
        logger.info(f"Loaded {len(tactics)} tactics from schema")
        return tactics
    
    def _convert_schema_tactic_to_tactic(self, schema_tactic: SchemaTactic) -> Optional[Tactic]:
        """Convert SchemaTactic to legacy Tactic format."""
        try:
            # Extract patterns/templates from the schema tactic
            templates = []
            if "variants" in schema_tactic.pattern:
                templates = schema_tactic.pattern["variants"]
            elif "template" in schema_tactic.pattern:
                templates = [schema_tactic.pattern["template"]]
            
            # Map constraints to legacy format
            min_turn = schema_tactic.min_turn
            max_visit_count = schema_tactic.max_visit_count
            

            
            # Create metadata from additional schema information
            metadata = {
                "intent": schema_tactic.intent,
                "trigger": schema_tactic.trigger,
                "produces_node_types": schema_tactic.produces_node_types,
                "valid_edge_types": schema_tactic.valid_edge_types,
                "followups": schema_tactic.followups,
                "pattern": schema_tactic.pattern
            }
            
            return Tactic(
                id=schema_tactic.id,
                name=schema_tactic.id.replace("_", " ").title(),  # Convert ID to readable name
                description=schema_tactic.intent,
                min_turn=min_turn,
                max_visit_count=max_visit_count,


                templates=templates,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error converting schema tactic {schema_tactic.id}: {e}")
            return None
    

    
    def get_tactic_by_id(self, tactic_id: str) -> Optional[Tactic]:
        """Get a specific tactic by ID."""
        if tactic_id in self._tactics_cache:
            return self._tactics_cache[tactic_id]
        
        # Load from schema if not cached
        schema_tactic = self.schema_loader.get_tactic(tactic_id)
        if schema_tactic:
            tactic = self._convert_schema_tactic_to_tactic(schema_tactic)
            if tactic:
                self._tactics_cache[tactic_id] = tactic
                return tactic
        
        return None
    
    def get_tactics_for_strategy(self, strategy_name: str) -> List[Tactic]:
        """Get all tactics associated with a strategy."""
        schema_tactics = self.schema_loader.get_tactics_for_strategy(strategy_name)
        
        tactics = []
        for schema_tactic in schema_tactics:
            # Check cache first
            if schema_tactic.id in self._tactics_cache:
                tactics.append(self._tactics_cache[schema_tactic.id])
            else:
                # Convert and cache
                tactic = self._convert_schema_tactic_to_tactic(schema_tactic)
                if tactic:
                    self._tactics_cache[tactic.id] = tactic
                    tactics.append(tactic)
        
        return tactics
    
    def get_tactics_by_node_type(self, node_type: str) -> List[Tactic]:
        """Get tactics that produce specific node types."""
        all_tactics = self.load_tactics()
        return [
            tactic for tactic in all_tactics
            if node_type in tactic.metadata.get("produces_node_types", [])
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