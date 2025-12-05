"""
Unified Schema Loader for handling the new v0.2 schema format.
Loads and manages schemas that contain node types, edge types, strategies, and tactics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NodeType(BaseModel):
    """Definition of a node type from schema."""
    name: str
    description: str


class EdgeType(BaseModel):
    """Definition of an edge type from schema."""
    name: str
    description: str
    valid_sources: List[str]
    valid_targets: List[str]


class TacticDefinition(BaseModel):
    """Complete tactic definition from schema."""
    id: str
    intent: str
    trigger: Dict[str, Any]
    pattern: Dict[str, Any]
    followups: Dict[str, str]
    produces_node_types: List[str]
    valid_edge_types: List[str]
    constraints: Dict[str, Any]
    
    @property
    def min_turn(self) -> int:
        """Get minimum turn constraint."""
        return self.constraints.get("min_turn", 0)
    
    @property
    def max_visit_count(self) -> int:
        """Get maximum visit count constraint."""
        return self.constraints.get("max_visit_count", 10)


class StrategyDefinition(BaseModel):
    """Strategy definition with associated tactics."""
    name: str
    description: str
    tactics: List[str]  # List of tactic IDs


class SchemaConfig(BaseModel):
    """Complete schema configuration."""
    schema_version: str
    domain: str
    node_types: List[NodeType]
    edge_types: List[EdgeType]
    strategies: Dict[str, StrategyDefinition]
    tactics: Dict[str, TacticDefinition]
    seed_nodes: Optional[List[Dict[str, Any]]] = None
    schema_description: str = ""
    author: str = ""
    created_date: str = ""


class SchemaLoader:
    """Loads and manages interview schemas with integrated tactics and strategies."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize with schema path."""
        self.schema_path = schema_path or Path("schemas/means_end_chain_v0.2.yaml")
        self._schema_config: Optional[SchemaConfig] = None
        self._tactics_cache: Dict[str, TacticDefinition] = {}
        self._strategies_cache: Dict[str, StrategyDefinition] = {}
        logger.info(f"SchemaLoader initialized with schema: {self.schema_path}")
    
    def load_schema(self) -> SchemaConfig:
        """Load and parse the schema file."""
        if self._schema_config is not None:
            return self._schema_config
        
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                raw_schema = yaml.safe_load(f)
            
            # Parse node types
            node_types = [
                NodeType(name=nt["name"], description=nt["description"])
                for nt in raw_schema.get("node_types", [])
            ]
            
            # Parse edge types
            edge_types = [
                EdgeType(
                    name=et["name"],
                    description=et["description"],
                    valid_sources=et.get("valid_sources", []),
                    valid_targets=et.get("valid_targets", [])
                )
                for et in raw_schema.get("edge_types", [])
            ]
            
            # Parse tactics
            tactics = {}
            for tactic_id, tactic_data in raw_schema.get("tactics", {}).items():
                tactics[tactic_id] = TacticDefinition(
                    id=tactic_id,
                    intent=tactic_data.get("intent", ""),
                    trigger=tactic_data.get("trigger", {}),
                    pattern=tactic_data.get("pattern", {}),
                    followups=tactic_data.get("followups", {}),
                    produces_node_types=tactic_data.get("produces_node_types", []),
                    valid_edge_types=tactic_data.get("valid_edge_types", []),
                    constraints=tactic_data.get("constraints", {})
                )
            
            # Parse strategies
            strategies = {}
            for strategy_name, strategy_data in raw_schema.get("strategies", {}).items():
                strategies[strategy_name] = StrategyDefinition(
                    name=strategy_name,
                    description=strategy_data.get("description", ""),
                    tactics=strategy_data.get("tactics", [])
                )
            
            self._schema_config = SchemaConfig(
                schema_version=raw_schema.get("schema_version", "1.0.0"),
                domain=raw_schema.get("domain", "unknown"),
                node_types=node_types,
                edge_types=edge_types,
                strategies=strategies,
                tactics=tactics,
                seed_nodes=raw_schema.get("seed_nodes"),
                schema_description=raw_schema.get("schema_description", ""),
                author=raw_schema.get("author", ""),
                created_date=raw_schema.get("created_date", "")
            )
            
            # Cache tactics and strategies for quick access
            self._tactics_cache = tactics
            self._strategies_cache = strategies
            
            logger.info(f"Successfully loaded schema v{self._schema_config.schema_version} "
                       f"with {len(node_types)} node types, {len(edge_types)} edge types, "
                       f"{len(tactics)} tactics, and {len(strategies)} strategies")
            
            return self._schema_config
            
        except Exception as e:
            logger.error(f"Error loading schema from {self.schema_path}: {e}")
            raise
    
    def get_node_types(self) -> List[NodeType]:
        """Get all node types from schema."""
        schema = self.load_schema()
        return schema.node_types
    
    def get_edge_types(self) -> List[EdgeType]:
        """Get all edge types from schema."""
        schema = self.load_schema()
        return schema.edge_types
    
    def get_tactic(self, tactic_id: str) -> Optional[TacticDefinition]:
        """Get a specific tactic by ID."""
        if not self._tactics_cache:
            self.load_schema()
        return self._tactics_cache.get(tactic_id)
    
    def get_all_tactics(self) -> List[TacticDefinition]:
        """Get all available tactics."""
        if not self._tactics_cache:
            self.load_schema()
        return list(self._tactics_cache.values())
    
    def get_strategy(self, strategy_name: str) -> Optional[StrategyDefinition]:
        """Get a specific strategy by name."""
        if not self._strategies_cache:
            self.load_schema()
        return self._strategies_cache.get(strategy_name)
    
    def get_all_strategies(self) -> List[StrategyDefinition]:
        """Get all available strategies."""
        if not self._strategies_cache:
            self.load_schema()
        return list(self._strategies_cache.values())
    
    def get_tactics_for_strategy(self, strategy_name: str) -> List[TacticDefinition]:
        """Get all tactics associated with a strategy."""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return []
        
        tactics = []
        for tactic_id in strategy.tactics:
            tactic = self.get_tactic(tactic_id)
            if tactic:
                tactics.append(tactic)
        
        return tactics
    
    def validate_node_type(self, node_type: str) -> bool:
        """Validate that a node type is defined in the schema."""
        valid_types = {nt.name for nt in self.get_node_types()}
        return node_type in valid_types
    
    def validate_edge_type(self, edge_type: str, source_type: str, target_type: str) -> bool:
        """Validate that an edge type is valid for the given source and target node types."""
        for et in self.get_edge_types():
            if et.name == edge_type:
                return (source_type in et.valid_sources and 
                       target_type in et.valid_targets)
        return False
    
    def get_node_type_description(self, node_type: str) -> str:
        """Get the description for a node type."""
        for nt in self.get_node_types():
            if nt.name == node_type:
                return nt.description
        return ""
    
    def get_edge_type_description(self, edge_type: str) -> str:
        """Get the description for an edge type."""
        for et in self.get_edge_types():
            if et.name == edge_type:
                return et.description
        return ""
    
    def format_node_types_for_prompt(self) -> str:
        """Format node types for inclusion in prompts."""
        lines = []
        for nt in self.get_node_types():
            lines.append(f"- {nt.name}: {nt.description}")
        return "\n".join(lines) if lines else "No node types defined"
    
    def format_edge_types_for_prompt(self) -> str:
        """Format edge types for inclusion in prompts."""
        lines = []
        for et in self.get_edge_types():
            lines.append(f"- {et.name}: {et.description}")
        return "\n".join(lines) if lines else "No edge types defined"