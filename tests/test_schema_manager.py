"""
Unit tests for SchemaManager.
"""


import pytest

from src.core.schema_manager import SchemaManager


def test_load_valid_schema():
    """Test loading a valid schema file."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manifest = manager.load_schema()

    assert manifest.schema_version == "0.1.0"
    assert manifest.domain == "fmcg_means_end_chain"
    assert len(manifest.node_types) == 4
    assert len(manifest.edge_types) == 2


def test_node_type_lookup():
    """Test retrieving node type configuration."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    attr_config = manager.get_node_type("attribute")
    assert attr_config.richness_weight == 0.5
    assert "What specifically" in attr_config.probing_prompt

    func_config = manager.get_node_type("functional_consequence")
    assert func_config.richness_weight == 1.0

    value_config = manager.get_node_type("value")
    assert value_config.richness_weight == 2.0


def test_edge_type_lookup():
    """Test retrieving edge type configuration."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    leads_to_config = manager.get_edge_type("leads_to")
    assert leads_to_config.richness_boost == 1.0
    assert "attribute" in leads_to_config.valid_sources


def test_edge_validation():
    """Test edge validation against schema rules."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    # Valid edge
    assert manager.is_valid_edge("leads_to", "attribute", "functional_consequence")

    # Invalid edge (wrong direction)
    assert not manager.is_valid_edge("leads_to", "value", "attribute")

    # Invalid edge type
    assert not manager.is_valid_edge("nonexistent_type", "attribute", "value")


def test_invalid_schema_file():
    """Test handling of missing schema file."""
    manager = SchemaManager("nonexistent.yaml")
    with pytest.raises(FileNotFoundError):
        manager.load_schema()


def test_richness_weights():
    """Test richness weight retrieval."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    assert manager.get_richness_weight("attribute") == 0.5
    assert manager.get_richness_weight("functional_consequence") == 1.0
    assert manager.get_richness_weight("psychosocial_consequence") == 1.5
    assert manager.get_richness_weight("value") == 2.0
    assert manager.get_richness_boost("leads_to") == 1.0


def test_probing_prompt_template():
    """Test probing prompt template with node substitution."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    prompt = manager.get_probing_prompt("attribute", "affordable_price")
    assert "affordable_price" in prompt
    assert "{node}" not in prompt


def test_llm_extraction_prompt():
    """Test LLM extraction prompt retrieval."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    prompt = manager.get_llm_extraction_prompt("attribute")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_interview_config():
    """Test interview configuration retrieval."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    config = manager.get_interview_config()
    assert "max_turns" in config
    assert config["max_turns"] == 20


def test_seed_nodes():
    """Test seed nodes retrieval."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    seeds = manager.get_seed_nodes()
    assert len(seeds) > 0
    assert all(hasattr(seed, "name") for seed in seeds)
    assert all(hasattr(seed, "type") for seed in seeds)


def test_schema_validation_success():
    """Test schema validation on valid schema."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    assert manager.validate_schema() is True


def test_schema_properties():
    """Test schema property accessors."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    assert manager.schema_version == "0.1.0"
    assert manager.domain == "fmcg_means_end_chain"
    assert len(manager.node_types) == 4
    assert len(manager.edge_types) == 2


def test_get_unknown_node_type():
    """Test error on unknown node type."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    with pytest.raises(KeyError):
        manager.get_node_type("nonexistent_type")


def test_get_unknown_edge_type():
    """Test error on unknown edge type."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")
    manager.load_schema()

    with pytest.raises(KeyError):
        manager.get_edge_type("nonexistent_type")


def test_validate_before_load():
    """Test validation fails if schema not loaded."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")

    with pytest.raises(ValueError, match="not loaded"):
        manager.validate_schema()


def test_get_interview_config_before_load():
    """Test config retrieval fails if schema not loaded."""
    manager = SchemaManager("schemas/means_end_chain_v0.1.yaml")

    with pytest.raises(ValueError, match="not loaded"):
        manager.get_interview_config()
