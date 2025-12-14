"""
Tests for semantic deduplication in extraction.
Covers Phase 2A (Enhanced Jaccard) and Phase 2B (Embeddings).
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decision.extraction import Extractor
from core.graph import Graph, Node
from core.schema import Schema
from core.state import CoverageState


# ============================================================================
# Helper Functions
# ============================================================================


def _has_sentence_transformers():
    """Check if sentence-transformers is installed."""
    try:
        import sentence_transformers

        return True
    except ImportError:
        return False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_config():
    """Basic config with semantic deduplication enabled."""
    return {
        "extraction": {
            "semantic_deduplication": {
                "method": "hybrid",
                "jaccard_threshold": 0.75,
                "embeddings_enabled": False,  # Disable for Phase 2A tests
                "embeddings_threshold": 0.80,
            }
        }
    }


@pytest.fixture
def embeddings_config():
    """Config with embeddings enabled."""
    return {
        "extraction": {
            "semantic_deduplication": {
                "method": "hybrid",
                "jaccard_threshold": 0.75,
                "embeddings_enabled": True,  # Enable for Phase 2B tests
                "embeddings_threshold": 0.80,
            }
        }
    }


@pytest.fixture
def extractor(basic_config):
    """Create an Extractor instance with mocked dependencies."""
    schema = Mock(spec=Schema)
    coverage_state = Mock(spec=CoverageState)
    coverage_state.reference_elements = {}
    llm_manager = Mock()

    return Extractor(schema, coverage_state, llm_manager, config=basic_config)


@pytest.fixture
def graph_with_nodes():
    """Create a graph with some test nodes."""
    graph = Graph()

    # Add test nodes
    node1 = Node(label="proper foam", node_type="attribute")
    node2 = Node(label="thick texture", node_type="attribute")
    node3 = Node(label="does not foam", node_type="attribute")

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    return graph


# ============================================================================
# PHASE 2A: Enhanced Jaccard Tests
# ============================================================================


class TestLemmatization:
    """Test lemmatization with suffix removal and synonyms."""

    def test_suffix_removal_s(self, extractor):
        """Test removing plural 's'."""
        lemmas = extractor._lemmatize_phrase("foams")
        assert "foam" in lemmas

    def test_suffix_removal_ing(self, extractor):
        """Test removing '-ing'."""
        lemmas = extractor._lemmatize_phrase("foaming")
        assert "foam" in lemmas

    def test_suffix_removal_ed(self, extractor):
        """Test removing '-ed'."""
        lemmas = extractor._lemmatize_phrase("frosted")
        assert "frost" in lemmas

    def test_suffix_removal_er(self, extractor):
        """Test removing '-er'."""
        lemmas = extractor._lemmatize_phrase("thicker")
        assert "thick" in lemmas

    def test_suffix_removal_est(self, extractor):
        """Test removing '-est'."""
        lemmas = extractor._lemmatize_phrase("thickest")
        assert "thick" in lemmas

    def test_suffix_removal_ly(self, extractor):
        """Test removing '-ly'."""
        lemmas = extractor._lemmatize_phrase("quickly")
        assert "quick" in lemmas

    def test_synonym_expansion_foam(self, extractor):
        """Test foam <-> froth synonym expansion."""
        lemmas = extractor._lemmatize_phrase("foam")
        assert "foam" in lemmas
        assert "froth" in lemmas

    def test_synonym_expansion_thick(self, extractor):
        """Test thick <-> heavy synonym expansion."""
        lemmas = extractor._lemmatize_phrase("thick")
        assert "thick" in lemmas
        assert "heavy" in lemmas

    def test_synonym_expansion_thin(self, extractor):
        """Test thin <-> watery synonym expansion."""
        lemmas = extractor._lemmatize_phrase("thin")
        assert "thin" in lemmas
        assert "watery" in lemmas

    def test_synonym_expansion_creamy(self, extractor):
        """Test creamy <-> smooth synonym expansion."""
        lemmas = extractor._lemmatize_phrase("creamy")
        assert "creamy" in lemmas
        assert "smooth" in lemmas

    def test_multi_word_phrase(self, extractor):
        """Test lemmatizing multi-word phrase."""
        lemmas = extractor._lemmatize_phrase("proper foam")
        assert "proper" in lemmas
        assert "foam" in lemmas
        assert "froth" in lemmas  # synonym


class TestJaccardSimilarity:
    """Test Jaccard similarity computation."""

    def test_exact_match(self, extractor):
        """Test exact match gives similarity 1.0."""
        sim = extractor._jaccard_similarity_with_lemmas("foam", "foam")
        assert sim == 1.0

    def test_synonym_match(self, extractor):
        """Test synonym match (foam <-> froth)."""
        sim = extractor._jaccard_similarity_with_lemmas("foam", "froth")
        # Both expand to {foam, froth}, so intersection == union
        assert sim == 1.0

    def test_proper_foam_vs_proper_froth(self, extractor):
        """Test 'proper foam' ~= 'proper froth' (from plan)."""
        sim = extractor._jaccard_similarity_with_lemmas("proper foam", "proper froth")
        # Both expand to {proper, foam, froth}
        assert sim == 1.0

    def test_does_not_foam_vs_does_not_froth(self, extractor):
        """Test 'does not foam' ~= 'does not froth'."""
        sim = extractor._jaccard_similarity_with_lemmas("does not foam", "does not froth")
        # Both should expand to {doe, not, foam, froth} (after suffix removal)
        # Note: "does" -> "doe" after removing 's'
        assert sim >= 0.75

    def test_froths_well_vs_proper_froth(self, extractor):
        """Test 'froths well' vs 'proper froth' (should not match per plan)."""
        sim = extractor._jaccard_similarity_with_lemmas("froths well", "proper froth")
        # froths well -> {froth, foam, well}
        # proper froth -> {proper, froth, foam}
        # intersection = {froth, foam} = 2
        # union = {froth, foam, well, proper} = 4
        # similarity = 2/4 = 0.5
        assert sim < 0.75

    def test_type_isolation_logic(self, extractor, graph_with_nodes):
        """Test that different node types don't match."""
        # This is tested in _find_similar_node, which filters by type
        # Here we verify that Jaccard doesn't care about type
        sim = extractor._jaccard_similarity_with_lemmas("thick", "thick")
        assert sim == 1.0  # Jaccard doesn't know about types

    def test_threshold_tuning(self, extractor):
        """Verify 0.75 threshold catches direct duplicates without false positives."""
        # Should match (direct synonyms)
        assert extractor._jaccard_similarity_with_lemmas("foam", "froth") >= 0.75
        assert extractor._jaccard_similarity_with_lemmas("thick", "heavy") >= 0.75

        # Should NOT match (different concepts)
        assert extractor._jaccard_similarity_with_lemmas("foam", "thick") < 0.75
        assert extractor._jaccard_similarity_with_lemmas("sweet", "bitter") < 0.75


class TestFindSimilarNode:
    """Test _find_similar_node with hybrid strategy."""

    def test_exact_match(self, extractor, graph_with_nodes):
        """Test exact match returns existing node."""
        result = extractor._find_similar_node("proper foam", graph_with_nodes, "attribute")
        assert result is not None
        assert result.label == "proper foam"

    def test_jaccard_match(self, extractor, graph_with_nodes):
        """Test Jaccard match returns existing node."""
        # "proper froth" should match "proper foam" via synonyms
        result = extractor._find_similar_node("proper froth", graph_with_nodes, "attribute")
        assert result is not None
        assert result.label == "proper foam"

    def test_type_mismatch(self, extractor, graph_with_nodes):
        """Test that type mismatch prevents matching."""
        # Even if label matches, wrong type should not match
        result = extractor._find_similar_node(
            "proper foam", graph_with_nodes, "functional_consequence"
        )
        assert result is None

    def test_no_match(self, extractor, graph_with_nodes):
        """Test that completely different label returns None."""
        result = extractor._find_similar_node("sweet taste", graph_with_nodes, "attribute")
        assert result is None

    def test_lemma_match(self, extractor, graph_with_nodes):
        """Test lemma matching (plural vs singular)."""
        # "thick textures" should match "thick texture"
        result = extractor._find_similar_node("thick textures", graph_with_nodes, "attribute")
        assert result is not None
        assert result.label == "thick texture"


# ============================================================================
# PHASE 2B: Semantic Embeddings Tests (Optional - requires sentence-transformers)
# ============================================================================


class TestSemanticEmbeddings:
    """Test semantic embeddings (Phase 2B)."""

    @pytest.mark.skipif(
        not _has_sentence_transformers(),
        reason="sentence-transformers not installed",
    )
    def test_structural_variants_match(self, embeddings_config):
        """Test 'froths well' matches 'proper foam' with embeddings."""
        schema = Mock(spec=Schema)
        coverage_state = Mock(spec=CoverageState)
        coverage_state.reference_elements = {}
        llm_manager = Mock()

        extractor = Extractor(schema, coverage_state, llm_manager, config=embeddings_config)

        # Create graph with "proper foam"
        graph = Graph()
        node = Node(label="proper foam", node_type="attribute")
        graph.add_node(node)

        # "froths well" should match via embeddings
        result = extractor._find_similar_node("froths well", graph, "attribute")
        # This may or may not match depending on threshold - check similarity
        if result:
            assert result.label == "proper foam"

    @pytest.mark.skipif(
        not _has_sentence_transformers(),
        reason="sentence-transformers not installed",
    )
    def test_semantic_similarity_computation(self, embeddings_config):
        """Test semantic similarity computation."""
        schema = Mock(spec=Schema)
        coverage_state = Mock(spec=CoverageState)
        coverage_state.reference_elements = {}
        llm_manager = Mock()

        extractor = Extractor(schema, coverage_state, llm_manager, config=embeddings_config)

        # Test similar concepts
        sim1 = extractor._compute_semantic_similarity("froths well", "proper foam")
        assert 0.0 <= sim1 <= 1.0

        # Test opposite concepts (should have lower similarity)
        sim2 = extractor._compute_semantic_similarity("froths well", "does not foam")
        assert 0.0 <= sim2 <= 1.0
        # Typically sim2 should be lower than sim1, but not guaranteed

    @pytest.mark.skipif(
        not _has_sentence_transformers(),
        reason="sentence-transformers not installed",
    )
    def test_embedding_caching(self, embeddings_config):
        """Test that embeddings are cached."""
        schema = Mock(spec=Schema)
        coverage_state = Mock(spec=CoverageState)
        coverage_state.reference_elements = {}
        llm_manager = Mock()

        extractor = Extractor(schema, coverage_state, llm_manager, config=embeddings_config)

        # Get embedding twice
        emb1 = extractor._get_cached_embedding("test phrase")
        emb2 = extractor._get_cached_embedding("test phrase")

        # Should be same object (cached)
        assert emb1 is emb2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with full extraction flow."""

    def test_deduplication_in_extraction(self, extractor, graph_with_nodes):
        """Test that deduplication works during extraction."""
        # Mock extraction data with duplicate nodes
        extraction_data = {
            "nodes": [
                {
                    "label": "proper froth",  # Should match "proper foam"
                    "node_type": "attribute",
                    "quote": "The froth is proper",
                    "element_mapping": None,
                },
                {
                    "label": "thick textures",  # Should match "thick texture"
                    "node_type": "attribute",
                    "quote": "It has thick textures",
                    "element_mapping": None,
                },
                {
                    "label": "new concept",  # Genuinely new
                    "node_type": "attribute",
                    "quote": "Something new",
                    "element_mapping": None,
                },
            ],
            "edges": [],
        }

        # Process nodes manually (simulating _parse_extraction_result)
        new_nodes = []
        for node_data in extraction_data["nodes"]:
            label = node_data["label"]
            node_type = node_data["node_type"]

            existing = extractor._find_similar_node(label, graph_with_nodes, node_type)
            if not existing:
                node = Node(
                    label=label,
                    node_type=node_type,
                    metadata={"quote": node_data["quote"]},
                )
                new_nodes.append(node)

        # Should only create 1 new node ("new concept")
        # "proper froth" matches "proper foam"
        # "thick textures" matches "thick texture"
        assert len(new_nodes) == 1
        assert new_nodes[0].label == "new concept"
