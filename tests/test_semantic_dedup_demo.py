"""
Demonstration test for semantic deduplication feature.
Shows how the 7 fragmented foam nodes from the plan are handled.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decision.extraction import Extractor
from core.graph import Graph, Node
from core.schema import Schema
from core.state import CoverageState


def test_foam_example_from_plan():
    """
    Demonstrate deduplication on the 7 fragmented foam nodes from the plan.

    Input (from plan):
    1. "froths well"
    2. "proper froth"
    3. "proper foam"
    4. "foam forms correctly"
    5. "does not foam"
    6. "does not froth"
    7. "foam is too weak"

    Expected results with Phase 2A (Jaccard only):
    - "proper froth" ↔ "proper foam" → MERGED (similarity = 1.0)
    - "does not foam" ↔ "does not froth" → MERGED (similarity = 1.0)
    - Result: 7 nodes → 5 nodes (29% reduction)

    Expected results with Phase 2A + 2B (Hybrid):
    - All positive foam quality nodes → merged to 2-3 variants
    - All negative foam quality nodes → merged to 1-2 variants
    - Result: 7 nodes → 2-3 nodes (57-71% reduction)
    """
    # Config with Jaccard only (Phase 2A)
    config_jaccard = {
        "extraction": {
            "semantic_deduplication": {
                "method": "hybrid",
                "jaccard_threshold": 0.75,
                "embeddings_enabled": False,
                "embeddings_threshold": 0.80,
            }
        }
    }

    # Create extractor
    schema = Mock(spec=Schema)
    coverage_state = Mock(spec=CoverageState)
    coverage_state.reference_elements = {}
    llm_manager = Mock()

    extractor = Extractor(schema, coverage_state, llm_manager, config=config_jaccard)

    # Create graph with first node only
    graph = Graph()
    node1 = Node(label="froths well", node_type="attribute")
    graph.add_node(node1)

    print("\n" + "=" * 70)
    print("SEMANTIC DEDUPLICATION DEMO: Foam Nodes from Plan")
    print("=" * 70)
    print("\nInitial graph:")
    print(f"  1. {node1.label} (id: {node1.id})")
    print(f"\nTotal nodes: {len(graph.nodes)}")

    # Add remaining 6 nodes with deduplication (simulating sequential extraction)
    new_labels = [
        "proper froth",
        "proper foam",
        "foam forms correctly",
        "does not foam",
        "does not froth",
        "foam is too weak",
    ]

    created_nodes = []
    merged_nodes = []

    print("\n" + "-" * 70)
    print("Adding new nodes with semantic deduplication:")
    print("-" * 70)

    for label in new_labels:
        existing = extractor._find_similar_node(label, graph, "attribute")
        if existing:
            print(f"\n✓ '{label}' MERGED with '{existing.label}'")
            merged_nodes.append((label, existing.label))
        else:
            node = Node(label=label, node_type="attribute")
            graph.add_node(node)
            created_nodes.append(label)
            print(f"\n✓ '{label}' CREATED as new node")

    print("\n" + "=" * 70)
    print("RESULTS (Phase 2A: Jaccard Only)")
    print("=" * 70)
    print(f"\nOriginal nodes: 7")
    print(f"Final nodes: {len(graph.nodes)}")
    print(f"Reduction: {7 - len(graph.nodes)} nodes ({(7 - len(graph.nodes)) / 7 * 100:.1f}%)")

    print("\nMerged pairs:")
    for new_label, existing_label in merged_nodes:
        print(f"  • '{new_label}' → '{existing_label}'")

    print("\nNew nodes created:")
    for label in created_nodes:
        print(f"  • '{label}'")

    print("\n" + "=" * 70)

    # Verify expected results
    # With Phase 2A (Jaccard only):
    # - "proper foam" should merge with "proper froth" (or vice versa)
    # - "does not froth" should merge with "does not foam"
    # - Result: 7 - 2 = 5 nodes

    assert len(graph.nodes) == 5, f"Expected exactly 5 nodes with Jaccard deduplication, got {len(graph.nodes)}"
    assert len(merged_nodes) == 2, f"Expected 2 merges, got {len(merged_nodes)}"

    # Check that the expected pairs merged
    merged_labels = [label for label, _ in merged_nodes]
    assert "proper foam" in merged_labels or "proper froth" in merged_labels, "Expected proper foam/froth to merge"
    assert "does not froth" in merged_labels or "does not foam" in merged_labels, "Expected does not foam/froth to merge"

    print("\n✅ Phase 2A deduplication working as expected!")
    print("\nNote: For full semantic deduplication (Phase 2B),")
    print("install sentence-transformers and set embeddings_enabled: true")
    print("Expected result with Phase 2B: 2-3 nodes (57-71% reduction)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_foam_example_from_plan()
