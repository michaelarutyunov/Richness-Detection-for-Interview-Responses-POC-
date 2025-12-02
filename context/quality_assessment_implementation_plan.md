# Multi-Phase Quality Assessment Implementation Plan

## Executive Summary

This plan implements a multi-dimensional quality assessment system to replace the current single-value richness metric. The implementation is divided into three phases:

- **Phase 1 (Immediate):** Basic quality metrics using graph topology only
- **Phase 2 (Core Features):** Dynamic progress tracking for real-time interview guidance
- **Phase 3 (Optional):** Advanced features including semantic coherence

Each phase builds on the previous one, allowing for incremental validation and rollback if needed.

---

## Current Problem Statement

**Critical Issue:** The current richness metric (`sum(node_weights) + sum(edge_boosts)`) measures interview quantity rather than quality.

**Evidence:**
- Interview with 23 nodes and 13 edges (richness 27.1) scores similarly to interview with 17 nodes and 14 edges (richness 28.0)
- Second interview scored higher despite not reaching any terminal values
- No penalty for disconnected nodes, dead-end questions, or structural issues
- Cannot distinguish participant verbosity from interviewer-guided depth

**Impact:**
- System cannot detect when interviews plateau or go off-track
- No guidance for probing strategy (breadth vs depth)
- Quality assessment happens only after interview ends
- Thresholds (0.5, 2.0, 5.0) are meaningless compared to actual scores (25-30)

---

# PHASE 1: Replace Current Richness Metric

## Overview

**Goal:** Fix critical issues with existing metric using only graph topology analysis

**Priority:** IMMEDIATE - Addresses fundamental measurement problems

**Dependencies:** None (uses existing networkx graph structure)

**Expected Duration:** 2-3 days implementation + 1 day validation

## What We're Building

Three core quality metrics that replace the single richness score:

1. **Depth Metrics** - How deep do attribute→value chains go?
2. **Chain Completion** - What percentage of attributes connect to values?
3. **Connectivity Analysis** - Is the graph fragmented or coherent?

Plus a composite quality score combining all three dimensions.

---

## New Module Structure

Create new quality assessment module:

```
src/quality_assessment/
├── __init__.py
├── established_metrics.py    # Phase 1 metrics
├── config.py                  # Thresholds and parameters
└── README.md                  # Documentation
```

---

## Implementation Details

### 1. Create Module Foundation

**File:** `src/quality_assessment/__init__.py`

```python
"""
Quality assessment system for interview analysis.

Replaces single richness metric with multi-dimensional quality assessment.
"""

from .established_metrics import (
    calculate_depth_metrics,
    calculate_chain_completion,
    analyze_connectivity,
    calculate_level_balance,
    validate_dag_structure,
    calculate_composite_quality_score
)

from .config import QUALITY_THRESHOLDS

__all__ = [
    'calculate_depth_metrics',
    'calculate_chain_completion',
    'analyze_connectivity',
    'calculate_level_balance',
    'validate_dag_structure',
    'calculate_composite_quality_score',
    'QUALITY_THRESHOLDS'
]
```

**File:** `src/quality_assessment/config.py`

```python
"""Configuration for quality thresholds and parameters."""

QUALITY_THRESHOLDS = {
    'depth': {
        'poor': 0,
        'fair': 2,
        'good': 3,
        'excellent': 4
    },
    'completion': {
        'poor': 0.0,
        'fair': 0.3,
        'good': 0.6,
        'excellent': 0.8
    },
    'connectivity': {
        'poor': 0.4,
        'fair': 0.6,
        'good': 0.75,
        'excellent': 0.85
    }
}

# Weight configuration for composite score
COMPOSITE_WEIGHTS = {
    'depth': 0.3,        # 30% - Depth is critical for laddering
    'completion': 0.4,   # 40% - Primary goal is complete chains
    'connectivity': 0.2, # 20% - Graph should be coherent
    'has_values': 0.1    # 10% - Must reach terminal values
}

# Target values for composite score calculation
DEPTH_TARGET = 4.0        # Ideal: attribute → func → psycho → value
COMPLETION_TARGET = 1.0   # Ideal: 100% attributes connect to values
CONNECTIVITY_TARGET = 1.0 # Ideal: All nodes in one component
```

---

### 2. Implement Core Metrics

**File:** `src/quality_assessment/established_metrics.py`

This file contains all Phase 1 metric implementations:

#### Function 1: calculate_depth_metrics()

```python
from typing import Dict, List, Optional
from src.core.interview_graph import InterviewGraph
import logging

logger = logging.getLogger(__name__)

def calculate_depth_metrics(graph: InterviewGraph) -> Dict:
    """
    Calculate path depth statistics for means-end chains.

    Analyzes shortest paths from attributes to values, which represents
    the core laddering structure we're trying to extract.

    Args:
        graph: InterviewGraph instance

    Returns:
        dict with:
            - max_depth: Longest attribute→value chain (edges)
            - avg_depth: Average depth across all chains
            - has_complete_chains: Whether any attribute→value paths exist
            - chain_count: Number of distinct attribute→value paths
            - connected_attributes: List of attribute IDs with value connections
    """
    attributes = graph.get_nodes_by_type('attribute')
    values = graph.get_nodes_by_type('value')

    if not values:
        return {
            'max_depth': 0,
            'avg_depth': 0.0,
            'has_complete_chains': False,
            'chain_count': 0,
            'connected_attributes': []
        }

    path_lengths = []
    connected_attr_ids = set()

    for attr in attributes:
        for value in values:
            paths = graph.find_all_paths(attr.id, value.id)
            if paths:
                shortest = min(len(p) - 1 for p in paths)  # -1 for edge count
                path_lengths.append(shortest)
                connected_attr_ids.add(attr.id)

    if not path_lengths:
        return {
            'max_depth': 0,
            'avg_depth': 0.0,
            'has_complete_chains': False,
            'chain_count': 0,
            'connected_attributes': []
        }

    return {
        'max_depth': max(path_lengths),
        'avg_depth': sum(path_lengths) / len(path_lengths),
        'has_complete_chains': True,
        'chain_count': len(path_lengths),
        'connected_attributes': list(connected_attr_ids)
    }
```

#### Function 2: calculate_chain_completion()

```python
def calculate_chain_completion(graph: InterviewGraph) -> Dict:
    """
    Calculate what percentage of attributes successfully ladder to values.

    This is a goal-oriented metric: the purpose of laddering interviews
    is to connect concrete attributes to abstract values.

    Args:
        graph: InterviewGraph instance

    Returns:
        dict with:
            - completion_index: Ratio of connected attributes (0.0-1.0)
            - connected_count: Number of attributes with value connections
            - total_attributes: Total attribute nodes
            - incomplete_attributes: List of attribute IDs without value paths
            - avg_depth: Average depth from all attributes (not just connected)
            - status: Interpretation string
    """
    attributes = graph.get_nodes_by_type('attribute')
    values = graph.get_nodes_by_type('value')

    if not attributes:
        return {
            'completion_index': 1.0,
            'connected_count': 0,
            'total_attributes': 0,
            'incomplete_attributes': [],
            'avg_depth': 0.0,
            'status': 'no_attributes'
        }

    if not values:
        return {
            'completion_index': 0.0,
            'connected_count': 0,
            'total_attributes': len(attributes),
            'incomplete_attributes': [a.id for a in attributes],
            'avg_depth': 0.0,
            'status': 'no_values_reached'
        }

    connected_attributes = 0
    incomplete_attributes = []
    total_depth = 0

    for attr in attributes:
        has_path_to_value = False
        max_attr_depth = 0

        for value in values:
            paths = graph.find_all_paths(attr.id, value.id)
            if paths:
                has_path_to_value = True
                max_attr_depth = max(max_attr_depth, max(len(p) - 1 for p in paths))

        if has_path_to_value:
            connected_attributes += 1
        else:
            incomplete_attributes.append(attr.id)

        total_depth += max_attr_depth

    completion_index = connected_attributes / len(attributes)
    avg_depth = total_depth / len(attributes)

    # Interpret status
    if completion_index >= 0.7 and avg_depth >= 2.5:
        status = 'strong_chains'
    elif completion_index >= 0.5 or avg_depth >= 2.0:
        status = 'partial_chains'
    elif completion_index >= 0.3:
        status = 'weak_chains'
    else:
        status = 'fragmented'

    return {
        'completion_index': completion_index,
        'connected_count': connected_attributes,
        'total_attributes': len(attributes),
        'incomplete_attributes': incomplete_attributes,
        'avg_depth': avg_depth,
        'status': status
    }
```

#### Function 3: analyze_connectivity()

```python
def analyze_connectivity(graph: InterviewGraph) -> Dict:
    """
    Analyze graph connectivity and detect fragmentation.

    A well-conducted interview should produce a mostly connected graph.
    Multiple disconnected components suggest poor topic integration.

    Args:
        graph: InterviewGraph instance

    Returns:
        dict with:
            - component_count: Number of disconnected subgraphs
            - largest_component_size: Size of biggest connected component
            - connectivity_ratio: Largest component size / total nodes
            - has_orphans: Whether any single-node components exist
            - components: List of component sizes
    """
    components = graph.find_connected_components()

    if not components:
        return {
            'component_count': 0,
            'largest_component_size': 0,
            'connectivity_ratio': 0.0,
            'has_orphans': False,
            'components': []
        }

    component_sizes = [len(c) for c in components]
    largest = max(component_sizes)
    total_nodes = len(graph.nodes)

    return {
        'component_count': len(components),
        'largest_component_size': largest,
        'connectivity_ratio': largest / total_nodes if total_nodes > 0 else 0.0,
        'has_orphans': any(size == 1 for size in component_sizes),
        'components': sorted(component_sizes, reverse=True)
    }
```

#### Function 4: calculate_level_balance()

```python
def calculate_level_balance(graph: InterviewGraph) -> Dict:
    """
    Calculate distribution of nodes across abstraction levels.

    Based on Reynolds & Gutman (1988) abstractness ratio.
    Good interviews should have balanced exploration across levels.

    Args:
        graph: InterviewGraph instance

    Returns:
        dict with:
            - ratios: Dict of type -> percentage
            - balance_score: Chi-square distance from expected distribution
            - has_values: Whether any value nodes exist
            - total_nodes: Total node count
    """
    total = len(graph.nodes)
    if total == 0:
        return {
            'ratios': {},
            'balance_score': 0.0,
            'has_values': False,
            'total_nodes': 0
        }

    counts = {
        'attribute': len(graph.get_nodes_by_type('attribute')),
        'functional': len(graph.get_nodes_by_type('functional_consequence')),
        'psychosocial': len(graph.get_nodes_by_type('psychosocial_consequence')),
        'value': len(graph.get_nodes_by_type('value'))
    }

    ratios = {k: v/total for k, v in counts.items()}

    # Expected ratios from laddering literature
    expected = {
        'attribute': 0.40,
        'functional': 0.35,
        'psychosocial': 0.20,
        'value': 0.05
    }

    # Chi-square distance from expected
    balance_score = sum((ratios[k] - expected[k])**2 for k in expected)

    return {
        'ratios': ratios,
        'balance_score': balance_score,
        'has_values': counts['value'] > 0,
        'total_nodes': total
    }
```

#### Function 5: validate_dag_structure()

```python
def validate_dag_structure(graph: InterviewGraph) -> Dict:
    """
    Validate that graph is a directed acyclic graph (DAG).

    Means-end chains should be acyclic by definition. Cycles indicate
    extraction errors or logical inconsistencies.

    Args:
        graph: InterviewGraph instance

    Returns:
        dict with:
            - is_dag: Whether graph is acyclic
            - cycle_count: Number of cycles detected
            - cycles: List of node ID sequences forming cycles
    """
    visited = set()
    recursion_stack = set()
    cycles_found = []

    def dfs(node_id: str, path: List[str]) -> bool:
        visited.add(node_id)
        recursion_stack.add(node_id)

        for neighbor in graph.get_outgoing_neighbors(node_id):
            if neighbor not in visited:
                if dfs(neighbor, path + [neighbor]):
                    return True
            elif neighbor in recursion_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                cycles_found.append(path[cycle_start:] + [neighbor])
                return True

        recursion_stack.remove(node_id)
        return False

    for node in graph.nodes:
        if node.id not in visited:
            dfs(node.id, [node.id])

    return {
        'is_dag': len(cycles_found) == 0,
        'cycle_count': len(cycles_found),
        'cycles': cycles_found
    }
```

#### Function 6: calculate_composite_quality_score()

```python
from .config import COMPOSITE_WEIGHTS, DEPTH_TARGET, COMPLETION_TARGET, CONNECTIVITY_TARGET

def calculate_composite_quality_score(graph: InterviewGraph) -> Dict:
    """
    Calculate overall quality score combining multiple dimensions.

    Replaces single richness metric with weighted composite of:
    - Depth (30%): How deep are the chains?
    - Completion (40%): What % of attributes connect to values?
    - Connectivity (20%): How coherent is the graph?
    - Has values (10%): Have we reached terminal values?

    Args:
        graph: InterviewGraph instance

    Returns:
        dict with:
            - composite_score: Weighted quality score (0.0-1.0)
            - depth_component: Contribution from depth
            - completion_component: Contribution from completion
            - connectivity_component: Contribution from connectivity
            - values_component: Contribution from value presence
            - overall_status: Interpretation string
            - details: Full breakdown from all metrics
    """
    # Calculate all component metrics
    depth = calculate_depth_metrics(graph)
    completion = calculate_chain_completion(graph)
    connectivity = analyze_connectivity(graph)
    balance = calculate_level_balance(graph)
    dag_valid = validate_dag_structure(graph)

    # Normalize each component to 0-1 scale
    depth_score = min(depth['max_depth'] / DEPTH_TARGET, 1.0)
    completion_score = completion['completion_index']  # Already 0-1
    connectivity_score = connectivity['connectivity_ratio']  # Already 0-1
    values_score = 1.0 if balance['has_values'] else 0.0

    # Calculate weighted composite
    composite = (
        depth_score * COMPOSITE_WEIGHTS['depth'] +
        completion_score * COMPOSITE_WEIGHTS['completion'] +
        connectivity_score * COMPOSITE_WEIGHTS['connectivity'] +
        values_score * COMPOSITE_WEIGHTS['has_values']
    )

    # Determine overall status
    if not dag_valid['is_dag']:
        status = 'structural_error'
    elif connectivity_score < 0.5:
        status = 'fragmented'
    elif depth['max_depth'] == 0 and not balance['has_values']:
        status = 'surface_level'
    elif depth['max_depth'] >= 3 and completion_score > 0.6 and connectivity_score > 0.7:
        status = 'high_quality'
    elif composite >= 0.5:
        status = 'developing'
    else:
        status = 'needs_improvement'

    return {
        'composite_score': composite,
        'depth_component': depth_score * COMPOSITE_WEIGHTS['depth'],
        'completion_component': completion_score * COMPOSITE_WEIGHTS['completion'],
        'connectivity_component': connectivity_score * COMPOSITE_WEIGHTS['connectivity'],
        'values_component': values_score * COMPOSITE_WEIGHTS['has_values'],
        'overall_status': status,
        'details': {
            'depth': depth,
            'completion': completion,
            'connectivity': connectivity,
            'balance': balance,
            'dag_validation': dag_valid
        }
    }
```

---

### 3. Add Graph Helper Methods

**File:** `src/core/interview_graph.py`

Add these helper methods if they don't already exist:

```python
def find_all_paths(self, source_id: str, target_id: str, max_depth: int = 10) -> List[List[str]]:
    """
    Find all simple paths from source to target node.

    Args:
        source_id: Starting node ID
        target_id: Ending node ID
        max_depth: Maximum path length to search

    Returns:
        List of paths, where each path is a list of node IDs
    """
    import networkx as nx
    try:
        paths = list(nx.all_simple_paths(
            self.graph,
            source_id,
            target_id,
            cutoff=max_depth
        ))
        return paths
    except nx.NetworkXNoPath:
        return []
    except nx.NodeNotFound:
        logger.warning(f"Node not found when finding paths: {source_id} -> {target_id}")
        return []

def get_outgoing_neighbors(self, node_id: str) -> List[str]:
    """
    Get IDs of all nodes this node points to (outgoing edges).

    Args:
        node_id: Source node ID

    Returns:
        List of target node IDs
    """
    if not self.graph.has_node(node_id):
        return []
    return list(self.graph.successors(node_id))

def find_connected_components(self) -> List[List[str]]:
    """
    Find all connected components in the graph.

    Returns:
        List of components, where each component is a list of node IDs
    """
    import networkx as nx
    # Convert to undirected for connectivity analysis
    undirected = self.graph.to_undirected()
    components = list(nx.connected_components(undirected))
    return [list(comp) for comp in components]
```

---

### 4. Integration with Existing System

**File:** `src/interview/interview_manager.py`

Update to use new quality metrics alongside existing richness (gradual migration):

```python
from src.quality_assessment import calculate_composite_quality_score

class InterviewManager:
    def __init__(self, ...):
        # ... existing code ...
        self.quality_tracker = []  # Track quality over time

    async def process_turn(self, participant_response: str) -> Dict:
        # ... existing extraction logic ...

        # Calculate legacy richness (keep for now)
        richness = self.graph.calculate_richness()

        # Calculate new quality metrics
        quality = calculate_composite_quality_score(self.graph)

        # Store both for comparison
        self.quality_tracker.append({
            'turn': self.turn_number,
            'richness': richness,
            'quality_score': quality['composite_score'],
            'status': quality['overall_status']
        })

        # Log for debugging
        logger.info(
            f"Turn {self.turn_number}: "
            f"richness={richness:.2f}, "
            f"quality={quality['composite_score']:.2f}, "
            f"status={quality['overall_status']}"
        )

        return {
            'richness': richness,  # Legacy
            'quality': quality     # New
        }
```

---

### 5. Update Interview Termination Logic

**File:** `src/interview/interview_manager.py`

Modify `should_continue()` to use quality metrics:

```python
def should_continue(self) -> bool:
    """
    Determine if interview should continue based on quality goals.

    Uses composite quality score instead of richness threshold.
    """
    # Hard limit on turns
    if self.turn_number >= self.max_turns:
        logger.info(f"Reached max turns ({self.max_turns})")
        return False

    # Quality-based stopping
    if self.turn_number >= 8:  # Minimum viable interview
        quality = calculate_composite_quality_score(self.graph)

        # High quality achieved
        if (quality['composite_score'] >= 0.7 and
            quality['details']['completion']['completion_index'] > 0.6 and
            quality['details']['depth']['max_depth'] >= 3):
            logger.info(
                f"Quality goals achieved: score={quality['composite_score']:.2f}, "
                f"completion={quality['details']['completion']['completion_index']:.2f}"
            )
            return False

    return True
```

---

### 6. Update Report Generation

**File:** `src/reporting/report_generator.py`

Add quality metrics section to extended report:

```python
from src.quality_assessment import calculate_composite_quality_score

def _generate_quality_summary(graph: InterviewGraph) -> List[str]:
    """Generate quality assessment section for report."""
    quality = calculate_composite_quality_score(graph)

    lines = [
        "## Quality Assessment",
        "",
        f"**Composite Quality Score:** {quality['composite_score']:.2f}/1.00",
        f"**Overall Status:** {quality['overall_status']}",
        "",
        "### Component Breakdown",
        "",
        f"- **Depth Score:** {quality['depth_component']:.2f} (max depth: {quality['details']['depth']['max_depth']})",
        f"- **Completion Score:** {quality['completion_component']:.2f} ({quality['details']['completion']['connected_count']}/{quality['details']['completion']['total_attributes']} attributes connected to values)",
        f"- **Connectivity Score:** {quality['connectivity_component']:.2f} (connectivity ratio: {quality['details']['connectivity']['connectivity_ratio']:.2f})",
        f"- **Values Reached:** {quality['values_component']:.2f}",
        "",
        "### Structural Metrics",
        "",
        f"- **Max Chain Depth:** {quality['details']['depth']['max_depth']} edges",
        f"- **Average Chain Depth:** {quality['details']['depth']['avg_depth']:.2f} edges",
        f"- **Chain Count:** {quality['details']['depth']['chain_count']} complete paths",
        f"- **Connected Components:** {quality['details']['connectivity']['component_count']}",
        f"- **Largest Component:** {quality['details']['connectivity']['largest_component_size']} nodes",
        f"- **Has Orphan Nodes:** {'Yes' if quality['details']['connectivity']['has_orphans'] else 'No'}",
        "",
    ]

    # Show incomplete attributes if any
    incomplete = quality['details']['completion']['incomplete_attributes']
    if incomplete:
        lines.extend([
            "### Incomplete Attributes (no path to values)",
            "",
        ])
        for attr_id in incomplete[:5]:  # Show first 5
            lines.append(f"- {attr_id}")
        if len(incomplete) > 5:
            lines.append(f"- ... and {len(incomplete) - 5} more")
        lines.append("")

    return lines

# In generate_extended_report():
def generate_extended_report(...) -> str:
    # ... existing sections ...

    # Add quality summary after turn-by-turn analysis
    quality_summary = _generate_quality_summary(graph)
    lines.extend(quality_summary)

    # ... rest of report ...
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_quality_assessment.py`

```python
import pytest
from src.core.interview_graph import InterviewGraph
from src.core.models import Node, Edge
from src.quality_assessment import (
    calculate_depth_metrics,
    calculate_chain_completion,
    analyze_connectivity,
    calculate_composite_quality_score
)

def create_simple_chain_graph():
    """Create graph with single attribute→value chain."""
    graph = InterviewGraph()

    nodes = [
        Node(id='attr1', type='attribute', label='attr1', creation_turn=1),
        Node(id='func1', type='functional_consequence', label='func1', creation_turn=1),
        Node(id='psych1', type='psychosocial_consequence', label='psych1', creation_turn=1),
        Node(id='val1', type='value', label='val1', creation_turn=1),
    ]

    for node in nodes:
        graph.add_node(node)

    edges = [
        Edge(id='attr1-func1', type='leads_to', source='attr1', target='func1'),
        Edge(id='func1-psych1', type='leads_to', source='func1', target='psych1'),
        Edge(id='psych1-val1', type='leads_to', source='psych1', target='val1'),
    ]

    for edge in edges:
        graph.add_edge(edge)

    return graph

def test_depth_metrics_complete_chain():
    """Test depth calculation for complete chain."""
    graph = create_simple_chain_graph()
    depth = calculate_depth_metrics(graph)

    assert depth['max_depth'] == 3
    assert depth['has_complete_chains'] == True
    assert depth['chain_count'] == 1
    assert 'attr1' in depth['connected_attributes']

def test_chain_completion_perfect_score():
    """Test completion index for fully connected graph."""
    graph = create_simple_chain_graph()
    completion = calculate_chain_completion(graph)

    assert completion['completion_index'] == 1.0
    assert completion['connected_count'] == 1
    assert completion['total_attributes'] == 1
    assert len(completion['incomplete_attributes']) == 0
    assert completion['status'] == 'strong_chains'

def test_connectivity_single_component():
    """Test connectivity for fully connected graph."""
    graph = create_simple_chain_graph()
    connectivity = analyze_connectivity(graph)

    assert connectivity['component_count'] == 1
    assert connectivity['connectivity_ratio'] == 1.0
    assert connectivity['has_orphans'] == False

def test_composite_score_high_quality():
    """Test composite score for high-quality interview."""
    graph = create_simple_chain_graph()
    quality = calculate_composite_quality_score(graph)

    assert quality['composite_score'] > 0.7
    assert quality['overall_status'] in ['high_quality', 'developing']

def test_fragmented_graph():
    """Test metrics for disconnected graph."""
    graph = InterviewGraph()

    # Add two disconnected chains
    graph.add_node(Node(id='a1', type='attribute', label='a1', creation_turn=1))
    graph.add_node(Node(id='f1', type='functional_consequence', label='f1', creation_turn=1))
    graph.add_edge(Edge(id='a1-f1', type='leads_to', source='a1', target='f1'))

    graph.add_node(Node(id='a2', type='attribute', label='a2', creation_turn=1))
    graph.add_node(Node(id='f2', type='functional_consequence', label='f2', creation_turn=1))
    graph.add_edge(Edge(id='a2-f2', type='leads_to', source='a2', target='f2'))

    connectivity = analyze_connectivity(graph)

    assert connectivity['component_count'] == 2
    assert connectivity['connectivity_ratio'] < 1.0

def test_no_values_reached():
    """Test metrics when no values extracted."""
    graph = InterviewGraph()
    graph.add_node(Node(id='a1', type='attribute', label='a1', creation_turn=1))
    graph.add_node(Node(id='f1', type='functional_consequence', label='f1', creation_turn=1))

    depth = calculate_depth_metrics(graph)
    completion = calculate_chain_completion(graph)

    assert depth['max_depth'] == 0
    assert completion['completion_index'] == 0.0
    assert completion['status'] == 'no_values_reached'
```

### Integration Tests

**File:** `tests/test_interview_quality_integration.py`

```python
async def test_quality_tracking_during_interview():
    """Test that quality metrics are calculated each turn."""
    # Setup interview manager
    # Run mock interview for 5 turns
    # Verify quality_tracker populated
    # Verify quality improves or plateaus
    pass

async def test_quality_based_termination():
    """Test interview ends when quality goals achieved."""
    # Setup interview with high-quality responses
    # Verify interview ends before max_turns
    # Check termination reason is quality-based
    pass
```

---

## Migration Strategy

### Step 1: Parallel Operation (Week 1)

- Keep existing richness calculation
- Add quality metrics alongside
- Log both for comparison
- No behavioral changes

### Step 2: Validation (Week 2)

- Run 10-20 interviews
- Compare richness vs quality scores
- Validate quality metrics correlate with human assessment
- Adjust thresholds based on data

### Step 3: Gradual Migration (Week 3)

- Update termination logic to use quality score
- Keep richness in reports for reference
- Monitor for regressions

### Step 4: Full Replacement (Week 4)

- Remove richness-based logic
- Update all documentation
- Archive old metric code

---

## Expected Impact

### Before Phase 1:
```
Interview metrics:
- Richness: 27.1 (meaningless number)
- No depth information
- No completion tracking
- No fragmentation detection
```

### After Phase 1:
```
Interview quality assessment:
- Composite Score: 0.68/1.00
- Max Depth: 3 edges (attribute→value)
- Completion: 43% of attributes connected
- Connectivity: 0.85 (coherent graph)
- Status: developing

Actionable insights:
- 13 attributes need deeper probing
- 2 disconnected components to merge
- Target: Reach depth 4 and 60% completion
```

---

## Files to Create/Modify

### New Files (Create):
1. `src/quality_assessment/__init__.py`
2. `src/quality_assessment/established_metrics.py` (~300 lines)
3. `src/quality_assessment/config.py` (~50 lines)
4. `src/quality_assessment/README.md`
5. `tests/test_quality_assessment.py` (~200 lines)
6. `tests/test_interview_quality_integration.py` (~100 lines)

### Existing Files (Modify):
1. `src/core/interview_graph.py` - Add helper methods (~50 lines)
2. `src/interview/interview_manager.py` - Integrate quality tracking (~30 lines)
3. `src/reporting/report_generator.py` - Add quality section (~60 lines)

**Total:** ~6 new files, 3 modified files, ~790 new lines of code

---

## Validation Criteria

Phase 1 is successful if:

- [ ] All unit tests pass
- [ ] Quality metrics calculated without errors
- [ ] Interview_7 quality report shows meaningful metrics
- [ ] High-quality interviews score > 0.7
- [ ] Poor interviews score < 0.4
- [ ] Depth metrics correctly identify shallow vs deep interviews
- [ ] Completion index correctly identifies fragmented graphs
- [ ] No performance degradation (quality calculation < 100ms)

---

# PHASE 2: Dynamic Progress Tracking

## Overview

**Goal:** Enable real-time interview guidance through dynamic quality tracking

**Priority:** HIGH - Enables adaptive interviewing strategy

**Dependencies:** Phase 1 (established metrics)

**Expected Duration:** 3-4 days implementation + 2 days validation

## What We're Building

Three dynamic trackers that monitor interview progress turn-by-turn:

1. **MomentumTracker** - Is the interview moving up the abstraction hierarchy?
2. **ProbingEfficiencyTracker** - Which question strategies are working?
3. **SaturationTracker** - Is the interview yielding diminishing returns?

Plus integration into question generation logic for adaptive strategy selection.

---

## Implementation Details

### 1. Create Dynamic Trackers Module

**File:** `src/quality_assessment/dynamic_trackers.py`

#### Class 1: MomentumTracker

```python
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MomentumTracker:
    """
    Track interview momentum - rate of upward movement through abstraction levels.

    Good interviews maintain upward pressure toward values. Plateaus or downward
    movement indicate probing issues or saturation.
    """

    def __init__(self):
        self.turn_history: List[Tuple[int, int]] = []  # (turn_num, max_level_reached)
        self.level_map = {
            'attribute': 0,
            'functional_consequence': 1,
            'psychosocial_consequence': 2,
            'value': 3
        }

    def update(self, turn_num: int, graph) -> None:
        """Record maximum abstraction level reached at this turn."""
        if graph.nodes:
            max_level = max(self.level_map[n.type] for n in graph.nodes)
        else:
            max_level = 0

        self.turn_history.append((turn_num, max_level))

        logger.debug(f"Turn {turn_num}: max_level={max_level}")

    def calculate_momentum(self, window: int = 3) -> Dict:
        """
        Calculate recent upward movement velocity.

        Args:
            window: Number of recent turns to analyze

        Returns:
            dict with:
                - momentum: Slope of level progression (positive = upward)
                - current_level: Current max level (0-3)
                - status: Interpretation (rapid_progress, steady, plateau, regression)
                - trend: List of recent (turn, level) pairs
        """
        if len(self.turn_history) < window:
            return {
                'momentum': 0.0,
                'current_level': self.turn_history[-1][1] if self.turn_history else 0,
                'status': 'insufficient_data',
                'trend': self.turn_history
            }

        recent = self.turn_history[-window:]

        # Linear regression slope
        turns = [t[0] for t in recent]
        levels = [t[1] for t in recent]

        n = len(turns)
        mean_turn = sum(turns) / n
        mean_level = sum(levels) / n

        numerator = sum((t - mean_turn) * (l - mean_level) for t, l in zip(turns, levels))
        denominator = sum((t - mean_turn) ** 2 for t in turns)

        slope = numerator / denominator if denominator > 0 else 0

        current_level = levels[-1]

        # Interpret momentum
        if slope > 0.5:
            status = 'rapid_progress'
        elif slope > 0:
            status = 'steady_progress'
        elif slope == 0:
            status = 'plateau'
        else:
            status = 'regression'

        return {
            'momentum': slope,
            'current_level': current_level,
            'status': status,
            'trend': recent
        }

    def get_full_trajectory(self) -> List[Tuple[int, int]]:
        """Return complete turn history for visualization."""
        return self.turn_history.copy()
```

#### Class 2: ProbingEfficiencyTracker

```python
from typing import Dict, List, Optional

class ProbingEfficiencyTracker:
    """
    Track information gain per interviewer intervention.

    Monitors which question strategies yield results and which produce dead ends.
    Enables adaptive strategy selection based on what's working.
    """

    def __init__(self):
        self.turn_data: List[Dict] = []

    def record_turn(
        self,
        turn_num: int,
        strategy: str,
        nodes_before: int,
        edges_before: int,
        max_level_before: int,
        graph_after
    ) -> None:
        """
        Record productivity metrics for this turn.

        Args:
            turn_num: Current turn number
            strategy: Question strategy used (introduce_topic, dig_deeper, etc.)
            nodes_before: Node count before extraction
            edges_before: Edge count before extraction
            max_level_before: Highest abstraction level before extraction
            graph_after: InterviewGraph after extraction
        """
        nodes_added = len(graph_after.nodes) - nodes_before
        edges_added = len(graph_after.edges) - edges_before

        # Determine max level after
        if graph_after.nodes:
            level_map = {'attribute': 0, 'functional_consequence': 1,
                        'psychosocial_consequence': 2, 'value': 3}
            max_level_after = max(level_map[n.type] for n in graph_after.nodes)
            new_level = max_level_after > max_level_before
        else:
            new_level = False

        self.turn_data.append({
            'turn': turn_num,
            'strategy': strategy,
            'nodes_added': nodes_added,
            'edges_added': edges_added,
            'new_level_reached': new_level,
            'is_productive': nodes_added > 0 or edges_added > 0
        })

        logger.debug(
            f"Turn {turn_num} ({strategy}): "
            f"nodes+{nodes_added}, edges+{edges_added}, "
            f"new_level={new_level}"
        )

    def calculate_efficiency(self, recent_n: int = 5) -> Optional[Dict]:
        """
        Analyze recent probing success rate.

        Args:
            recent_n: Number of recent turns to analyze

        Returns:
            dict with:
                - success_rate: Proportion of productive turns
                - level_advance_rate: Proportion reaching new levels
                - avg_nodes_per_turn: Average nodes extracted
                - avg_edges_per_turn: Average edges extracted
                - status: Interpretation (highly_efficient, moderate, struggling, ineffective)
        """
        if not self.turn_data:
            return None

        recent = self.turn_data[-recent_n:] if len(self.turn_data) >= recent_n else self.turn_data

        total_turns = len(recent)
        productive_turns = sum(1 for t in recent if t['is_productive'])
        level_advances = sum(1 for t in recent if t['new_level_reached'])

        avg_nodes = sum(t['nodes_added'] for t in recent) / total_turns
        avg_edges = sum(t['edges_added'] for t in recent) / total_turns

        success_rate = productive_turns / total_turns

        # Interpret efficiency
        if success_rate >= 0.8:
            status = 'highly_efficient'
        elif success_rate >= 0.6:
            status = 'moderate'
        elif success_rate >= 0.4:
            status = 'struggling'
        else:
            status = 'ineffective'

        return {
            'success_rate': success_rate,
            'level_advance_rate': level_advances / total_turns,
            'avg_nodes_per_turn': avg_nodes,
            'avg_edges_per_turn': avg_edges,
            'status': status
        }

    def strategy_effectiveness(self) -> Dict[str, Dict]:
        """
        Calculate success rate by strategy type.

        Returns:
            dict mapping strategy -> {attempts, successes, success_rate}
        """
        strategy_stats = {}

        for turn in self.turn_data:
            s = turn['strategy']
            if s not in strategy_stats:
                strategy_stats[s] = {'attempts': 0, 'successes': 0}

            strategy_stats[s]['attempts'] += 1
            if turn['is_productive']:
                strategy_stats[s]['successes'] += 1

        for s, stats in strategy_stats.items():
            stats['success_rate'] = stats['successes'] / stats['attempts']

        return strategy_stats
```

#### Class 3: SaturationTracker

```python
class SaturationTracker:
    """
    Detect when interview reaches saturation (diminishing returns).

    Based on theoretical saturation concept from grounded theory (Glaser & Strauss 1967).
    Adapted for real-time detection during interviews.
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.new_nodes_per_turn: List[int] = []

    def update(self, turn_number: int, new_node_count: int) -> None:
        """Record number of new nodes extracted this turn."""
        self.new_nodes_per_turn.append(new_node_count)
        logger.debug(f"Turn {turn_number}: {new_node_count} new nodes")

    def is_saturated(self) -> bool:
        """
        Check if interview has reached saturation.

        Returns True if last N turns yielded zero new nodes.
        """
        if len(self.new_nodes_per_turn) < self.window_size:
            return False

        recent = self.new_nodes_per_turn[-self.window_size:]
        return sum(recent) == 0

    def saturation_velocity(self) -> Optional[float]:
        """
        Calculate rate of decline in new concept generation.

        Returns:
            Slope of recent trend (negative = declining productivity)
            None if insufficient data
        """
        if len(self.new_nodes_per_turn) < 4:
            return None

        recent = self.new_nodes_per_turn[-4:]

        # Simple linear regression
        x = list(range(len(recent)))
        mean_x = sum(x) / len(x)
        mean_y = sum(recent) / len(recent)

        numerator = sum((i - mean_x) * (v - mean_y) for i, v in zip(x, recent))
        denominator = sum((i - mean_x) ** 2 for i in x)

        slope = numerator / denominator if denominator > 0 else 0

        return slope

    def get_productivity_trend(self) -> List[int]:
        """Return full history of new nodes per turn."""
        return self.new_nodes_per_turn.copy()
```

---

### 2. Integrate with InterviewManager

**File:** `src/interview/interview_manager.py`

```python
from src.quality_assessment.dynamic_trackers import (
    MomentumTracker,
    ProbingEfficiencyTracker,
    SaturationTracker
)

class InterviewManager:
    def __init__(self, ...):
        # ... existing code ...

        # Phase 2: Add dynamic trackers
        self.momentum_tracker = MomentumTracker()
        self.efficiency_tracker = ProbingEfficiencyTracker()
        self.saturation_tracker = SaturationTracker(window_size=3)

    async def process_turn(self, participant_response: str) -> Dict:
        # Capture state before extraction
        nodes_before = len(self.graph.nodes)
        edges_before = len(self.graph.edges)

        level_map = {'attribute': 0, 'functional_consequence': 1,
                    'psychosocial_consequence': 2, 'value': 3}
        max_level_before = max([level_map[n.type] for n in self.graph.nodes]) if self.graph.nodes else 0

        # ... existing extraction logic ...
        extraction_result = await self.extractor.extract(participant_response)

        # Update graph
        for node in extraction_result.nodes:
            self.graph.add_node(node)
        for edge in extraction_result.edges:
            self.graph.add_edge(edge)

        # Track dynamics
        new_node_count = len(extraction_result.nodes)
        self.saturation_tracker.update(self.turn_number, new_node_count)
        self.momentum_tracker.update(self.turn_number, self.graph)

        # ... question generation ...
        strategy = selected_opportunity.strategy.value

        # Record efficiency after question generation
        self.efficiency_tracker.record_turn(
            self.turn_number,
            strategy,
            nodes_before,
            edges_before,
            max_level_before,
            self.graph
        )

        # Calculate current state
        momentum = self.momentum_tracker.calculate_momentum()
        efficiency = self.efficiency_tracker.calculate_efficiency()
        is_saturated = self.saturation_tracker.is_saturated()

        logger.info(
            f"Dynamics: momentum={momentum['status']}, "
            f"efficiency={efficiency['status'] if efficiency else 'N/A'}, "
            f"saturated={is_saturated}"
        )

        return {
            # ... existing return values ...
            'dynamics': {
                'momentum': momentum,
                'efficiency': efficiency,
                'saturated': is_saturated
            }
        }
```

---

### 3. Adaptive Question Generation

**File:** `src/interview/interview_manager.py`

Add decision logic based on dynamic metrics:

```python
def _select_strategy_adaptively(self, opportunities: List) -> Opportunity:
    """
    Select opportunity using dynamic quality signals.

    Adapts strategy based on:
    - Momentum (are we climbing the ladder?)
    - Efficiency (what's working?)
    - Saturation (are we done?)
    """
    # Check saturation first
    if self.saturation_tracker.is_saturated() and self.turn_number >= 8:
        logger.info("Interview saturated - preparing to wrap up")
        # Return wrap-up strategy
        return self._create_wrap_up_opportunity()

    # Check efficiency
    efficiency = self.efficiency_tracker.calculate_efficiency()
    if efficiency and efficiency['success_rate'] < 0.4 and self.turn_number >= 5:
        # Current approach not working - try best-performing strategy
        strategy_stats = self.efficiency_tracker.strategy_effectiveness()
        if strategy_stats:
            best_strategy = max(strategy_stats.items(), key=lambda x: x[1]['success_rate'])[0]
            logger.info(f"Low efficiency - switching to {best_strategy}")
            # Filter opportunities by best strategy
            filtered = [o for o in opportunities if o.strategy.value == best_strategy]
            if filtered:
                return filtered[0]

    # Check momentum
    momentum = self.momentum_tracker.calculate_momentum()
    if momentum['status'] == 'plateau' and self.turn_number >= 4:
        # Stuck at current level - explicitly probe upward
        logger.info("Momentum plateau detected - probing for higher abstraction")
        # Prioritize opportunities that could yield higher-level nodes
        return self._select_upward_opportunity(opportunities)

    # Default: use standard opportunity ranking
    return opportunities[0]

def _select_upward_opportunity(self, opportunities: List) -> Opportunity:
    """Select opportunity most likely to yield higher abstraction level."""
    # Prefer nodes at higher current levels (more likely to climb further)
    level_map = {'attribute': 0, 'functional_consequence': 1,
                'psychosocial_consequence': 2, 'value': 3}

    scored = []
    for opp in opportunities:
        node = self.graph.get_node(opp.node_id)
        level_score = level_map[node.type]
        scored.append((opp, level_score))

    # Sort by level (highest first), then by priority
    scored.sort(key=lambda x: (x[1], x[0].priority_score), reverse=True)

    return scored[0][0]

def _create_wrap_up_opportunity(self) -> Opportunity:
    """Create wrap-up opportunity when interview is complete."""
    # Implementation for graceful conclusion
    pass
```

---

### 4. Update Termination Logic

**File:** `src/interview/interview_manager.py`

```python
def should_continue(self) -> bool:
    """Enhanced termination logic using dynamic signals."""
    # Hard limit
    if self.turn_number >= self.max_turns:
        return False

    # Minimum viable interview
    if self.turn_number < 8:
        return True

    # Quality-based stopping (from Phase 1)
    quality = calculate_composite_quality_score(self.graph)

    # Saturation detection (Phase 2)
    if self.saturation_tracker.is_saturated():
        logger.info("Saturation detected")
        # Check if we've achieved minimum quality
        if quality['composite_score'] >= 0.6:
            logger.info("Quality sufficient - ending interview")
            return False

    # Efficiency-based early stopping
    efficiency = self.efficiency_tracker.calculate_efficiency()
    if efficiency and efficiency['success_rate'] < 0.3 and self.turn_number >= 10:
        logger.warning(f"Low efficiency ({efficiency['success_rate']:.2f}) - ending interview")
        return False

    # High quality achieved
    if quality['composite_score'] >= 0.7 and quality['details']['completion']['completion_index'] > 0.6:
        logger.info("High quality achieved - ending interview")
        return False

    return True
```

---

### 5. Update Reports with Dynamic Metrics

**File:** `src/reporting/report_generator.py`

```python
def _generate_dynamics_summary(manager: InterviewManager) -> List[str]:
    """Generate summary of interview dynamics."""
    momentum = manager.momentum_tracker.calculate_momentum()
    efficiency = manager.efficiency_tracker.calculate_efficiency()
    saturation_vel = manager.saturation_tracker.saturation_velocity()
    strategy_stats = manager.efficiency_tracker.strategy_effectiveness()

    lines = [
        "## Interview Dynamics",
        "",
        "### Momentum Analysis",
        "",
        f"- **Status:** {momentum['status']}",
        f"- **Current Level:** {momentum['current_level']}/3",
        f"- **Trajectory:** {momentum['momentum']:.3f} levels/turn",
        "",
    ]

    if efficiency:
        lines.extend([
            "### Probing Efficiency",
            "",
            f"- **Overall Status:** {efficiency['status']}",
            f"- **Success Rate:** {efficiency['success_rate']:.1%} of turns productive",
            f"- **Level Advances:** {efficiency['level_advance_rate']:.1%} of turns reached new levels",
            f"- **Average Extraction:** {efficiency['avg_nodes_per_turn']:.1f} nodes, {efficiency['avg_edges_per_turn']:.1f} edges per turn",
            "",
        ])

    if strategy_stats:
        lines.extend([
            "### Strategy Effectiveness",
            "",
        ])
        for strategy, stats in sorted(strategy_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            lines.append(
                f"- **{strategy}:** {stats['success_rate']:.1%} success "
                f"({stats['successes']}/{stats['attempts']} attempts)"
            )
        lines.append("")

    if saturation_vel is not None:
        lines.extend([
            "### Saturation Analysis",
            "",
            f"- **Velocity:** {saturation_vel:.3f} (negative = declining productivity)",
            f"- **Status:** {'Saturated' if manager.saturation_tracker.is_saturated() else 'Active'}",
            "",
        ])

    return lines
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_dynamic_trackers.py`

```python
def test_momentum_tracker_upward_progress():
    """Test momentum calculation for improving interview."""
    tracker = MomentumTracker()

    # Simulate upward progression
    mock_graphs = [
        create_graph_with_max_level(0),  # Turn 1: attribute only
        create_graph_with_max_level(1),  # Turn 2: functional
        create_graph_with_max_level(2),  # Turn 3: psychosocial
    ]

    for turn, graph in enumerate(mock_graphs, 1):
        tracker.update(turn, graph)

    momentum = tracker.calculate_momentum(window=3)

    assert momentum['momentum'] > 0
    assert momentum['status'] in ['rapid_progress', 'steady_progress']
    assert momentum['current_level'] == 2

def test_efficiency_tracker_strategy_learning():
    """Test that tracker identifies successful strategies."""
    tracker = ProbingEfficiencyTracker()

    # Record turns with different strategies
    tracker.record_turn(1, 'introduce_topic', 0, 0, 0, graph_with_new_nodes)
    tracker.record_turn(2, 'dig_deeper', 3, 2, 1, graph_with_new_nodes)
    tracker.record_turn(3, 'dig_deeper', 5, 4, 1, graph_with_new_nodes)
    tracker.record_turn(4, 'introduce_topic', 7, 6, 1, graph_no_change)

    stats = tracker.strategy_effectiveness()

    assert stats['dig_deeper']['success_rate'] > stats['introduce_topic']['success_rate']

def test_saturation_detection():
    """Test saturation detection after multiple empty turns."""
    tracker = SaturationTracker(window_size=3)

    tracker.update(1, 3)  # Productive
    tracker.update(2, 2)  # Productive
    tracker.update(3, 0)  # Empty
    tracker.update(4, 0)  # Empty
    tracker.update(5, 0)  # Empty

    assert tracker.is_saturated() == True
```

---

## Expected Impact

### Before Phase 2:
```
Interview progression:
- No feedback on strategy effectiveness
- Random question strategy selection
- No detection of plateaus
- Interviews run to max_turns regardless of productivity
```

### After Phase 2:
```
Turn 5:
- Momentum: plateau (stuck at functional level)
- Efficiency: 60% success rate
- Action: Switch to upward-probing questions

Turn 8:
- Momentum: steady progress (reached psychosocial)
- Efficiency: dig_deeper at 80% success (best strategy)
- Action: Continue dig_deeper on psychosocial nodes

Turn 11:
- Momentum: slow
- Saturation: detected (3 empty turns)
- Quality: 0.72 (sufficient)
- Action: End interview (goals achieved)
```

---

## Files to Create/Modify

### New Files (Create):
1. `src/quality_assessment/dynamic_trackers.py` (~350 lines)
2. `tests/test_dynamic_trackers.py` (~200 lines)

### Existing Files (Modify):
1. `src/interview/interview_manager.py` - Add trackers and adaptive logic (~120 lines)
2. `src/reporting/report_generator.py` - Add dynamics section (~80 lines)
3. `src/quality_assessment/__init__.py` - Export new classes (~10 lines)

**Total:** 2 new files, 3 modified files, ~760 new lines of code

---

## Validation Criteria

Phase 2 is successful if:

- [ ] Momentum correctly identifies upward vs plateau vs regression
- [ ] Efficiency tracker identifies best-performing strategies
- [ ] Saturation detection prevents endless interviews
- [ ] Adaptive strategy selection improves interview quality
- [ ] Interviews end when quality achieved + saturated (not just max_turns)
- [ ] Reports show meaningful dynamics metrics
- [ ] Strategy success rates guide future question selection

---

# PHASE 3: Advanced Features (Optional)

## Overview

**Goal:** Enhanced quality features for production deployment

**Priority:** OPTIONAL - Provides incremental improvements beyond core functionality

**Dependencies:** Phases 1 and 2

**Expected Duration:** 4-5 days implementation + 3 days validation

## What We're Building

Three advanced features:

1. **SemanticCoherenceTracker** - Detect topic jumps and redundant concepts using embeddings
2. **DiversityDepthTracker** - Balance breadth vs depth exploration
3. **Enhanced Validation** - Schema-aware edge type validation
4. **Quality Dashboard** - Real-time visualization (bonus)

**Important:** Phase 3 requires adding `sentence-transformers` dependency.

---

## Implementation Details

### 1. Add Dependencies

**File:** `requirements.txt` or `pyproject.toml`

```
sentence-transformers>=2.2.0  # For semantic coherence
```

Install model (one-time setup):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

### 2. Semantic Coherence Tracker

**File:** `src/quality_assessment/dynamic_trackers.py` (append to existing)

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional

class SemanticCoherenceTracker:
    """
    Track semantic coherence of extracted concepts.

    Uses sentence embeddings to detect:
    - Topic drift (participant jumping between unrelated concepts)
    - Redundancy (extracting near-duplicate nodes)
    - Coherent exploration (new concepts related to existing graph)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings: Dict[str, np.ndarray] = {}  # node_id -> embedding
        logger.info(f"Initialized semantic coherence tracker with {model_name}")

    def add_node_embedding(self, node_id: str, node_label: str) -> None:
        """Compute and store embedding for new node."""
        embedding = self.model.encode(node_label)
        self.embeddings[node_id] = embedding
        logger.debug(f"Added embedding for node: {node_id}")

    def calculate_turn_coherence(
        self,
        new_node_ids: List[str],
        existing_node_ids: List[str]
    ) -> Optional[Dict]:
        """
        Calculate how semantically related new nodes are to existing graph.

        Args:
            new_node_ids: IDs of nodes just extracted
            existing_node_ids: IDs of nodes from previous turns

        Returns:
            dict with:
                - coherence_score: Average max similarity (0-1)
                - status: Interpretation (highly_related, moderately_related, etc.)
                - similarities: Per-node similarity scores
        """
        if not new_node_ids or not existing_node_ids:
            return None

        new_embeds = np.array([self.embeddings[nid] for nid in new_node_ids
                              if nid in self.embeddings])
        existing_embeds = np.array([self.embeddings[nid] for nid in existing_node_ids
                                   if nid in self.embeddings])

        if len(new_embeds) == 0 or len(existing_embeds) == 0:
            return None

        # For each new node, find max similarity to existing nodes
        similarities = []
        for new_emb in new_embeds:
            sims = [self._cosine_similarity(new_emb, ex_emb) for ex_emb in existing_embeds]
            max_sim = max(sims)
            similarities.append(max_sim)

        avg_coherence = float(np.mean(similarities))

        # Interpret coherence
        if avg_coherence > 0.7:
            status = 'highly_related'
        elif avg_coherence > 0.5:
            status = 'moderately_related'
        elif avg_coherence > 0.3:
            status = 'loosely_related'
        else:
            status = 'disconnected'

        return {
            'coherence_score': avg_coherence,
            'status': status,
            'similarities': similarities
        }

    def detect_redundancy(
        self,
        new_node_id: str,
        threshold: float = 0.85
    ) -> Optional[Dict]:
        """
        Check if new node is too similar to existing nodes (potential duplicate).

        Args:
            new_node_id: ID of newly extracted node
            threshold: Similarity threshold for redundancy (default 0.85)

        Returns:
            dict with:
                - is_redundant: Boolean
                - similar_nodes: List of (node_id, similarity) tuples
        """
        if new_node_id not in self.embeddings:
            return None

        new_emb = self.embeddings[new_node_id]

        similarities = [
            (node_id, float(self._cosine_similarity(new_emb, emb)))
            for node_id, emb in self.embeddings.items()
            if node_id != new_node_id
        ]

        duplicates = [(nid, sim) for nid, sim in similarities if sim > threshold]

        return {
            'is_redundant': len(duplicates) > 0,
            'similar_nodes': duplicates
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

---

### 3. Diversity-Depth Balance Tracker

**File:** `src/quality_assessment/dynamic_trackers.py` (append)

```python
class DiversityDepthTracker:
    """
    Track balance between exploring breadth (new topics) and depth (laddering).

    Optimal interviews balance coverage and depth. This tracker provides
    guidance on whether to introduce new topics or probe existing ones deeper.
    """

    def __init__(self):
        self.turn_metrics: List[Tuple[int, int, int]] = []  # (turn, topics, max_depth)

    def update(self, turn_num: int, graph) -> None:
        """Record current breadth and depth state."""
        # Estimate topics via connected components
        components = graph.find_connected_components()
        unique_topics = len(components)

        # Calculate max chain depth
        from src.quality_assessment import calculate_depth_metrics
        depth_metrics = calculate_depth_metrics(graph)
        max_depth = depth_metrics['max_depth']

        self.turn_metrics.append((turn_num, unique_topics, max_depth))

    def calculate_balance(
        self,
        topic_target: float = 2.5,
        depth_target: float = 3.5
    ) -> Dict:
        """
        Calculate current balance state.

        Args:
            topic_target: Ideal number of topics (adjust based on domain)
            depth_target: Ideal max depth (adjust based on domain)

        Returns:
            dict with:
                - current_topics: Number of disconnected components
                - current_depth: Max chain depth
                - balance_score: Overall balance (0-1, higher = better)
                - recommendation: What to do next
        """
        if not self.turn_metrics:
            return {
                'current_topics': 0,
                'current_depth': 0,
                'balance_score': 0.0,
                'recommendation': 'introduce_new_topic'
            }

        _, topics, depth = self.turn_metrics[-1]

        # Normalized scores (1.0 = at target, lower = farther from target)
        topic_score = min(topics / topic_target, 2 - topics / topic_target)
        depth_score = min(depth / depth_target, 2 - depth / depth_target)

        # Balance: both should be close to 1.0
        balance_score = (topic_score + depth_score) / 2

        # Recommendation
        recommendation = self._recommend_action(topics, depth, topic_target, depth_target)

        return {
            'current_topics': topics,
            'current_depth': depth,
            'balance_score': balance_score,
            'recommendation': recommendation
        }

    @staticmethod
    def _recommend_action(
        topics: int,
        depth: int,
        topic_target: float,
        depth_target: float
    ) -> str:
        """Determine recommended next action based on current state."""
        if topics < topic_target * 0.7 and depth < depth_target * 0.7:
            return 'introduce_new_topic'
        elif topics < topic_target * 0.7:
            return 'explore_breadth'
        elif depth < depth_target * 0.7:
            return 'probe_deeper'
        else:
            return 'maintain_balance'

    def trajectory_analysis(self, window: int = 5) -> Optional[Dict]:
        """
        Analyze recent trajectory trends.

        Returns:
            dict with:
                - topic_trajectory: expanding, stable, or consolidating
                - depth_trajectory: deepening, stable, or shallowing
        """
        if len(self.turn_metrics) < window:
            return None

        recent = self.turn_metrics[-window:]

        # Trend in topics
        topic_trend = recent[-1][1] - recent[0][1]

        # Trend in depth
        depth_trend = recent[-1][2] - recent[0][2]

        return {
            'topic_trajectory': (
                'expanding' if topic_trend > 0 else
                'stable' if topic_trend == 0 else
                'consolidating'
            ),
            'depth_trajectory': (
                'deepening' if depth_trend > 0 else
                'stable' if depth_trend == 0 else
                'shallowing'
            )
        }
```

---

### 4. Enhanced Schema Validation

**File:** `src/quality_assessment/validation.py` (new file)

```python
"""Enhanced validation for schema compliance."""

from typing import Dict, List
from src.core.interview_graph import InterviewGraph
import logging

logger = logging.getLogger(__name__)

# Define valid edge types between node types (from means-end chain theory)
VALID_EDGE_TYPES = {
    ('attribute', 'functional_consequence'): ['leads_to', 'enables', 'correlates_with'],
    ('attribute', 'psychosocial_consequence'): ['leads_to', 'enables'],
    ('attribute', 'value'): ['leads_to'],  # Direct jumps rare but valid
    ('functional_consequence', 'psychosocial_consequence'): ['leads_to', 'enables', 'correlates_with'],
    ('functional_consequence', 'value'): ['leads_to', 'enables'],
    ('psychosocial_consequence', 'value'): ['leads_to', 'enables'],
    # Same-level connections (less common)
    ('attribute', 'attribute'): ['correlates_with', 'exemplifies'],
    ('functional_consequence', 'functional_consequence'): ['correlates_with', 'enables'],
    # Blocking relationships
    ('attribute', 'functional_consequence'): ['blocks'],
    ('functional_consequence', 'psychosocial_consequence'): ['blocks'],
}

def validate_schema_compliance(graph: InterviewGraph) -> Dict:
    """
    Validate that all edges comply with means-end chain schema.

    Returns:
        dict with:
            - is_valid: Whether all edges are valid
            - invalid_edges: List of edge violations
            - warnings: List of unusual but not invalid patterns
    """
    invalid_edges = []
    warnings = []

    for edge in graph.edges:
        source_node = graph.get_node(edge.source)
        target_node = graph.get_node(edge.target)

        if not source_node or not target_node:
            invalid_edges.append({
                'edge_id': edge.id,
                'reason': 'missing_node',
                'details': f"Source or target node not found"
            })
            continue

        edge_key = (source_node.type, target_node.type)

        # Check if this connection type is allowed
        if edge_key not in VALID_EDGE_TYPES:
            invalid_edges.append({
                'edge_id': edge.id,
                'reason': 'invalid_connection',
                'details': f"{source_node.type} -> {target_node.type} not in schema"
            })
            continue

        # Check if edge type is allowed for this connection
        allowed_types = VALID_EDGE_TYPES[edge_key]
        if edge.type not in allowed_types:
            invalid_edges.append({
                'edge_id': edge.id,
                'reason': 'invalid_edge_type',
                'details': f"{edge.type} not allowed for {source_node.type} -> {target_node.type}"
            })
            continue

        # Check for unusual patterns (warnings, not errors)
        if source_node.type == target_node.type and edge.type == 'leads_to':
            warnings.append({
                'edge_id': edge.id,
                'warning': 'same_level_leads_to',
                'details': f"leads_to between same level ({source_node.type}) is unusual"
            })

    return {
        'is_valid': len(invalid_edges) == 0,
        'invalid_edges': invalid_edges,
        'warnings': warnings
    }
```

---

### 5. Integration with Interview Manager

**File:** `src/interview/interview_manager.py`

```python
class InterviewManager:
    def __init__(self, ..., use_semantic_coherence: bool = False):
        # ... existing trackers ...

        # Phase 3: Optional advanced trackers
        self.semantic_tracker = SemanticCoherenceTracker() if use_semantic_coherence else None
        self.diversity_tracker = DiversityDepthTracker()

    async def process_turn(self, participant_response: str) -> Dict:
        # ... existing extraction ...

        # Phase 3: Semantic coherence tracking
        if self.semantic_tracker:
            existing_node_ids = [n.id for n in self.graph.nodes if n.creation_turn < self.turn_number]
            new_node_ids = [n.id for n in extraction_result.nodes]

            # Add embeddings for new nodes
            for node in extraction_result.nodes:
                self.semantic_tracker.add_node_embedding(node.id, node.label)

            # Calculate coherence
            if existing_node_ids and new_node_ids:
                coherence = self.semantic_tracker.calculate_turn_coherence(
                    new_node_ids,
                    existing_node_ids
                )

                if coherence and coherence['status'] == 'disconnected':
                    logger.warning(
                        f"Low semantic coherence ({coherence['coherence_score']:.2f}) - "
                        f"possible topic jump"
                    )

        # Diversity-depth balance
        self.diversity_tracker.update(self.turn_number, self.graph)
        balance = self.diversity_tracker.calculate_balance()

        logger.info(
            f"Balance: {balance['current_topics']} topics, "
            f"depth {balance['current_depth']} - {balance['recommendation']}"
        )

        return {
            # ... existing return ...
            'coherence': coherence if self.semantic_tracker else None,
            'balance': balance
        }
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_semantic_coherence.py`

```python
def test_semantic_coherence_related_concepts():
    """Test that semantically related concepts score high."""
    tracker = SemanticCoherenceTracker()

    # Add related concepts
    tracker.add_node_embedding('fresh_beans', 'fresh beans')
    tracker.add_node_embedding('aromatic_coffee', 'aromatic coffee')
    tracker.add_node_embedding('pleasant_smell', 'pleasant smell')

    coherence = tracker.calculate_turn_coherence(
        new_node_ids=['aromatic_coffee'],
        existing_node_ids=['fresh_beans']
    )

    assert coherence['coherence_score'] > 0.5
    assert coherence['status'] in ['highly_related', 'moderately_related']

def test_redundancy_detection():
    """Test detection of near-duplicate nodes."""
    tracker = SemanticCoherenceTracker()

    tracker.add_node_embedding('fresh_beans', 'fresh beans')
    tracker.add_node_embedding('freshly_roasted_beans', 'freshly roasted beans')

    redundancy = tracker.detect_redundancy('freshly_roasted_beans', threshold=0.85)

    assert redundancy['is_redundant'] == True
    assert len(redundancy['similar_nodes']) > 0
```

---

## Expected Impact

### Phase 3 Capabilities:

```
Turn 5:
- Semantic coherence: 0.72 (moderately related to previous topics)
- Diversity-depth balance: 3 topics, depth 2 - recommendation: probe_deeper
- No redundant concepts detected

Turn 8:
- Semantic coherence: 0.35 (disconnected) - WARNING: possible topic jump
- Action: Ask clarifying question to connect to existing topics

Turn 10:
- Redundancy detected: "freshly_roasted" similar to "fresh_beans" (0.89)
- Action: Skip probing redundant node, focus on unique concepts
```

---

## Files to Create/Modify

### New Files (Create):
1. `src/quality_assessment/validation.py` (~100 lines)
2. `tests/test_semantic_coherence.py` (~150 lines)
3. `tests/test_diversity_depth.py` (~100 lines)

### Existing Files (Modify):
1. `src/quality_assessment/dynamic_trackers.py` - Add 2 new classes (~250 lines)
2. `src/interview/interview_manager.py` - Integrate Phase 3 features (~60 lines)
3. `requirements.txt` - Add sentence-transformers

**Total:** 3 new files, 3 modified files, ~660 new lines of code

**Dependencies Added:** sentence-transformers (~500MB model download)

---

## Validation Criteria

Phase 3 is successful if:

- [ ] Semantic coherence correctly identifies related vs unrelated concepts
- [ ] Redundancy detection flags near-duplicate nodes
- [ ] Diversity-depth tracker provides actionable guidance
- [ ] Schema validation catches invalid edge types
- [ ] No performance regression (semantic processing < 200ms per turn)
- [ ] Model downloads and initializes successfully

---

# OVERALL IMPLEMENTATION TIMELINE

## Week 1: Phase 1 Foundation
- Days 1-2: Implement established metrics
- Day 3: Add graph helper methods
- Day 4: Integration and unit tests
- Day 5: Validation with real interviews

## Week 2: Phase 2 Dynamics
- Days 1-2: Implement dynamic trackers
- Day 3: Adaptive question generation
- Day 4: Integration and testing
- Day 5: Validation and refinement

## Week 3: Phase 3 Advanced (Optional)
- Days 1-2: Semantic coherence tracker
- Day 3: Diversity-depth balance
- Days 4-5: Integration, testing, validation

## Week 4: Refinement and Documentation
- Calibrate thresholds based on real data
- Update all documentation
- Create migration guide
- Final validation suite

---

# SUCCESS METRICS

The multi-phase implementation is successful if:

## Quantitative Metrics:
- [ ] Quality score correlates 0.7+ with human assessment
- [ ] High-quality interviews (human-rated) score > 0.7
- [ ] Poor interviews (human-rated) score < 0.4
- [ ] Strategy success rate improves by 30%+
- [ ] Interview length reduces 20% while maintaining quality
- [ ] No runtime performance degradation

## Qualitative Improvements:
- [ ] Reports clearly show interview strengths/weaknesses
- [ ] Saturation detection prevents overly long interviews
- [ ] Adaptive strategy selection improves probing effectiveness
- [ ] Fragmentation and shallow interviews are detected
- [ ] Clear actionable feedback for interview improvement

---

# ROLLBACK PLAN

If any phase fails validation:

1. **Phase 1 Rollback:** Revert to richness metric, keep code in feature branch
2. **Phase 2 Rollback:** Disable dynamic trackers, use static Phase 1 metrics only
3. **Phase 3 Rollback:** Disable semantic/diversity trackers, core functionality unaffected

Each phase is independent - can deploy 1 without 2, or 1+2 without 3.

---

**End of Implementation Plan**
