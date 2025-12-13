# Component Map

## Table of Contents
1. [Module Overview](#module-overview)
2. [Core Module](#core-module)
3. [Decision Module](#decision-module)
4. [Generation Module](#generation-module)
5. [Utils Module](#utils-module)
6. [UI Module](#ui-module)
7. [Controller](#controller)
8. [Component Dependency Graph](#component-dependency-graph)
9. [Interaction Patterns](#interaction-patterns)
10. [State Management](#state-management)
11. [Async/Concurrency](#asyncconcurrency)

---

## Module Overview

### Directory Structure
```
src/
├── core/               # Data structures & state
│   ├── graph.py
│   ├── state.py
│   ├── schema.py
│   └── history.py
├── decision/          # Strategy & extraction logic
│   ├── strategy.py
│   ├── arbitration.py
│   └── extraction.py
├── generation/        # Question generation
│   └── generator.py
├── utils/             # Cross-cutting utilities
│   ├── llm_manager.py
│   ├── logger.py
│   └── concept_parser.py
├── ui/                # User interface
│   └── gradio_app.py
├── config/            # Configuration files
│   ├── interview_logic.yaml
│   ├── llm_config.yaml
│   └── schemas/
└── controller.py      # Main orchestrator
```

### Module Dependencies

```
┌─────────────┐
│ controller  │
└──────┬──────┘
       │
       ├───────────┬──────────────┬───────────────┬────────────┐
       │           │              │               │            │
       ▼           ▼              ▼               ▼            ▼
   ┌──────┐   ┌─────────┐   ┌──────────┐   ┌─────────┐  ┌────────┐
   │ core │   │decision │   │generation│   │  utils  │  │  ui    │
   └──────┘   └─────────┘   └──────────┘   └─────────┘  └────────┘
       │           │              │               │            │
       │           └──────────────┴───────────────┘            │
       │                          │                            │
       └──────────────────────────┴────────────────────────────┘
```

---

## Core Module

### 1. Graph ([src/core/graph.py](src/core/graph.py))

**Purpose**: Pure data structure for knowledge graph storage and queries.

**Responsibilities**:
- Store nodes and edges
- Provide graph queries (neighbors, isolated nodes, etc.)
- Maintain referential integrity (no orphan edges)

**Dependencies**:
- `Node` (dataclass)
- `Edge` (dataclass)
- `typing` (standard library)
- `uuid` (for ID generation)

**Dependents**:
- `InterviewController` (primary consumer)
- `GraphState` (reads graph for state computation)
- `CoverageState` (reads graph for element mapping)
- `Extractor` (adds nodes/edges)
- `StrategySelector` (reads graph for decision-making)

**Key Methods**:

| Method | Return Type | Purpose | Complexity |
|--------|-------------|---------|------------|
| `add_node(node)` | `str` (node_id) | Add node, de-dup by label | O(N) |
| `add_edge(edge)` | `str` (edge_id) | Add edge, check nodes exist | O(1) |
| `get_node(node_id)` | `Optional[Node]` | Retrieve node by ID | O(1) |
| `get_node_by_label(label)` | `Optional[Node]` | Find node by label (case-insensitive) | O(N) |
| `get_neighbors(node_id)` | `List[Node]` | Get adjacent nodes | O(E) |
| `get_isolated_nodes()` | `List[Node]` | Find nodes with no edges | O(N + E) |
| `get_edge_between(source, target)` | `Optional[Edge]` | Check if edge exists | O(E) |

**State Management**:
- **Mutable**: Nodes and edges stored in dicts
- **Thread safety**: Not thread-safe (assumes single-threaded)
- **Consistency**: No caching; queries always return current state

**Interaction Pattern**:
```
Extractor extracts nodes/edges
         │
         └──► Graph.add_node(node)
                  │
                  ├──► Check if label exists (get_node_by_label)
                  │       │
                  │       ├─ exists ──► Return existing node ID
                  │       └─ new ─────► Add to nodes dict
                  │
         ┌────────┘
         │
         └──► Graph.add_edge(edge)
                  │
                  └──► Validate source/target exist
                       │
                       ├─ valid ──► Add to edges dict
                       └─ invalid ──► Log error, skip
```

**Critical Invariants**:
1. No duplicate labels (case-insensitive)
2. All edges reference existing nodes
3. Node IDs are unique UUIDs

---

### 2. Node ([src/core/graph.py](src/core/graph.py))

**Purpose**: Represents a concept, entity, or idea in the knowledge graph.

**Structure**:
```python
@dataclass
class Node:
    id: str                        # UUID
    label: str                     # Respondent's language
    node_type: Optional[str]       # Schema type (e.g., "attribute", "value")
    timestamp: datetime            # When extracted
    is_ambiguous: bool = False     # Flagged for clarification
    metadata: Dict = field(default_factory=dict)  # Extra data
```

**Metadata Keys** (common):
- `quote`: Verbatim text from respondent
- `element_mapping`: ID of stimulus element this maps to
- `reaction`: "positive" | "negative" | "neutral" | "skeptical" | "curious"
- `confidence`: 0.0-1.0 (if available)

**Dependencies**: None (pure dataclass)

**Dependents**: All components that work with graph

---

### 3. Edge ([src/core/graph.py](src/core/graph.py))

**Purpose**: Represents a relationship between two nodes.

**Structure**:
```python
@dataclass
class Edge:
    id: str                        # UUID
    source_id: str                 # Node ID
    target_id: str                 # Node ID
    relation_type: str             # Schema type (e.g., "leads_to", "requires")
    metadata: Dict = field(default_factory=dict)  # Extra data
```

**Metadata Keys** (common):
- `quote`: Verbatim text from respondent
- `confidence`: 0.0-1.0 (if available)
- `inferred`: bool (extracted directly vs inferred)

**Dependencies**: None (pure dataclass)

**Dependents**: All components that work with graph

---

### 4. Schema ([src/core/schema.py](src/core/schema.py))

**Purpose**: Defines methodology rules for valid node types, edge types, and constraints.

**Responsibilities**:
- Load schema from YAML config
- Validate graph elements against schema
- Provide LLM guidance for extraction

**Dependencies**:
- `yaml` (for config loading)
- `typing`

**Dependents**:
- `Extractor` (validates extracted nodes/edges)
- `GraphState` (uses terminal type definitions)
- `InterviewController` (passes to components)

**Key Methods**:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `is_valid_edge(source_type, target_type, relation_type)` | `bool` | Check if edge conforms to schema |
| `is_terminal_type(node_type)` | `bool` | Check if node type is terminal (e.g., value) |
| `get_valid_edge_types(source_type, target_type)` | `List[str]` | Get allowed edge types for node pair |
| `validate_graph(graph)` | `List[str]` | Return list of schema violations |

**Configuration Structure**:
```yaml
schema:
  node_types:
    attribute:
      name: "attribute"
      description: "Product feature or characteristic"
      llm_prompt: "Use for tangible features..."
      examples: ["security", "convenience", "design"]
      is_terminal: false

    value:
      name: "value"
      description: "Personal belief or value"
      llm_prompt: "Use for abstract beliefs..."
      examples: ["family safety", "peace of mind"]
      is_terminal: true

  edge_types:
    - name: "leads_to"
      valid_sources: ["attribute", "functional_consequence"]
      valid_targets: ["functional_consequence", "psychosocial_consequence"]
      llm_prompt: "Use when X causes Y..."
```

**Interaction Pattern**:
```
Extractor extracts node with node_type="attribute"
         │
         └──► Schema.validate_node_type("attribute")
                  │
                  ├──► Check if "attribute" in schema.node_types
                  │       │
                  │       ├─ yes ──► Valid
                  │       └─ no ───► Invalid (filter out)

Extractor extracts edge with relation_type="leads_to"
         │
         └──► Schema.is_valid_edge(source_type, target_type, "leads_to")
                  │
                  └──► Find edge_type definition
                       │
                       ├──► Check source_type in valid_sources
                       │       │
                       │       └──► Check target_type in valid_targets
                       │              │
                       │              ├─ both yes ──► Valid
                       │              └─ either no ──► Invalid
```

---

### 5. History ([src/core/history.py](src/core/history.py))

**Purpose**: Append-only log of interview turns for audit and replay.

**Responsibilities**:
- Record each interview turn
- Provide queries over history (recent questions, strategy usage, etc.)
- Calculate summary statistics

**Dependencies**:
- `Turn` (dataclass)
- `typing`

**Dependents**:
- `InterviewController` (records turns)
- `QuestionGenerator` (checks recent questions for deduplication)
- `Extractor` (passes history for momentum assessment)
- `StrategySelector` (checks recent strategy usage)

**Key Methods**:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `add_turn(turn)` | `None` | Append turn to history |
| `get_recent_turns(n)` | `List[Turn]` | Get last N turns |
| `get_recent_questions(n)` | `List[str]` | Get last N questions |
| `get_strategy_usage()` | `Dict[str, int]` | Count strategy usage |
| `get_total_turns()` | `int` | Count turns |

**Structure**:
```python
@dataclass
class Turn:
    turn_number: int
    question: str
    response: str
    extracted_nodes: List[str]           # Node IDs
    extracted_edges: List[Tuple[str, str]]  # (source_id, target_id) pairs
    strategy_used: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)
```

**Metadata Keys** (common):
- `focus_node_id`: Node ID that was focused
- `focus_element_id`: Element ID that was focused
- `momentum`: "high" | "medium" | "low"
- `extractable`: bool
- `score`: float (strategy score from arbitration)

**Critical Invariants**:
1. **Append-only**: Never modify existing turns
2. **Sequential**: Turn numbers strictly increasing (0, 1, 2, ...)
3. **Complete**: Every turn has question + response (even if empty)

---

### 6. GraphState ([src/core/state.py](src/core/state.py))

**Purpose**: Computed snapshot of graph topology for decision-making.

**Responsibilities**:
- Identify structural issues (isolated, ambiguous nodes)
- Trace active conversation branch
- Find unexplored nodes

**Dependencies**:
- `Graph`
- `Schema`
- `History`
- `typing`

**Dependents**:
- `StrategySelector` (uses for strategy applicability)
- `Strategy` (uses for focus selection)
- `InterviewController` (computes after each turn)

**Key Properties**:

| Property | Type | Computation |
|----------|------|-------------|
| `isolated_nodes` | `List[Node]` | Nodes with 0 edges |
| `ambiguous_nodes` | `List[Node]` | Nodes with `is_ambiguous=True` |
| `terminal_nodes` | `List[Node]` | Nodes with terminal type per schema |
| `unexplored_nodes` | `List[Node]` | Nodes with <2 edges, not recently focused |
| `active_branch` | `List[str]` | Node IDs from most recent node to root |

**Computation Method**:
```python
@classmethod
def compute(cls, graph: Graph, schema: Schema, history: History) -> 'GraphState':
    isolated = cls._identify_isolated_nodes(graph)
    ambiguous = cls._identify_ambiguous_nodes(graph)
    terminal = cls._identify_terminal_nodes(graph, schema)
    unexplored = cls._identify_unexplored_nodes(graph, history)

    most_recent_node_id = cls._get_most_recent_node(graph, history)
    active_branch = cls._trace_branch(graph, most_recent_node_id)

    return cls(
        isolated_nodes=isolated,
        ambiguous_nodes=ambiguous,
        terminal_nodes=terminal,
        unexplored_nodes=unexplored,
        active_branch=active_branch
    )
```

**State Management**:
- **Immutable**: Once computed, never modified
- **Fresh computation**: Recomputed after every turn
- **No caching**: Always derived from graph + schema + history

**Complexity**:
- `isolated_nodes`: O(N + E)
- `ambiguous_nodes`: O(N)
- `terminal_nodes`: O(N)
- `unexplored_nodes`: O(N + H) where H = history size
- `active_branch`: O(E) worst case

---

### 7. CoverageState ([src/core/state.py](src/core/state.py))

**Purpose**: Track how stimulus concept elements are addressed during interview.

**Responsibilities**:
- Map nodes to reference elements
- Track reactions, comprehension for each element
- Identify coverage gaps
- Detect element exhaustion

**Dependencies**:
- `Graph`
- `ReferenceElement` (dataclass)
- `CoverageGap` (dataclass)
- `typing`

**Dependents**:
- `StrategySelector` (uses for strategy applicability)
- `Strategy` (uses for focus selection)
- `InterviewController` (updates after extraction)

**Key Properties**:

| Property | Type | Purpose |
|----------|------|---------|
| `reference_elements` | `Dict[str, ReferenceElement]` | Stimulus elements from concept |
| `element_node_mappings` | `Dict[str, List[str]]` | element_id → [node_ids] |
| `element_reactions` | `Dict[str, str]` | element_id → reaction |
| `element_focus_counts` | `Dict[str, int]` | element_id → focus count |
| `exhausted_elements` | `List[str]` | Elements focused ≥3 times |
| `gaps` | `List[CoverageGap]` | What's still needed |

**Key Methods**:

| Method | Purpose | Complexity |
|--------|---------|------------|
| `update(graph, node_element_mappings)` | Update mappings and recompute gaps | O(E * N²) |
| `record_reaction(element_id, reaction)` | Record respondent reaction | O(1) |
| `record_element_focus(element_id)` | Track focus, detect exhaustion | O(1) |
| `get_uncovered_elements()` | Get elements with gaps | O(G) |
| `_recompute_gaps()` | Compute 4 gap types | O(E * N²) |

**Gap Types**:

| Gap Type | Definition | Requires |
|----------|------------|----------|
| `unmentioned` | No nodes map to element | ≥1 node mapping |
| `no_reaction` | No reaction recorded | Reaction (positive/negative/neutral/...) |
| `no_comprehension` | Not understood | Comprehension confirmation |
| `unconnected` | Element not connected to target | Edge between element nodes and target nodes |

**Gap Recomputation** (most complex operation):
```python
def _recompute_gaps(self):
    gaps = []

    for element_id, element in self.reference_elements.items():
        # Check unmentioned
        if element_id not in self.element_node_mappings:
            gaps.append(CoverageGap(type="unmentioned", element_id=element_id))

        # Check no_reaction
        if element.requires_reaction and element_id not in self.element_reactions:
            gaps.append(CoverageGap(type="no_reaction", element_id=element_id))

        # Check unconnected (O(N²) for each requirement)
        for requirement in element.connection_requirements:
            connected = False
            source_nodes = self.element_node_mappings.get(element_id, [])
            target_nodes = self.element_node_mappings.get(requirement.target_element_id, [])

            for source_id in source_nodes:
                for target_id in target_nodes:
                    if self.graph.get_edge_between(source_id, target_id):
                        connected = True
                        break
                if connected:
                    break

            if not connected:
                gaps.append(CoverageGap(
                    type="unconnected",
                    element_id=element_id,
                    target_element_id=requirement.target_element_id
                ))

    self.gaps = gaps
```

**Critical Invariants**:
1. Gaps always reflect current graph state
2. Exhausted elements never reset (permanent)
3. Element focus counts reset when switching topics

---

### 8. Momentum ([src/core/state.py](src/core/state.py))

**Purpose**: Track respondent engagement level and detect fatigue.

**Responsibilities**:
- Record momentum level after each turn
- Detect fatigue (3 consecutive low momentum turns)
- Provide momentum history for analysis

**Dependencies**: None (standalone dataclass)

**Dependents**:
- `Extractor` (assesses and updates momentum)
- `StrategySelector` (uses for strategy selection)
- `InterviewController` (checks for fatigue in closing logic)

**Structure**:
```python
@dataclass
class Momentum:
    current_level: str                    # "high" | "medium" | "low"
    history: List[MomentumRecord]         # Historical levels
    consecutive_low_count: int = 0        # For fatigue detection
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `update(new_level)` | Record new momentum level, update consecutive count |
| `is_fatigued()` | Check if ≥3 consecutive low momentum turns |
| `get_recent_trend(n)` | Get last N momentum levels |

**Update Logic**:
```python
def update(self, new_level: str):
    self.history.append(MomentumRecord(
        level=new_level,
        timestamp=datetime.now()
    ))
    self.current_level = new_level

    if new_level == "low":
        self.consecutive_low_count += 1
    else:
        self.consecutive_low_count = 0  # Reset on any non-low
```

**Fatigue Detection**:
```python
def is_fatigued(self) -> bool:
    return self.consecutive_low_count >= 3
```

**Critical Invariants**:
1. Consecutive count resets on any non-low momentum
2. Fatigue requires ≥3 consecutive lows (not total lows)
3. History is append-only

---

## Decision Module

### 9. StrategySelector ([src/decision/strategy.py](src/decision/strategy.py))

**Purpose**: Choose the next interview strategy using multi-scorer arbitration.

**Responsibilities**:
- Evaluate all strategies for applicability
- Score applicable strategies using 9 scorers
- Select strategy with highest weighted score
- Determine focus target for selected strategy

**Dependencies**:
- `Strategy`
- `ArbitrationEngine`
- `GraphState`
- `CoverageState`
- `Momentum`
- `History`
- `Graph`
- `Schema`

**Dependents**:
- `InterviewController` (calls `select()` each turn)

**Key Methods**:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `select(graph, graph_state, coverage_state, momentum, history)` | `Tuple[Strategy, FocusTarget]` | Select best strategy and focus |
| `_get_applicable_strategies(...)` | `List[Strategy]` | Filter strategies where `applies()=True` |
| `_score_strategy(strategy, focus, ...)` | `float` | Use arbitration engine to score strategy |

**Selection Process**:
```
1. Filter applicable strategies
   └──► strategy.applies(graph_state, coverage_state, momentum)
         │
         └──► [ensure_coverage, connect_isolate, deepen_branch, ...]

2. For each applicable strategy:
   ├──► Determine focus: strategy.get_focus(graph, graph_state, ...)
   │         │
   │         └──► FocusTarget(node_id="...", element_id="...", ...)
   │
   └──► Score strategy: arbitration_engine.score(strategy, focus, ...)
             │
             └──► Apply 9 scorers, multiply scores: 1.0 * 2.0 * 0.8 * 1.5 = 2.4

3. Select strategy with highest score
   └──► max(scored_strategies, key=lambda s: s.score)
```

**Interaction Pattern**:
```
InterviewController.process_response()
         │
         └──► StrategySelector.select(graph, graph_state, ...)
                  │
                  ├──► For each strategy:
                  │    ├──► strategy.applies(...)
                  │    │        │
                  │    │        └──► Check conditions (e.g., has isolated nodes?)
                  │    │
                  │    ├──► strategy.get_focus(...)
                  │    │        │
                  │    │        └──► Return FocusTarget
                  │    │
                  │    └──► arbitration_engine.score(...)
                  │             │
                  │             └──► Apply 9 scorers
                  │
                  └──► Return (selected_strategy, focus_target)
```

**State Management**:
- **Stateless**: No internal state (pure function of inputs)
- **Thread safety**: Thread-safe (reads only)

---

### 10. Strategy ([src/decision/strategy.py](src/decision/strategy.py))

**Purpose**: Define interview strategy intent, applicability conditions, and focus selection logic.

**Responsibilities**:
- Determine when strategy applies
- Select focus target (node, element, etc.)
- Provide LLM guidance for question generation

**Dependencies**:
- `GraphState`
- `CoverageState`
- `Momentum`
- `Graph`
- `History`

**Dependents**:
- `StrategySelector` (evaluates and selects)
- `QuestionGenerator` (uses intent for generation)

**Structure**:
```python
@dataclass
class Strategy:
    id: str                       # "ensure_coverage"
    intent: str                   # "Address uncovered stimulus elements"
    applies_when: str             # English description of conditions
    suggested_tactics: List[str]  # ["open_element_probe", ...]
    llm_guidance: str             # Instructions for LLM
```

**Key Methods**:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `applies(graph_state, coverage_state, momentum)` | `bool` | Check if strategy is applicable |
| `get_focus(graph, graph_state, coverage_state, ...)` | `FocusTarget` | Select what to focus on |

**Implementation Pattern** (dynamic dispatch):
```python
def applies(self, graph_state, coverage_state, momentum) -> bool:
    method_name = f"_check_{self.id}"
    if hasattr(self, method_name):
        return getattr(self, method_name)(graph_state, coverage_state, momentum)
    return False

def _check_ensure_coverage(self, graph_state, coverage_state, momentum) -> bool:
    return len(coverage_state.gaps) > 0

def _check_connect_isolate(self, graph_state, coverage_state, momentum) -> bool:
    return len(graph_state.isolated_nodes) > 0
```

**Built-in Strategies**:

| Strategy ID | Applies When | Focus |
|------------|--------------|-------|
| `ensure_coverage` | Has coverage gaps | Uncovered element |
| `resolve_ambiguity` | Has ambiguous nodes | Ambiguous node |
| `connect_isolate` | Has isolated nodes | Isolated node |
| `resolve_schema_tension` | Has invalid edges | Invalid edge |
| `deepen_branch` | Active branch exists | Most recent node |
| `explore_breadth` | Has unexplored nodes | Underexplored node |
| `introduce_seed` | No other strategy applies | None (open-ended) |

**Configuration** (from YAML):
```yaml
strategies:
  ensure_coverage:
    intent: "Address stimulus elements that haven't been covered"
    applies_when: "There are coverage gaps (unmentioned, no_reaction, etc.)"
    suggested_tactics:
      - "open_element_probe"
      - "reaction_elicitation"
    llm_guidance: "Ask about the specific element directly..."
```

---

### 11. ArbitrationEngine ([src/decision/arbitration.py](src/decision/arbitration.py))

**Purpose**: Multi-scorer optimization to prevent monotonic interview behavior.

**Responsibilities**:
- Apply 9 independent scorers to each strategy
- Multiply scorer outputs (weighted) to get final score
- Log scoring decisions for debugging

**Dependencies**:
- `StrategyScorer` (abstract base + 9 concrete scorers)
- `Strategy`
- `FocusTarget`
- `History`
- `GraphState`
- `CoverageState`
- `Momentum`

**Dependents**:
- `StrategySelector` (uses for scoring)

**Key Methods**:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `score(strategy, focus, context)` | `float` | Apply all scorers, return weighted product |
| `_apply_scorer(scorer, strategy, focus, context)` | `float` | Apply single scorer |

**Scoring Formula**:
```python
def score(self, strategy, focus, context) -> float:
    final_score = 1.0  # Start with neutral

    for scorer in self.scorers:
        multiplier = scorer.score(strategy, focus, context)
        weight = scorer.weight

        # Apply weighted multiplier
        final_score *= (multiplier ** weight)

        logger.debug(f"[{scorer.name}] {multiplier:.2f}x (weight={weight})")

    logger.info(f"[{strategy.id}] Final score: {final_score:.2f}")
    return final_score
```

**Context Object**:
```python
@dataclass
class ScoringContext:
    graph: Graph
    graph_state: GraphState
    coverage_state: CoverageState
    momentum: Momentum
    history: History
    schema: Schema  # For methodology-agnostic terminal type detection
```

The `schema` field enables scorers to detect terminal node types across different methodologies (MEC, JTBD, etc.) using `schema.is_terminal_type()`, eliminating the need for hardcoded node type lists in scorer implementations.

**Interaction Pattern**:
```
StrategySelector.select()
         │
         └──► For each applicable strategy:
                  │
                  └──► ArbitrationEngine.score(strategy, focus, context)
                           │
                           ├──► RedundancyScorer.score(...)
                           │         │
                           │         └──► Check if question similar to recent
                           │                  │
                           │                  ├─ similar ──► 0.5x penalty
                           │                  └─ different ──► 1.0x neutral
                           │
                           ├──► KnowledgeCeilingScorer.score(...)
                           │         │
                           │         └──► Check if respondent knows topic
                           │                  │
                           │                  ├─ doesn't know ──► 0.0x veto
                           │                  └─ knows ──► 1.0x neutral
                           │
                           ├──► ... (7 more scorers)
                           │
                           └──► Multiply all multipliers:
                                     final = 1.0 * 0.5 * 1.0 * 2.0 * ... = 1.8
```

**State Management**:
- **Stateless**: No internal state
- **Thread safety**: Thread-safe (reads only)

---

### 12. StrategyScorer (Abstract Base) ([src/decision/arbitration.py](src/decision/arbitration.py))

**Purpose**: Abstract base class for scoring logic.

**Responsibilities**:
- Define scoring interface
- Provide weight for importance tuning

**Structure**:
```python
class StrategyScorer(ABC):
    name: str
    weight: float = 1.0  # Importance multiplier

    @abstractmethod
    def score(self, strategy, focus, context) -> float:
        """Return multiplier: 1.0 = neutral, <1.0 = penalty, >1.0 = boost"""
        pass
```

**9 Concrete Scorers**:

#### 12.1. RedundancyScorer
**Purpose**: Penalize questions similar to recent ones.

**Logic**:
```python
def score(self, strategy, focus, context) -> float:
    # Generate hypothetical question
    hypothetical_question = self._generate_question(strategy, focus)

    # Check similarity to last 6 questions
    recent_questions = context.history.get_recent_questions(6)
    for recent in recent_questions:
        similarity = self._jaccard_similarity(hypothetical_question, recent)
        if similarity > 0.85:
            return 0.5  # 50% penalty

    return 1.0  # Neutral
```

**Weight**: 1.0

---

#### 12.2. KnowledgeCeilingScorer
**Purpose**: Stop probing topics respondent doesn't know.

**Logic**:
```python
def score(self, strategy, focus, context) -> float:
    # Check if respondent said "I don't know" about focus
    if focus.node_id:
        node = context.graph.get_node(focus.node_id)
        if node.metadata.get("knowledge_ceiling"):
            return 0.0  # Veto (complete penalty)

    return 1.0  # Neutral
```

**Weight**: 1.0

---

#### 12.3. MomentumAlignmentScorer
**Purpose**: Match strategy intensity to engagement level.

**Logic**:
```python
def score(self, strategy, focus, context) -> float:
    momentum = context.momentum.current_level

    # High momentum → boost depth strategies
    if momentum == "high":
        if strategy.id == "deepen_branch":
            return 1.3  # depth_boost (reduced from 1.5)

    # Low momentum → boost breadth/seeding
    elif momentum == "low":
        if strategy.id == "explore_breadth":
            return 1.1  # breadth_boost (reduced from 1.5)
        if strategy.id == "deepen_branch":
            return 0.2  # depth_penalty (reduced from 0.5)

    return 1.0  # Neutral
```

**Parameters** (from interview_logic.yaml):
- `breadth_boost`: 1.1 (reduced from 1.5 to prevent breadth dominance)
- `depth_penalty`: 0.2 (reduced from 0.5 to enable more depth exploration)
- `depth_boost`: 1.3 (for high momentum)

**Weight**: 1.0

---

#### 12.4. RecencyDiversityScorer
**Purpose**: Promote strategy variety (avoid 3+ consecutive same strategy).

**Logic**:
```python
def score(self, strategy, focus, context) -> float:
    recent_strategies = [turn.strategy_used for turn in context.history.get_recent_turns(3)]

    # Count consecutive uses of this strategy
    consecutive_count = 0
    for recent_strategy in reversed(recent_strategies):
        if recent_strategy == strategy.id:
            consecutive_count += 1
        else:
            break

    # Penalize if used 2+ times consecutively
    if consecutive_count >= 2:
        return 0.6  # 40% penalty

    return 1.0  # Neutral
```

**Weight**: 0.8

---

#### 12.5. VerticalLadderingScorer
**Purpose**: Boost depth exploration toward terminal nodes (methodology-agnostic).

**Logic**:
```python
def score(self, strategy, focus, context) -> float:
    if strategy.id == "deepen_branch":
        # Check if active branch is approaching terminal nodes
        active_branch = context.graph_state.active_branch
        if active_branch:
            branch_nodes = [context.graph.get_node(nid) for nid in active_branch]
            terminal_count = sum(1 for n in branch_nodes
                                if context.schema.is_terminal_type(n.node_type))

            # Boost if close to terminals
            if terminal_count > 0:
                return 1.8  # 80% boost

    return 1.0  # Neutral
```

**Methodology-Agnostic Design**: Uses `schema.is_terminal_type()` to detect terminal nodes. Works with Means-End Chain (terminal = `value`), Jobs-to-Be-Done (terminal = `constraint`), and any future methodology that defines terminal types in schema YAML.

**Weight**: 1.0

---

#### 12.6. BranchHealthScorer
**Purpose**: Detect stale conversation threads, force topic switch.

**Logic**:
```python
def score(self, strategy, focus, context) -> float:
    # Check if same strategy used stale_threshold+ consecutive turns
    recent_strategies = [turn.strategy_used for turn in context.history.get_recent_turns(4)]

    # Count consecutive uses of current branch strategy
    consecutive_count = sum(1 for s in recent_strategies if s == strategy.id)

    if consecutive_count >= 4:  # stale_threshold (increased from 2)
        # Boost breadth to switch topics
        if strategy.id == "explore_breadth":
            return 1.2  # breadth_boost (reduced from 2.0)
        # Penalize continuing same strategy
        elif strategy.id == recent_strategies[0]:
            return 0.15  # depth_penalty (reduced from 0.7)
        # Also penalize connect_isolate on stale
        elif strategy.id == "connect_isolate":
            return 0.5  # connect_isolate_penalty

    return 1.0  # Neutral
```

**Parameters** (from interview_logic.yaml):
- `stale_threshold`: 4 (increased from 2 - allow longer depth exploration)
- `breadth_boost`: 1.2 (reduced from 1.8 - less aggressive breadth forcing)
- `depth_penalty`: 0.15 (reduced from 0.3 - softer depth suppression)
- `severe_stale_threshold`: 4
- `connect_isolate_penalty`: 0.5 (penalize connect_isolate on stale branches)

**Weight**: 1.0

---

#### 12.7. CoverageQualityScorer
**Purpose**: Prioritize first-time element coverage, stop drilling exhausted elements.

**Logic**:
```python
def score(self, strategy, focus, context) -> float:
    if strategy.id == "ensure_coverage" and focus.element_id:
        # Check if element has been focused before
        if context.coverage_state.element_focus_counts.get(focus.element_id, 0) == 0:
            return 2.5  # first_touch_boost: 150% boost for first-time coverage

        # Penalize if exhausted (probed multiple times without new edges)
        if focus.element_id in context.coverage_state.exhausted_elements:
            return 1.2  # exhaustion_penalty (INVERSE: >1.0 boosts, <1.0 penalizes)
                        # Note: This is actually a DETERRENT multiplier in the config
                        # Implementation uses: score /= exhaustion_penalty
                        # So 1.2 means divide by 1.2 = 0.83x penalty

    return 1.0  # Neutral
```

**Parameters** (from interview_logic.yaml):
- `first_touch_boost`: 2.5 (strong priority for new elements)
- `exhaustion_threshold`: 2 (turns probing without new edges)
- `exhaustion_penalty`: 1.2 (increased from 0.15 - STOP drilling exhausted elements)
  - **Note**: In implementation, this is a divisor: `score /= exhaustion_penalty`
  - So 1.2 means score is divided by 1.2, resulting in ~0.83x effective penalty
  - Higher value = stronger deterrent

**Weight**: 1.0

---

#### 12.8. SchemaTensionReadinessScorer
**Purpose**: Time relationship clarification appropriately.

**Logic**:
```python
def score(self, strategy, focus, context) -> float:
    if strategy.id == "resolve_schema_tension":
        # Only resolve if coverage mostly complete
        coverage_percent = self._compute_coverage_percent(context.coverage_state)

        if coverage_percent < 50:
            return 0.5  # 50% penalty (too early)
        elif coverage_percent > 70:
            return 1.5  # 50% boost (good timing)

    return 1.0  # Neutral
```

**Weight**: 0.8

---

#### 12.9. ReflectionModeScorer
**Purpose**: Trigger conclusion phase when appropriate (methodology-agnostic).

**Logic**:
```python
def score(self, strategy, focus, context) -> float:
    # Check if should enter reflection mode
    if not self._should_enter_reflection_mode(context):
        return 1.0

    # Penalize depth strategies (interview should wrap up)
    if strategy.id in ["deepen_branch", "connect_isolate"]:
        return 0.2  # depth_penalty

    # Boost reflection strategies (encourage conclusion)
    if strategy.id in ["synthesize", "summarize"]:
        return 2.0  # reflection_boost

    # Slight boost for breadth (explore remaining areas)
    if strategy.id == "explore_breadth":
        return 1.3

    return 1.0  # Neutral

def _should_enter_reflection_mode(self, context) -> bool:
    # Condition 1: Coverage complete
    if context.coverage_state.gaps:
        return False

    # Condition 2: No new nodes for N recent turns
    if not self._no_new_nodes_recently(context):
        return False

    # Condition 3: Terminal nodes exist (schema-agnostic)
    if not self._has_terminal_nodes(context):
        return False

    return True

def _has_terminal_nodes(self, context) -> bool:
    """Check terminal nodes using schema.is_terminal_type() (methodology-agnostic)."""
    terminal_count = 0
    for node in context.graph.nodes.values():
        if node.node_type and context.schema.is_terminal_type(node.node_type):
            terminal_count += 1
    return terminal_count >= self.min_terminal_nodes
```

**Parameters** (from interview_logic.yaml):
- `no_new_nodes_threshold`: 3 (turns without extraction before reflection)
- `min_terminal_nodes`: 0 (changed from 1 - schema-agnostic, allows reflection without terminal nodes)
- `depth_penalty`: 0.2 (strong penalty for depth strategies in reflection mode)
- `reflection_boost`: 2.0 (strong boost for reflection strategies)

**Methodology-Agnostic Design**: Uses `schema.is_terminal_type()` to detect terminal nodes. Works with Means-End Chain (terminal = `value`), Jobs-to-Be-Done (terminal = `constraint`), and any future methodology. The `min_terminal_nodes: 0` setting makes reflection mode schema-agnostic by not requiring any terminal nodes (though they boost reflection mode if present).

**Weight**: 1.0

---

### 13. Extractor ([src/decision/extraction.py](src/decision/extraction.py))

**Purpose**: Extract graph deltas (nodes + edges) from respondent text using LLM.

**Responsibilities**:
- Assess if response contains extractable content
- Extract structured nodes/edges via LLM function calling
- Map extracted nodes to stimulus elements
- Assess respondent momentum level
- Validate extractions against schema

**Dependencies**:
- `LLMManager`
- `Graph`
- `Schema`
- `CoverageState`
- `History`
- `ExtractionResult` (dataclass)

**Dependents**:
- `InterviewController` (calls `extract()` each turn)

**Key Methods**:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `assess_extractability(response)` | `bool` | Check if response has extractable content |
| `extract(response, graph, coverage_state)` | `ExtractionResult` | Extract nodes + edges from response |
| `assess_momentum(response, history)` | `str` | Return "high" \| "medium" \| "low" |
| `_find_existing_node_semantic(label, node_type, graph)` | `Optional[str]` | Three-tier semantic deduplication (exact → Jaccard → embeddings) |
| `_lemmatize(text)` | `str` | Remove common suffixes for normalization |
| `_expand_synonyms(text)` | `str` | Expand with domain-specific synonyms |
| `_jaccard_similarity(set1, set2)` | `float` | Compute Jaccard similarity for word sets |
| `_semantic_similarity(text1, text2)` | `float` | Compute cosine similarity using sentence embeddings |

**Extraction Pipeline**:
```
1. Extractability Check
   └──► LLM call: "Does this contain extractable content?"
         │
         └──► {"extractable": bool, "reason": str}
              │
              ├─ false ──► Return empty extraction
              └─ true ──► Continue to extraction

2. Structured Extraction
   └──► LLM call with EXTRACTION_TOOL (function calling)
         │
         └──► {
                "nodes": [{"label": "...", "node_type": "...", ...}],
                "edges": [{"source_label": "...", "target_label": "...", ...}]
              }
              │
              └──► Parse JSON

3. Validation + Semantic Deduplication
   ├──► For each node:
   │    ├──► Validate node_type in schema
   │    ├──► Validate element_mapping (if present)
   │    ├──► Validate reaction enum
   │    ├──► Check for semantic duplicates:
   │    │    ├──► Tier 1: Exact match (hash lookup, <1ms)
   │    │    ├──► Tier 2: Enhanced Jaccard (lemmatize + synonyms, 1-5ms)
   │    │    └──► Tier 3: Embeddings (sentence-transformers, 10-50ms cached)
   │    └──► Create Node object or reuse existing ID
   │
   └──► For each edge:
        ├──► Find source/target nodes by label
        ├──► Validate relation_type in schema
        ├──► Validate schema.is_valid_edge(...)
        └──► Create Edge object

4. Element Mapping
   └──► For each node with element_mapping:
         └──► Link to reference element in coverage_state

5. Momentum Assessment (Improved 2025-12-12)
   └──► LLM call with explicit NEUTRAL criteria:
         - NEUTRAL: Thinking-aloud hedging WITH elaboration ("I guess... [explanation]")
         - LOW: Vague hedging WITHOUT elaboration ("I dunno")
         - Balancing rules: elaboration outweighs hedging
         - History window: 5 turns (increased from 3)
         │
         └──► {"level": "high" | "medium" | "low", "reason": str}
```

**Interaction Pattern**:
```
InterviewController.process_response(response_text)
         │
         ├──► Extractor.assess_extractability(response_text)
         │         │
         │         └──► LLMManager.complete("extractability_check", ...)
         │                  │
         │                  └──► {"extractable": true}
         │
         ├──► Extractor.extract(response_text, graph, coverage_state)
         │         │
         │         ├──► LLMManager.complete("graph_extraction", ...)
         │         │         │
         │         │         └──► Function call JSON
         │         │
         │         ├──► Validate nodes/edges against schema
         │         │
         │         └──► Return ExtractionResult(nodes=[...], edges=[...], ...)
         │
         └──► Extractor.assess_momentum(response_text, history)
                   │
                   └──► LLMManager.complete("momentum_assessment", ...)
                             │
                             └──► {"level": "medium"}
```

**State Management**:
- **Stateless**: No internal state (except LLMManager reference)
- **Thread safety**: Thread-safe (LLMManager handles concurrency)

---

## Generation Module

### 14. QuestionGenerator ([src/generation/generator.py](src/generation/generator.py))

**Purpose**: Generate natural language questions based on strategy intent and focus.

**Responsibilities**:
- Generate opening question (turn 0)
- Generate follow-up questions based on strategy + focus
- Deduplicate questions (check similarity to recent questions)
- Adjust temperature based on momentum

**Dependencies**:
- `LLMManager`
- `Strategy`
- `FocusTarget`
- `History`
- `Graph`
- `Momentum`

**Dependents**:
- `InterviewController` (calls `generate()` each turn)

**Key Methods**:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `generate_opening(concept_text, history)` | `str` | Generate opening question |
| `generate(strategy, focus, graph, history, momentum)` | `str` | Generate follow-up question |
| `_is_duplicate(question, history)` | `bool` | Check if question similar to recent ones |

**Generation Process**:
```
1. Build prompt context
   ├──► Strategy intent: "Address uncovered stimulus elements"
   ├──► Focus details: "Element: security"
   ├──► Recent turns: Last 3 Q&A pairs
   └──► Momentum: "high" (use temperature=0.7)

2. Generate question
   └──► LLMManager.complete("question_generation", system_prompt, user_prompt)
         │
         └──► "How important is security to you when choosing a product?"

3. Check for duplication
   └──► _is_duplicate(generated_question, history)
         │
         ├──► Compute Jaccard similarity with last 6 questions
         │         │
         │         └──► similarity = 0.9 (too similar!)
         │
         ├─ duplicate ──► Retry with higher temperature (0.8)
         │                      │
         │                      └──► Generate again (up to 2 retries)
         │
         └─ unique ──► Return question
```

**Deduplication Logic**:
```python
def _is_duplicate(self, question: str, history: History, threshold: float = 0.85) -> bool:
    recent_questions = history.get_recent_questions(6)
    question_words = set(question.lower().split())

    for recent in recent_questions:
        recent_words = set(recent.lower().split())

        # Jaccard similarity
        similarity = len(question_words & recent_words) / len(question_words | recent_words)

        if similarity >= threshold:
            logger.warning(f"Duplicate question detected (similarity={similarity:.2f})")
            return True

    return False
```

**Temperature Adjustment**:
```python
def _get_temperature(self, momentum: Momentum) -> float:
    if momentum.current_level == "high":
        return 0.7  # More varied questions
    elif momentum.current_level == "medium":
        return 0.6
    else:  # low
        return 0.5  # More conservative questions
```

**Interaction Pattern**:
```
InterviewController chooses strategy + focus
         │
         └──► QuestionGenerator.generate(strategy, focus, ...)
                  │
                  ├──► Build system prompt from strategy.intent + strategy.llm_guidance
                  │
                  ├──► Build user prompt with focus details + recent turns
                  │
                  ├──► LLMManager.complete("question_generation", ...)
                  │         │
                  │         └──► "How does security affect your decision?"
                  │
                  ├──► _is_duplicate(question, history)
                  │         │
                  │         ├─ duplicate ──► Retry (up to 2 times)
                  │         └─ unique ──► Return question
                  │
                  └──► Return question
```

**State Management**:
- **Stateless**: No internal state (except LLMManager reference)
- **Thread safety**: Thread-safe

---

## Utils Module

### 15. LLMManager ([src/utils/llm_manager.py](src/utils/llm_manager.py))

**Purpose**: Unified interface for multiple LLM providers with cost tracking.

**Responsibilities**:
- Create provider-specific clients (Anthropic, OpenAI, Kimi, DeepSeek)
- Normalize API responses across providers
- Track token usage and costs
- Handle retries with exponential backoff
- Manage timeouts

**Dependencies**:
- `anthropic` (Anthropic SDK)
- `openai` (OpenAI SDK)
- Provider config from `llm_config.yaml`

**Dependents**:
- `Extractor`
- `QuestionGenerator`
- Any component needing LLM calls

**Key Methods**:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `complete(task, system_prompt, user_prompt, **kwargs)` | `LLMResponse` | Make LLM call for task |
| `get_provider_for_task(task)` | `str` | Get configured provider for task |
| `get_total_cost()` | `float` | Get cumulative cost for session |
| `_get_client(provider)` | `Client` | Get provider-specific client |

**Provider Architecture**:
```
LLMManager
    │
    ├──► Anthropic Client (claude-3-5-sonnet, claude-3-opus, ...)
    ├──► OpenAI Client (gpt-4, gpt-3.5-turbo, ...)
    ├──► Kimi Client (kimi-k2-turbo, kimi-k2-0905-preview, ...)
    └──► DeepSeek Client (deepseek-chat, deepseek-coder, ...)
```

**Task-Provider Mapping** (from config):
```yaml
graph_extraction_model: "deepseek"
question_generation_model: "kimi"
extractability_check_model: "deepseek"
momentum_assessment_model: "deepseek"
```

**LLM Call Flow**:
```
Component calls LLMManager.complete("graph_extraction", system_prompt, user_prompt)
         │
         ├──► Get provider: "deepseek"
         │
         ├──► Get task config:
         │         temperature: 0.3
         │         max_tokens: 1000
         │         timeout: 15s
         │
         ├──► Create client (if not cached)
         │
         ├──► Make API call with retry logic
         │         │
         │         ├──► Success
         │         │         │
         │         │         └──► Parse response
         │         │
         │         └──► Failure (timeout, rate limit, ...)
         │                   │
         │                   └──► Retry with exponential backoff (3 attempts)
         │
         ├──► Normalize response to LLMResponse format
         │         │
         │         └──► LLMResponse(
         │                  content="...",
         │                  tool_use={"name": "...", "arguments": "..."},
         │                  input_tokens=150,
         │                  output_tokens=300,
         │                  model="deepseek-chat"
         │              )
         │
         ├──► Calculate cost
         │         input_cost = 150 * 0.14 / 1_000_000 = $0.000021
         │         output_cost = 300 * 0.28 / 1_000_000 = $0.000084
         │         total = $0.000105
         │
         ├──► Update cumulative cost
         │
         └──► Return LLMResponse
```

**Cost Tracking**:
```python
def complete(self, task, system_prompt, user_prompt, **kwargs) -> LLMResponse:
    response = self._make_api_call(...)

    # Get pricing from config
    provider_config = self.config.providers[provider]
    model_config = provider_config.models[task]

    # Calculate cost
    input_cost = response.input_tokens * model_config.cost_input / 1_000_000
    output_cost = response.output_tokens * model_config.cost_output / 1_000_000

    # Track
    self.total_cost += input_cost + output_cost
    self.total_input_tokens += response.input_tokens
    self.total_output_tokens += response.output_tokens

    return response
```

**State Management**:
- **Mutable**: Tracks cumulative costs and token counts
- **Thread safety**: Not thread-safe (assumes single-threaded usage)
- **Client caching**: Clients cached per provider (not recreated)

---

### 16. InterviewLogger ([src/utils/logger.py](src/utils/logger.py))

**Purpose**: Session-level logging for interview tracking and debugging.

**Responsibilities**:
- Log each turn (question, response, extraction, strategy)
- Log LLM usage (tokens, costs)
- Generate session summary
- Write logs to file

**Dependencies**:
- `logging` (standard library)
- `json` (for structured logging)

**Dependents**:
- `InterviewController` (logs each turn)
- `LLMManager` (logs API calls)

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `log_turn(turn_number, question, response, strategy, ...)` | Record interview turn |
| `log_llm_usage(task, provider, tokens, cost)` | Record LLM API call |
| `get_session_summary()` | Generate summary stats |

**State Management**:
- **Mutable**: Appends to log file
- **Thread safety**: Not thread-safe

---

### 17. ConceptParser ([src/utils/concept_parser.py](src/utils/concept_parser.py))

**Purpose**: Parse stimulus concept text to extract reference elements.

**Responsibilities**:
- Extract elements from structured concept text
- Parse element requirements (reaction, comprehension, connections)
- Create ReferenceElement objects

**Dependencies**: None (pure parser)

**Dependents**:
- `InterviewController` (parses concept at initialization)

**Parsing Format**:
```
Concept: Mobile Banking App

Elements:
- Security [reaction, comprehension]
  Requires: Trust, Peace of Mind
- Convenience [reaction]
- Design [reaction, comprehension]
  Requires: Usability
```

**Output**:
```python
[
    ReferenceElement(
        id="security",
        label="Security",
        requires_reaction=True,
        requires_comprehension=True,
        connection_requirements=[
            ConnectionRequirement(target_element_id="trust"),
            ConnectionRequirement(target_element_id="peace_of_mind")
        ]
    ),
    ...
]
```

---

## UI Module

### 18. GradioApp ([src/ui/gradio_app.py](src/ui/gradio_app.py))

**Purpose**: Web interface for conducting interviews.

**Responsibilities**:
- Render chat interface
- Handle user input/output
- Display interview progress (turns, coverage, graph visualization)
- Provide controls (start, reset, export)

**Dependencies**:
- `gradio` (UI framework)
- `InterviewController`
- `Graph` (for visualization)

**Dependents**: None (entry point)

**Key Components**:

| Component | Type | Purpose |
|-----------|------|---------|
| `chat_interface` | Chatbot | Display Q&A turns |
| `concept_input` | Textbox | Stimulus concept text |
| `start_button` | Button | Initialize interview |
| `reset_button` | Button | Clear session |
| `progress_display` | Markdown | Show turns, coverage % |
| `graph_viz` | Plot | Visualize knowledge graph |

**Interaction Flow**:
```
User enters concept text
         │
         └──► Click "Start Interview"
                  │
                  └──► Initialize InterviewController
                           │
                           └──► Generate opening question
                                    │
                                    └──► Display in chat

User types response
         │
         └──► Submit
                  │
                  └──► InterviewController.process_response(response)
                           │
                           ├──► Extract graph delta
                           ├──► Select strategy
                           ├──► Generate question
                           └──► Update UI (chat, progress, graph)
```

---

## Controller

### 19. InterviewController ([src/controller.py](src/controller.py))

**Purpose**: Main orchestrator for interview loop.

**Responsibilities**:
- Initialize interview (parse concept, create graph, etc.)
- Process each response (extract, select strategy, generate question)
- Maintain interview state
- Determine when to close interview
- Provide session summary

**Dependencies**:
- ALL core, decision, generation, utils modules

**Dependents**:
- `GradioApp` (uses controller)

**Key Methods**:

| Method | Return Type | Purpose |
|--------|-------------|---------|
| `initialize(concept_text, schema_file)` | `str` | Setup interview, return opening question |
| `process_response(response_text)` | `str` | Process response, return next question |
| `should_close()` | `bool` | Check if interview should end |
| `get_session_summary()` | `Dict` | Get interview statistics |

**Interview Loop** (detailed):
```python
def process_response(self, response_text: str) -> str:
    # 1. Log user response
    self.logger.log_response(self.turn_number, response_text)

    # 2. Check extractability
    extractable = self.extractor.assess_extractability(response_text)

    if not extractable:
        self.logger.log_info("Response not extractable, skipping extraction")

    # 3. Extract graph delta
    if extractable:
        extraction_result = self.extractor.extract(
            response_text,
            self.graph,
            self.coverage_state
        )

        # 4. Update graph
        for node in extraction_result.nodes:
            self.graph.add_node(node)

        for edge in extraction_result.edges:
            self.graph.add_edge(edge)

        # 5. Update coverage state
        self.coverage_state.update(
            self.graph,
            extraction_result.node_element_mappings
        )

        for element_id, reaction in extraction_result.element_reactions.items():
            self.coverage_state.record_reaction(element_id, reaction)

    # 6. Assess momentum
    momentum_level = self.extractor.assess_momentum(response_text, self.history)
    self.momentum.update(momentum_level)

    # 7. Compute graph state
    graph_state = GraphState.compute(self.graph, self.schema, self.history)

    # 8. Check if should close
    if self.should_close():
        return self._generate_closing_message()

    # 9. Select strategy
    strategy, focus = self.strategy_selector.select(
        self.graph,
        graph_state,
        self.coverage_state,
        self.momentum,
        self.history
    )

    # 10. Generate question
    question = self.question_generator.generate(
        strategy,
        focus,
        self.graph,
        self.history,
        self.momentum
    )

    # 11. Record turn
    turn = Turn(
        turn_number=self.turn_number,
        question=question,
        response=response_text,
        extracted_nodes=[n.id for n in extraction_result.nodes],
        extracted_edges=[(e.source_id, e.target_id) for e in extraction_result.edges],
        strategy_used=strategy.id,
        timestamp=datetime.now(),
        metadata={
            "focus_node_id": focus.node_id,
            "focus_element_id": focus.element_id,
            "momentum": momentum_level,
            "extractable": extractable
        }
    )
    self.history.add_turn(turn)

    # 12. Increment turn counter
    self.turn_number += 1

    # 13. Log and return
    self.logger.log_turn(turn)
    return question
```

**Closing Logic**:
```python
def should_close(self) -> bool:
    # Condition 1: Max turns reached
    if self.turn_number >= self.config.max_turns:
        return True

    # Condition 2: Fatigue + adequate coverage
    if self.momentum.is_fatigued():
        coverage_percent = self._compute_coverage_percent()
        if coverage_percent >= 60:
            return True

    # Condition 3: Complete coverage + no unexplored + low momentum
    if len(self.coverage_state.gaps) == 0:
        graph_state = GraphState.compute(self.graph, self.schema, self.history)
        if len(graph_state.unexplored_nodes) == 0:
            if self.momentum.current_level == "low":
                return True

    return False
```

**State Management**:
- **Mutable**: Maintains all interview state
- **Thread safety**: Not thread-safe
- **Persistence**: State not persisted (in-memory only)

---

## Component Dependency Graph

### Full Dependency Map

```
┌─────────────────────────────────────────────────────────────┐
│                   InterviewController                       │
└────────┬──────────────────────────────────────────┬─────────┘
         │                                          │
    ┌────▼────┐                                ┌────▼────┐
    │  Graph  │                                │ History │
    └────┬────┘                                └────┬────┘
         │                                          │
    ┌────▼────┐                                     │
    │  Node   │                                     │
    │  Edge   │                                     │
    └─────────┘                                     │
         │                                          │
    ┌────▼────┐                                     │
    │ Schema  │                                     │
    └────┬────┘                                     │
         │                                          │
    ┌────▼────────────────────┐                    │
    │     GraphState          │◄───────────────────┘
    │   (compute from graph)  │
    └─────────────────────────┘
         │
    ┌────▼────────────────────┐
    │   CoverageState         │
    │ (element tracking)      │
    └─────────────────────────┘
         │
    ┌────▼────────────────────┐
    │     Momentum            │
    │ (engagement tracking)   │
    └─────────────────────────┘
         │
         │
    ┌────▼──────────────────────────────────────────┐
    │           StrategySelector                    │
    └────┬──────────────────────────────────────────┘
         │
    ┌────▼────┐          ┌────────────────┐
    │Strategy │          │ Arbitration    │
    └─────────┘          │    Engine      │
                         └────┬───────────┘
                              │
                         ┌────▼───────────┐
                         │  9 Scorers     │
                         │  (Red, KC, MA, │
                         │   RD, VL, BH,  │
                         │   CQ, STR, RM) │
                         └────────────────┘
         │
    ┌────▼────────────────┐
    │    Extractor        │
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │ QuestionGenerator   │
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │    LLMManager       │
    └────┬────────────────┘
         │
    ┌────▼────────────────────────────┐
    │  Provider Clients               │
    │  (Anthropic, OpenAI, Kimi,      │
    │   DeepSeek)                     │
    └─────────────────────────────────┘
```

### Dependency Table

| Component | Direct Dependencies | Depends On Me |
|-----------|-------------------|---------------|
| **Graph** | Node, Edge | InterviewController, GraphState, CoverageState, StrategySelector |
| **Node** | - | Graph, all consumers |
| **Edge** | - | Graph, all consumers |
| **Schema** | - | Extractor, GraphState, StrategySelector, InterviewController |
| **History** | Turn | InterviewController, GraphState, QuestionGenerator, Extractor, StrategySelector |
| **GraphState** | Graph, Schema, History | StrategySelector, Strategy, InterviewController |
| **CoverageState** | Graph, ReferenceElement | StrategySelector, Strategy, Extractor, InterviewController |
| **Momentum** | - | Extractor, StrategySelector, QuestionGenerator, InterviewController |
| **StrategySelector** | Strategy, ArbitrationEngine, GraphState, CoverageState, Momentum | InterviewController |
| **Strategy** | GraphState, CoverageState, Momentum | StrategySelector, ArbitrationEngine |
| **ArbitrationEngine** | StrategyScorer (9 subclasses) | StrategySelector |
| **Extractor** | LLMManager, Graph, Schema, CoverageState | InterviewController |
| **QuestionGenerator** | LLMManager, Strategy, FocusTarget, History, Momentum | InterviewController |
| **LLMManager** | Provider SDKs (anthropic, openai) | Extractor, QuestionGenerator |
| **InterviewLogger** | - | InterviewController, LLMManager |
| **ConceptParser** | - | InterviewController |
| **InterviewController** | ALL | GradioApp |
| **GradioApp** | InterviewController, gradio | - (entry point) |

---

## Interaction Patterns

### Pattern 1: Request-Response (LLM Calls)

**Components**: Extractor/QuestionGenerator → LLMManager → Provider

**Flow**:
```
Component needs LLM response
         │
         └──► Build prompt (system + user)
                  │
                  └──► LLMManager.complete(task, system_prompt, user_prompt)
                           │
                           └──► Get provider for task
                                    │
                                    └──► Make API call with retry logic
                                             │
                                             ├──► Success → Return LLMResponse
                                             └──► Failure → Retry (3x) or raise
```

**Characteristics**:
- Synchronous (blocking)
- Retries with exponential backoff
- Timeout handling
- Cost tracking

---

### Pattern 2: Compute-Query (State Computation)

**Components**: InterviewController → GraphState/CoverageState

**Flow**:
```
Graph modified (node/edge added)
         │
         └──► GraphState.compute(graph, schema, history)
                  │
                  ├──► Analyze topology (isolated, ambiguous, terminal, unexplored)
                  ├──► Trace active branch
                  └──► Return GraphState (immutable snapshot)

Coverage updated
         │
         └──► CoverageState.update(graph, node_element_mappings)
                  │
                  ├──► Update mappings
                  └──► _recompute_gaps() (O(E*N²))
```

**Characteristics**:
- Immutable outputs
- Fresh computation (no caching)
- Expensive for large graphs

---

### Pattern 3: Strategy Selection (Multi-Scorer Arbitration)

**Components**: InterviewController → StrategySelector → ArbitrationEngine → 9 Scorers

**Flow**:
```
Need to select strategy
         │
         └──► StrategySelector.select(graph, graph_state, coverage_state, momentum, history)
                  │
                  ├──► Filter applicable strategies
                  │         │
                  │         └──► [ensure_coverage, connect_isolate, deepen_branch, ...]
                  │
                  ├──► For each applicable strategy:
                  │    ├──► Determine focus: strategy.get_focus(...)
                  │    └──► Score: arbitration_engine.score(strategy, focus, context)
                  │              │
                  │              └──► Apply 9 scorers:
                  │                   ├──► RedundancyScorer: 0.5x
                  │                   ├──► KnowledgeCeilingScorer: 1.0x
                  │                   ├──► MomentumAlignmentScorer: 1.5x
                  │                   ├──► RecencyDiversityScorer: 1.0x
                  │                   ├──► VerticalLadderingScorer: 1.8x
                  │                   ├──► BranchHealthScorer: 1.0x
                  │                   ├──► CoverageQualityScorer: 2.5x
                  │                   ├──► SchemaTensionReadinessScorer: 1.0x
                  │                   └──► ReflectionModeScorer: 1.0x
                  │                        │
                  │                        └──► final_score = 1.0 * 0.5 * 1.0 * 1.5 * ... = 3.375
                  │
                  └──► Select strategy with max(final_score)
                           │
                           └──► Return (strategy, focus)
```

**Characteristics**:
- Multi-dimensional evaluation
- Multiplicative scoring (amplification/dampening)
- Emergent behavior from scorer interactions

---

### Pattern 4: Extract-Validate-Update (Graph Building)

**Components**: InterviewController → Extractor → Graph → CoverageState

**Flow**:
```
User response received
         │
         └──► Extractor.extract(response, graph, coverage_state)
                  │
                  ├──► LLM function calling: extract_graph_elements
                  │         │
                  │         └──► {"nodes": [...], "edges": [...]}
                  │
                  ├──► Validate against schema
                  │         │
                  │         ├──► node_type in schema.node_types?
                  │         ├──► relation_type in schema.edge_types?
                  │         └──► schema.is_valid_edge(source_type, target_type, relation_type)?
                  │
                  ├──► Filter invalid nodes/edges
                  │
                  └──► Return ExtractionResult(nodes=[...], edges=[...], node_element_mappings={...})

Controller receives ExtractionResult
         │
         ├──► For each node: graph.add_node(node)
         │         │
         │         └──► Check if label exists (de-dup)
         │
         ├──► For each edge: graph.add_edge(edge)
         │         │
         │         └──► Validate source/target exist
         │
         └──► coverage_state.update(graph, node_element_mappings)
                  │
                  └──► _recompute_gaps()
```

**Characteristics**:
- LLM extraction with validation
- De-duplication on add
- Gap recomputation after update

---

### Pattern 5: Generate-Deduplicate-Return (Question Generation)

**Components**: InterviewController → QuestionGenerator → LLMManager

**Flow**:
```
Need to generate question
         │
         └──► QuestionGenerator.generate(strategy, focus, graph, history, momentum)
                  │
                  ├──► Build prompt:
                  │    ├──► Strategy intent + llm_guidance
                  │    ├──► Focus details (node label, element text, etc.)
                  │    └──► Recent 3 turns (context)
                  │
                  ├──► LLMManager.complete("question_generation", system_prompt, user_prompt)
                  │         │
                  │         └──► "How does security affect your decision?"
                  │
                  ├──► _is_duplicate(question, history)
                  │         │
                  │         ├──► Compute Jaccard similarity with last 6 questions
                  │         │         │
                  │         │         └──► similarity = 0.9 (too similar!)
                  │         │
                  │         └─ duplicate ──► Retry with higher temperature (0.8)
                  │                              │
                  │                              └──► Generate again (up to 2 retries)
                  │
                  └──► Return question
```

**Characteristics**:
- Contextual prompt building
- Similarity-based deduplication
- Retry with temperature increase

---

## State Management

### State Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                  InterviewController                        │
│                  (Central State Manager)                    │
└────────┬────────────────────────────────────────────────────┘
         │
         ├──► Graph (Mutable)
         │    ├──► nodes: Dict[str, Node]
         │    └──► edges: Dict[str, Edge]
         │
         ├──► History (Append-Only)
         │    └──► turns: List[Turn]
         │
         ├──► CoverageState (Mutable)
         │    ├──► element_node_mappings: Dict
         │    ├──► element_reactions: Dict
         │    └──► gaps: List[CoverageGap]
         │
         ├──► Momentum (Mutable)
         │    ├──► current_level: str
         │    ├──► history: List[MomentumRecord]
         │    └──► consecutive_low_count: int
         │
         ├──► GraphState (Computed, Immutable)
         │    ├──► isolated_nodes: List[Node]
         │    ├──► ambiguous_nodes: List[Node]
         │    └──► active_branch: List[str]
         │
         └──► Tracking Objects (Mutable)
              ├──► NodeFocusTracker
              ├──► EdgeFocusTracker
              └──► Element exhaustion flags
```

### State Update Flow

```
User Response
     │
     ▼
[Extraction]
     │
     ├──► Graph.add_node(node)  ──► Modifies nodes dict
     ├──► Graph.add_edge(edge)  ──► Modifies edges dict
     │
     ▼
[Coverage Update]
     │
     ├──► CoverageState.update(graph, mappings)  ──► Modifies mappings
     ├──► CoverageState.record_reaction(...)      ──► Modifies reactions
     └──► CoverageState._recompute_gaps()         ──► Modifies gaps
     │
     ▼
[State Computation]
     │
     ├──► GraphState.compute(graph, schema, history)  ──► Returns NEW GraphState
     └──► Momentum.update(new_level)                  ──► Modifies momentum
     │
     ▼
[Strategy Selection]
     │
     └──► Uses GraphState, CoverageState, Momentum (READ ONLY)
     │
     ▼
[Question Generation]
     │
     └──► Uses Strategy, Focus, History, Momentum (READ ONLY)
     │
     ▼
[Turn Recording]
     │
     └──► History.add_turn(turn)  ──► Appends to turns list
```

### Immutability Guarantees

| Component | Mutability | Why |
|-----------|-----------|-----|
| **Graph** | Mutable | Needs to grow during interview |
| **History** | Append-only | Immutable after recording, for audit trail |
| **CoverageState** | Mutable | Tracks dynamic element coverage |
| **Momentum** | Mutable | Records engagement history |
| **GraphState** | Immutable | Computed snapshot, never modified |
| **Strategy** | Immutable | Configuration object |
| **Schema** | Immutable | Methodology definition |

---

## Async/Concurrency

### Current Architecture: Synchronous

**Key Characteristics**:
- Single-threaded execution
- No concurrent operations
- Blocking LLM calls

**Rationale**:
- Interview is inherently sequential (one turn at a time)
- State updates must be atomic
- Complexity of concurrent graph modifications outweighs benefits

### If Async/Concurrent Operations Needed

**Parallelization Opportunities**:

1. **LLM Calls** (I/O bound):
   ```python
   # Current (sequential)
   extractability = extractor.assess_extractability(response)
   extraction = extractor.extract(response, ...)
   momentum = extractor.assess_momentum(response, ...)

   # Potential (parallel)
   async def process_response_async(response):
       extractability_task = asyncio.create_task(extractor.assess_extractability_async(response))
       extraction_task = asyncio.create_task(extractor.extract_async(response, ...))
       momentum_task = asyncio.create_task(extractor.assess_momentum_async(response, ...))

       extractability, extraction, momentum = await asyncio.gather(
           extractability_task,
           extraction_task,
           momentum_task
       )
   ```

2. **State Computation** (CPU bound):
   ```python
   # Potential (parallel)
   async def compute_all_state_async(graph, schema, history, coverage_state):
       graph_state_task = asyncio.create_task(GraphState.compute_async(graph, schema, history))
       gaps_task = asyncio.create_task(coverage_state._recompute_gaps_async())

       graph_state, _ = await asyncio.gather(graph_state_task, gaps_task)
       return graph_state
   ```

### Concurrency Considerations

**What Needs Locking**:
1. **Graph modifications**: `add_node()`, `add_edge()`
2. **History appends**: `add_turn()`
3. **Coverage updates**: `update()`, `record_reaction()`
4. **Momentum updates**: `update()`
5. **Token/cost tracking**: `LLMManager` counters

**What Doesn't Need Locking** (read-only):
- GraphState computation (reads graph)
- Strategy selection (reads state)
- Question generation (reads history)

**Lock Strategy** (if implemented):
```python
class InterviewController:
    def __init__(self):
        self.graph_lock = asyncio.Lock()
        self.history_lock = asyncio.Lock()
        self.coverage_lock = asyncio.Lock()
        self.momentum_lock = asyncio.Lock()

    async def process_response(self, response):
        # Parallel LLM calls (no locks needed)
        extractability, extraction, momentum = await asyncio.gather(...)

        # Serial state updates (locks needed)
        async with self.graph_lock:
            for node in extraction.nodes:
                self.graph.add_node(node)
            for edge in extraction.edges:
                self.graph.add_edge(edge)

        async with self.coverage_lock:
            self.coverage_state.update(self.graph, extraction.node_element_mappings)

        async with self.momentum_lock:
            self.momentum.update(momentum_level)

        # Parallel state computation (read-only, no locks)
        graph_state = await GraphState.compute_async(self.graph, self.schema, self.history)

        # Strategy selection (read-only, no locks)
        strategy, focus = await self.strategy_selector.select_async(...)

        # Question generation (read-only, no locks)
        question = await self.question_generator.generate_async(...)

        # Record turn (lock needed)
        async with self.history_lock:
            self.history.add_turn(turn)

        return question
```

### Performance Implications

**Current (Synchronous)**:
- Turn processing time: 2-5 seconds (dominated by LLM calls)
- No locking overhead
- Simple debugging

**If Async (Parallel LLM Calls)**:
- Turn processing time: 1-2 seconds (LLM calls in parallel)
- Locking overhead: minimal (state updates are fast)
- Complex debugging (race conditions)

**Recommendation**: Stay synchronous unless interview throughput becomes bottleneck (unlikely for 20-turn interviews).

---

## Summary

This component map provides:

1. **Complete dependency graph**: Every component's dependencies and dependents
2. **Interaction patterns**: How components communicate (request-response, compute-query, etc.)
3. **State management**: What's mutable, what's immutable, why
4. **Async/concurrency**: Current architecture and considerations for parallelization

### Key Takeaways

**For AI Agents**:
- **Start at InterviewController**: Central orchestrator, easiest entry point
- **GraphState is computed**: Don't expect it to be cached or modified
- **Arbitration is complex**: 9 multiplicative scorers create emergent behavior
- **Gaps are expensive**: O(E*N²) recomputation for connection requirements
- **History is immutable**: Never modify past turns

**For Developers**:
- **Separation of concerns**: Graph (data) vs State (analysis) vs Strategy (decision)
- **Stateless components**: Extractor, QuestionGenerator, StrategySelector are pure functions
- **Mutable state centralized**: InterviewController owns all mutable state
- **LLM abstraction**: LLMManager handles all provider differences

**For Debugging**:
- **Check arbitration logs**: See which scorers fired and why
- **Inspect GraphState**: Understand topology before diving into strategies
- **Review History**: Replay interview turn-by-turn
- **Validate schema**: Ensure extracted nodes/edges conform to methodology
