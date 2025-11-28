# AI-Based Graph Interviewer: Technical Design Document

## Version Control
- **Version:** 1.0
- **Date:** November 28, 2025
- **Document Type:** System Architecture & Implementation Design
- **Scope:** Phase 1 - Single Interview System

---

## Section 1: Problem Statement & Context

### Problem
Current AI interviewing solutions for marketing research suffer from:
- Heavy reliance on pre-structured interview guides
- Limited adaptive probing capabilities that require manual examples
- No formal definition of "response richness"
- Inability to systematically explore conceptual space
- Poor detection of under-explored areas during interviews

### Core Insight
Marketing research interviews aim to map a consumer's mental model, which can be formalized as a **knowledge graph**. Response richness can be quantified by graph growth metrics (new nodes, edges, and structural properties). This transforms interviewing from script-following into goal-driven graph construction.

### Target Domain
**FMCG (Fast-Moving Consumer Goods) concept testing** - specifically consumer feedback on product innovations. This domain provides:
- Finite, well-defined conceptual vocabulary (~20-30 core concepts)
- Predictable interview structure blocks (need states, product reactions, usage occasions, value perception)
- Clear business deliverables (actionable insights for positioning, messaging, product development)

### Constraints

**Non-Negotiable:**
- Real-time interaction latency: <3 seconds per interview turn
- Graph evolution must be incremental (not batch processing)
- Schema must be flexible and experimentable during POC
- Human-interpretable graph outputs for validation
- Complete conversation transcript preservation

**Flexible:**
- Exact node/edge type taxonomy (will evolve through experimentation)
- Question generation method (template vs. LLM-generated)
- Richness scoring formula parameters
- Interview length and stopping criteria

**Environment:**
- Local development initially (Python, in-memory graph)
- Cloud deployment for Phase 2 (scalability)
- Integration with standard research platforms (future requirement)

---

## Section 2: Requirements

### Functional Requirements

**FR1: Schema-Driven Graph Construction**
- System must load graph ontology from external configuration (YAML/JSON)
- Support versioned schemas allowing A/B testing of different mental model representations
- Enable hot-swapping between different schema "lenses" (e.g., Means-End Chain vs. Usage Script)

**FR2: Incremental Response Processing**
- Accept text responses and extract graph deltas (new nodes/edges) in real-time
- Validate extracted entities against current schema
- Update graph transactionally with source attribution (quote, timestamp)
- Maintain conversation context window (last 2-3 turns)

**FR3: Dynamic Interview Management**
- Assess graph state after each turn to identify exploration opportunities
- Prioritize next question based on: coverage gaps, depth potential, connection opportunities
- Maintain conversational focus (prevent random topic-hopping)
- Implement time budgets per research block

**FR4: Adaptive Question Generation**
- Generate contextually appropriate follow-up questions
- Support both template-based and LLM-generated rendering
- Maintain natural conversation flow (avoid repetition)
- Implement probing strategies: dig_deeper, connect_concepts, switch_topic, introduce_seed

**FR5: Interview Termination**
- Detect graph saturation (diminishing returns)
- Identify "dead ends" (unproductive inquiry paths)
- Enforce maximum turn limits
- Provide interview completeness metrics

**FR6: Data Persistence & Observability**
- Log every turn: response, graph delta, interview state, generated question
- Store complete conversation transcript with graph version metadata
- Export final graph in standard format (GraphML, JSON)
- Generate post-interview analytics dashboard

### Non-Functional Requirements

**NFR1: Performance**
- Response-to-question cycle: ≤3 seconds (90th percentile)
- Graph delta extraction: ≤2 seconds
- Question generation: ≤1 second
- Support up to 30-turn interviews without degradation

**NFR2: Reliability**
- Graceful handling of LLM extraction failures (retry → template fallback)
- Graph validation errors must not crash interview
- Participant mid-conversation disconnect recovery
- Schema validation errors surface early (at load time)

**NFR3: Maintainability**
- Schema changes require only config file edits (no code changes)
- All decision logic (opportunity prioritization) must be deterministic and testable
- Clear separation: graph operations, LLM calls, conversation management
- Comprehensive logging for debugging unexpected question choices

**NFR4: Extensibility**
- Support for custom richness scoring functions
- Pluggable question generation strategies
- Multiple LLM backend support (OpenAI, Anthropic, local models)
- Graph analysis plugins (centrality measures, community detection)

### Dependencies

**Core Libraries:**
- `NetworkX` (≥3.0) - In-memory graph operations
- `pydantic` (≥2.0) - Schema validation and data models
- `PyYAML` (≥6.0) - Schema manifest parsing
- `anthropic` or `openai` - LLM API client
- `redis` (optional, for caching embeddings)

**External Services:**
- LLM API (Claude 3.5 Sonnet or GPT-4) for response extraction and question rendering
- Embedding service (optional) for semantic node similarity

**Data Sources:**
- Schema manifest files (YAML)
- Seed concept definitions (from research objectives)
- Question template library

---

## Section 3: System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Interview Controller                          │
│  (Orchestrates turn loop, manages conversation state)               │
└────────────┬────────────────────────────────────────┬───────────────┘
             │                                        │
             ▼                                        ▼
┌────────────────────────┐                 ┌─────────────────────────┐
│   Schema Manager       │◄────────────────┤  Interview Graph        │
│ (Loads/validates       │                 │  (NetworkX store)       │
│  ontology)             │                 │  - Nodes/edges          │
└────────────────────────┘                 │  - Visit counts         │
             │                             │  - Source quotes        │
             │                             └──────────┬──────────────┘
             │                                        │
             ▼                                        │
┌────────────────────────┐                           │
│  Response Processor    │◄──────────────────────────┘
│  - LLM extraction      │
│  - Entity validation   │
│  - Graph delta builder │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐         ┌──────────────────────────┐
│  Interview Manager     │────────►│  Question Generator      │
│  - Assess graph state  │         │  - Template selector     │
│  - Opportunity ranking │         │  - LLM renderer          │
│  - Focus stack         │         │  - Repetition checker    │
│  - Dead-end detection  │         └──────────────────────────┘
└────────────────────────┘
             │
             ▼
┌────────────────────────┐
│  Persistence Layer     │
│  - Turn logs           │
│  - Graph snapshots     │
│  - Transcript export   │
└────────────────────────┘
```

### Core Data Flow

**Main Interview Turn Loop:**

```
1. Participant Response (text)
   │
   ▼
2. Response Processor
   ├─► Extract graph delta (LLM call with schema context)
   ├─► Validate against schema
   └─► Apply to Interview Graph (transactional)
   │
   ▼
3. Interview Manager
   ├─► Calculate graph metrics (coverage, depth)
   ├─► Identify opportunities (unexplored seeds, shallow nodes)
   ├─► Rank by priority (schema weights + centrality + recency)
   └─► Select top opportunity
   │
   ▼
4. Question Generator
   ├─► Map opportunity → question intent
   ├─► Select template OR call LLM renderer
   └─► Check for repetition
   │
   ▼
5. Question Delivery (text)
   │
   ▼
6. Persist turn data (async)
   │
   └─► Loop back to step 1
```

**Termination Flow:**
```
Interview Manager checks after each turn:
├─► opportunity_priority < threshold? → END
├─► max_turns reached? → END
├─► participant_disconnect? → SUSPEND
└─► else → CONTINUE
```

---

## Section 4: Design Decisions & Rationale

### Decision 1: Meta-Schema Architecture

**Decision:** Implement a "schema manifest" system where graph ontology is defined in external YAML files loaded at runtime, not hardcoded.

**Rationale:**
- FMCG concept structure varies by product category (beverages vs. cleaning products)
- Research hypotheses evolve - need to test "Means-End Chain" vs. "Usage Script" mental model lenses
- Enabling zero-code experimentation for researchers (edit YAML, restart service)
- POC requires rapid iteration on node/edge types based on early findings

**Alternatives Considered:**
1. Hardcoded ontology in Python classes → Rejected: requires code changes for every experiment
2. Full database-backed schema versioning → Rejected: overkill for POC, adds latency
3. LLM defines schema on-the-fly → Rejected: too unstable, no reproducibility

**Tradeoffs:**
- **Gained:** Flexibility, versioned experiments, researcher self-service
- **Sacrificed:** Some runtime validation complexity, YAML parsing overhead (~100ms at startup)

**Constraints:**
- Schema must be locked per interview session (no mid-interview changes)
- Max 30 node/edge types to keep LLM extraction manageable
- Schema migration between interview versions happens at aggregation time (Phase 2)

### Decision 2: Incremental Graph Updates (Not Batch)

**Decision:** Process each response turn immediately, updating graph in real-time, rather than batch-processing the full transcript.

**Rationale:**
- System must decide next question dynamically based on current graph state
- Enables adaptive probing (dig deeper when shallow, switch when saturated)
- Participant sees coherent follow-up questioning
- Matches natural conversation flow

**Alternatives Considered:**
1. Batch transcript analysis post-interview → Rejected: defeats purpose of adaptive questioning
2. Hybrid (quick extraction + delayed deep analysis) → Considered for Phase 2

**Tradeoffs:**
- **Gained:** Dynamic interview adaptation, real-time richness feedback
- **Sacrificed:** Can't use full conversation history for every extraction (token limits)

**Constraints:**
- Must meet <3 second latency budget
- Context window limited to last 2-3 turns (not full transcript)
- No correction of past graph errors without manual review

### Decision 3: Opportunity-Driven Question Selection

**Decision:** Interview Manager computes ranked list of "exploration opportunities" from graph state, selects highest priority, then generates question for that intent.

**Rationale:**
- Separates "what to ask about" (deterministic graph analysis) from "how to phrase it" (LLM rendering)
- Makes decision logic testable and debuggable (no black-box prompting)
- Enables explicit research strategy (coverage → depth phases)
- Question quality becomes independent concern (template vs. LLM)

**Alternatives Considered:**
1. End-to-end LLM prompting ("ask next question given graph") → Rejected: opaque, inconsistent, slow
2. Fixed question tree → Rejected: loses adaptability

**Tradeoffs:**
- **Gained:** Explainability, determinism, faster iteration
- **Sacrificed:** Need to explicitly encode interview heuristics (but this is desirable for validation)

**Constraints:**
- Opportunity ranking formula must be tunable via schema config
- Focus stack required to prevent erratic topic-switching

### Decision 4: Hybrid Template + LLM Question Rendering

**Decision:** Maintain question templates for common patterns; fall back to LLM rendering for complex or novel situations.

**Rationale:**
- Templates are instant (<10ms), consistent, and cost-free
- LLM rendering provides natural variation and handles edge cases
- Repetition avoidance requires tracking last N questions (templates or LLM)

**Alternatives Considered:**
1. Templates only → Rejected: feels robotic, limited expressiveness
2. LLM only → Rejected: unnecessary latency and cost for routine follow-ups

**Tradeoffs:**
- **Gained:** Speed + naturalness balance, cost efficiency
- **Sacrificed:** Need dual rendering pipeline maintenance

**Constraints:**
- Template library must cover 60%+ of common opportunities (empirical target)
- LLM rendering must stay under 1-second budget (use faster models if needed)

### Decision 5: Graph Validation as Rejection, Not Correction

**Decision:** If extracted graph delta fails schema validation, reject it entirely and trigger a fallback question rather than attempting to "fix" it.

**Rationale:**
- Auto-correction risks polluting graph with hallucinated entities
- Failed extraction indicates ambiguous response or LLM error - better to re-ask
- Clean rejection surfaces issues for schema refinement

**Alternatives Considered:**
1. Auto-correction with confidence scoring → Rejected: false confidence dangerous
2. Human-in-loop validation → Rejected: breaks real-time requirement (for Phase 2 review)

**Tradeoffs:**
- **Gained:** Graph integrity, clear failure modes
- **Sacrificed:** Some valid extractions lost to overly strict validation

**Constraints:**
- Max 2 consecutive validation failures → abort interview or use safe fallback question
- Validation rules must be explicit in schema (regex, allowed edge types)

---

## Section 5: Component Specifications

### Component 1: Schema Manager

**Purpose:** Load, validate, and provide access to the graph ontology definition for all other components.

**Interface:**
- **Input:** Path to YAML schema manifest file
- **Output:** Validated `SchemaManifest` object
- **Methods:**
  - `load_schema(path: str) -> SchemaManifest`
  - `validate_node(node_data: dict) -> bool`
  - `validate_edge(edge_data: dict) -> bool`
  - `get_node_type_prompt(node_type: str) -> str`
  - `get_richness_weight(node_type: str) -> float`

**Behavior:**
1. Parse YAML file into structured object
2. Validate schema completeness (all node types have prompts, valid edge sources/targets)
3. Generate Pydantic validators from schema definitions
4. Cache schema for fast repeated access
5. Store schema version hash with every graph operation

**Error Handling:**
- **Missing required fields in schema** → Fail fast at startup with detailed error
- **Invalid edge type references** → List all orphaned references, abort load
- **Duplicate node type names** → Reject with conflict details
- **Malformed YAML** → Surface parsing error with line number

**Dependencies:** PyYAML, Pydantic

---

### Component 2: Interview Graph

**Purpose:** Maintain the evolving knowledge graph representing the participant's mental model.

**Interface:**
- **Input:** `GraphDelta` objects (new nodes, edges, metadata)
- **Output:** Graph state queries (nodes, edges, metrics)
- **Methods:**
  - `add_nodes(nodes: List[Node], source_quote: str) -> None`
  - `add_edges(edges: List[Edge], source_quote: str) -> None`
  - `get_node(node_id: str) -> Node`
  - `get_neighbors(node_id: str) -> List[Node]`
  - `get_visit_count(node_id: str) -> int`
  - `compute_centrality(node_id: str) -> float`
  - `find_unlinked_pairs() -> List[Tuple[Node, Node]]`
  - `export_graph(format: str) -> str` (GraphML, JSON)

**Behavior:**
1. Store nodes with attributes: type, label, creation_turn, visit_count, source_quotes
2. Store edges with attributes: type, weight, creation_turn, source_quote
3. Track exploration metrics per node (depth, centrality, richness_per_visit)
4. Maintain seed node list separately for coverage checks
5. Snapshot graph state after each turn (for rollback if needed)

**Error Handling:**
- **Duplicate node addition** → Increment visit_count, append source_quote, don't duplicate
- **Edge with non-existent nodes** → Reject with clear error listing missing node IDs
- **Graph becomes disconnected** → Log warning but continue (might be intentional)
- **Memory overflow** (>10k nodes) → Trigger early termination, flag anomaly

**Dependencies:** NetworkX

---

### Component 3: Response Processor

**Purpose:** Extract structured graph deltas from unstructured text responses using LLM-assisted parsing.

**Interface:**
- **Input:**
  - `response_text: str` - Current participant utterance
  - `conversation_context: List[str]` - Last 2-3 turns for coreference resolution
  - `schema: SchemaManifest` - Current ontology
  - `existing_graph: InterviewGraph` - For entity grounding
- **Output:** `GraphDelta` object containing validated nodes and edges, or `ValidationError`
- **Methods:**
  - `extract_delta(response: str, context: List[str]) -> GraphDelta`
  - `validate_delta(delta: GraphDelta) -> bool`

**Behavior:**
1. Construct LLM prompt with:
   - Schema node/edge type definitions from manifest
   - Conversation context (last 2-3 turns)
   - Recent graph state (for entity grounding)
   - Extraction examples from schema prompts
2. Call LLM with structured output format (function calling or constrained JSON)
3. Parse LLM response into candidate nodes and edges
4. Validate each node:
   - Matches defined node type
   - Label passes regex if specified
   - Not duplicate (or intentional revisit)
5. Validate each edge:
   - Both source and target nodes exist or are being created
   - Edge type allows this source→target pairing per schema
6. Return `GraphDelta` if valid, `ValidationError` with details if not

**Error Handling:**
- **LLM timeout (>10s)** → Retry once with shorter context, then fail
- **Malformed LLM output** → Return empty delta, log incident, use template fallback question
- **Validation failure** → Return error with specific rule violations
- **Ambiguous coreference ("it", "that")** → Extract mention, flag as uncertain, request clarification in next turn

**Dependencies:** LLM API client, Schema Manager, Interview Graph

---

### Component 4: Interview Manager

**Purpose:** Analyze current graph state to determine what to explore next and manage conversational focus.

**Interface:**
- **Input:**
  - `graph: InterviewGraph` - Current state
  - `schema: SchemaManifest` - For priority rules
  - `turn_number: int` - For phase-based strategy
  - `focus_stack: List[Node]` - Active inquiry threads
- **Output:** `InterviewState` containing ranked opportunities and recommended action
- **Methods:**
  - `assess_graph() -> InterviewState`
  - `rank_opportunities(opportunities: List[Opportunity]) -> List[Opportunity]`
  - `detect_dead_ends() -> Set[Node]`
  - `should_terminate() -> bool`

**Behavior:**
1. **Coverage Check:**
   - Identify unexplored seed nodes (visit_count == 0)
   - Create `introduce_topic` opportunity for each
   - Assign high priority (10) if in early turns, lower (5) if late
2. **Depth Check:**
   - Find recently mentioned nodes with shallow depth (<2)
   - Create `dig_deeper` opportunity
   - Priority = base (8) + centrality_score
3. **Connection Check:**
   - Find pairs of nodes from recent turns with no connecting edge
   - Create `connect_concepts` opportunity
   - Priority = 7 (medium)
4. **Dead-End Detection:**
   - Nodes probed 3+ times with richness_per_visit < threshold
   - Add to dead_ends set, deprioritize in future
5. **Focus Management:**
   - If current focus_stack top has opportunities, prioritize those
   - If focus_stack path exhausted (dead-end), pop and switch
6. **Rank & Select:**
   - Sort opportunities by priority score
   - Return top opportunity as recommended action

**Error Handling:**
- **No opportunities found** → Return terminate signal
- **All nodes dead-ends** → Return terminate signal
- **Focus stack overflow (>5 deep)** → Force pop to prevent spiral
- **Priority tie** → Use recency as tiebreaker

**Dependencies:** Interview Graph, Schema Manager

---

### Component 5: Question Generator

**Purpose:** Translate interview manager's intent into natural language questions.

**Interface:**
- **Input:**
  - `opportunity: Opportunity` - What to ask about and why
  - `schema: SchemaManifest` - For node-specific prompts
  - `recent_questions: List[str]` - For repetition checking
- **Output:** `question_text: str`
- **Methods:**
  - `generate_question(opportunity: Opportunity) -> str`
  - `render_template(template: str, **kwargs) -> str`
  - `render_with_llm(intent: str, context: dict) -> str`
  - `check_repetition(question: str, recent: List[str]) -> float` (similarity score)

**Behavior:**
1. **Template Selection:**
   - Map opportunity action to template library
   - For `dig_deeper` → choose from 3 variants based on node type
   - For `connect` → use connection template with node names
   - For `introduce_topic` → use schema-defined introduction prompt
2. **Variable Substitution:**
   - Fill template placeholders with node labels, context quotes
   - Ensure grammatical correctness (articles, plurals)
3. **Repetition Check:**
   - Compute similarity with last 3 questions (simple word overlap or embeddings)
   - If similarity >0.7, try alternate template or add variety phrase
4. **LLM Rendering (fallback):**
   - If no template fits or repetition unavoidable, call LLM
   - Prompt: "Given {opportunity}, generate conversational follow-up question"
   - Max tokens: 50 (keep latency <1s)
5. **Output question text**

**Error Handling:**
- **No template available** → Fall back to LLM render
- **LLM render fails** → Use generic template ("Tell me more about that")
- **Repetition unavoidable** → Log incident, proceed anyway (better than silence)
- **Generated question is off-topic** → Log for review, but deliver (trust LLM mostly)

**Dependencies:** Schema Manager, LLM API client (optional)

---

### Component 6: Interview Controller

**Purpose:** Orchestrate the entire interview turn loop, manage state transitions, and coordinate all other components.

**Interface:**
- **Input:**
  - Initial configuration (schema path, session ID, participant ID)
  - Stream of participant responses
- **Output:**
  - Stream of interviewer questions
  - Final interview data package (graph, transcript, analytics)
- **Methods:**
  - `start_interview(schema_path: str, session_id: str) -> None`
  - `process_turn(response: str) -> str` (returns next question)
  - `end_interview() -> InterviewResult`

**Behavior:**
1. **Initialization:**
   - Load schema via Schema Manager
   - Create empty Interview Graph with seed nodes from schema
   - Initialize Interview Manager and Question Generator
   - Start turn counter and focus stack
2. **Turn Loop:**
   - Accept participant response
   - Pass to Response Processor → get GraphDelta
   - On validation success: apply delta to graph
   - On validation failure: log error, generate fallback question, continue
   - Call Interview Manager → get next opportunity
   - If terminate signal: end interview
   - Else: call Question Generator → get question text
   - Log complete turn data (response, delta, state, question)
   - Return question to participant
3. **Termination:**
   - Export final graph (GraphML + JSON)
   - Compile transcript with annotations
   - Generate summary analytics (coverage %, avg depth, richness score)
   - Store to persistence layer

**Error Handling:**
- **Schema load failure** → Abort interview startup, surface error to user
- **Component crash (any)** → Log stack trace, attempt graceful recovery or end interview early
- **Participant disconnect** → Suspend state, enable resume from last turn
- **API rate limit** → Pause interview, wait, resume (with timeout)

**Dependencies:** All other components

---

### Component 7: Persistence Layer

**Purpose:** Store all interview data for reproducibility, debugging, and post-hoc analysis.

**Interface:**
- **Input:** Turn logs, graph snapshots, configuration
- **Output:** Queryable interview database
- **Methods:**
  - `log_turn(turn_data: TurnLog) -> None`
  - `save_graph_snapshot(graph: InterviewGraph, turn: int) -> None`
  - `export_interview(session_id: str, format: str) -> str`

**Behavior:**
1. Write turn log JSON: response, delta, state, question, timestamp
2. Snapshot graph every N turns (e.g., every 5)
3. Store schema version with every write
4. On interview end, write complete transcript and final graph

**Error Handling:**
- **Disk full** → Log to stderr, continue interview (data loss acceptable in POC)
- **Write permission denied** → Fail gracefully, warn user
- **Corrupt log file** → Isolate bad turn, don't crash whole interview

**Dependencies:** File system, JSON

---

## Section 6: Algorithms & Logic

### Algorithm 1: Richness Scoring

**Purpose:** Quantify how much new information a response added to the graph.

**Approach:** Weighted sum of graph growth metrics derived from the delta.

**Steps:**
1. For each new node added, add `schema.richness_weight(node.type)` to score
2. For each new edge added, add `schema.richness_boost(edge.type)` to score
3. If new node connects two previously disconnected clusters, add community bridging bonus (+2.0)
4. If new node has high centrality (connects to 3+ existing nodes), add centrality bonus (+ node_degree * 0.5)
5. Normalize by response length (divide by log(word_count)) to penalize rambling

**Edge Cases:**
- **Empty response** → Richness = 0
- **Response adds only duplicate nodes** → Richness = 0, but update visit_counts
- **Response adds cycle edge** → Still counts, but log for potential consistency check

**Formula:**
```
richness = (Σ node_weight + Σ edge_boost + bridging_bonus + centrality_bonus) / log(word_count + 1)
```

**Complexity:** O(n + m) where n = nodes in delta, m = edges in delta

---

### Algorithm 2: Opportunity Ranking

**Purpose:** Prioritize which aspect of the graph to explore next.

**Approach:** Multi-factor scoring combining research objectives, graph structure, and conversational context.

**Steps:**
1. Enumerate all possible opportunities from current graph state:
   - Unexplored seeds (coverage)
   - Shallow nodes (depth)
   - Unconnected pairs (connection)
2. For each opportunity, compute base priority from schema
3. Adjust priority based on:
   - **Centrality:** High-centrality nodes get +2 priority
   - **Recency:** Nodes mentioned in last 2 turns get +1 priority
   - **Focus alignment:** On-stack nodes get +3 priority
   - **Phase bias:** Early turns (<5) get coverage +5, later turns get depth +3
4. Apply dead-end penalty: nodes in dead_ends set get priority *0.1
5. Sort opportunities by adjusted priority (descending)
6. Return top opportunity

**Edge Cases:**
- **All opportunities are dead-ends** → Return None (triggers termination)
- **Priority tie** → Use alphabetical node label as stable tiebreaker
- **No opportunities at all** → Return None (triggers termination)

**Formula:**
```
priority = (base + centrality_bonus + recency_bonus + focus_bonus + phase_bonus) * dead_end_penalty
```

**Complexity:** O(k log k) where k = number of opportunities (~10-50 typical)

---

### Algorithm 3: Dead-End Detection

**Purpose:** Identify nodes that are unlikely to yield further richness, preventing infinite probing.

**Approach:** Track richness-per-visit ratio over multiple probe attempts.

**Steps:**
1. For each node, maintain running average: `avg_richness = total_richness / visit_count`
2. After each turn, if a node was probed (appeared in focus), update its richness history
3. If node has been probed 3+ times AND `avg_richness < schema.shallow_threshold`, mark as dead-end
4. Dead-end nodes get low priority (×0.1) in future opportunity ranking
5. If participant spontaneously mentions a dead-end node (not in response to direct probe), remove from dead-end set (re-activated)

**Edge Cases:**
- **First visit yields zero richness** → Don't mark as dead-end yet (might be poor phrasing)
- **Node has high centrality but low richness** → Still mark as dead-end (quantity ≠ quality)
- **Seed nodes** → Never mark as dead-end (must ensure coverage)

**Stopping Condition:** Node remains dead-end unless spontaneously re-activated by participant.

**Complexity:** O(1) per node update

---

### Algorithm 4: Focus Stack Management

**Purpose:** Maintain conversational coherence by tracking active inquiry threads.

**Approach:** Stack-based state machine with explicit push/pop rules.

**Steps:**
1. **Push:** When `dig_deeper` opportunity is selected, push target node onto focus_stack
2. **Continue:** If top of stack still has unexplored depth (richness not declining), stay on it
3. **Pop:** When top node becomes dead-end or max depth reached (4 consecutive probes), pop it
4. **Switch:** When stack is empty or forced switch triggered (time budget), select new opportunity via ranking
5. **Limit:** If stack depth exceeds 5, force pop (prevent conversational spiral)

**Edge Cases:**
- **Stack empty at turn 1** → Push first introduced seed node
- **Participant switches topic unprompted** → Don't force back to stack; pop and follow participant's lead (store old focus for potential return)
- **Stack full (5 deep)** → Pop 2 levels, resume from 3rd

**Complexity:** O(1) per operation

---

### Algorithm 5: Coreference Resolution (Lightweight)

**Purpose:** Link pronouns and vague references ("it", "that") to concrete graph nodes.

**Approach:** Recency-based heuristic with LLM fallback.

**Steps:**
1. Detect coreference indicators in response (pronouns: "it", "that", "this", "they")
2. Retrieve last 2 mentioned nodes from conversation context
3. If single candidate → link reference to that node
4. If multiple candidates → use LLM with context: "Does '{pronoun}' refer to {node_A} or {node_B}?"
5. If unresolved → extract as ambiguous mention, add to graph with uncertainty flag, request clarification in next turn

**Edge Cases:**
- **No prior context** (first turn) → Can't resolve, extract as new node
- **Reference to concept outside graph** → Extract as new node
- **LLM resolution fails** → Default to most recent node (best guess)

**Complexity:** O(1) typical, O(n) worst case where n = context size

---

## Section 7: Configuration & Parameters

### Schema Manifest Parameters

**`schema_version`:** string  
Purpose: Track schema evolution for reproducibility  
Default: "1.0.0"  
Format: Semantic versioning  
Impact: Enables loading appropriate graph migration logic

**`domain`:** string  
Purpose: High-level categorization of ontology  
Default: "fmcg_concept_test"  
Impact: Documentation only, no runtime effect

---

### Node Type Parameters (per type)

**`name`:** string  
Purpose: Unique identifier for node type  
Example: "pain_point", "benefit", "usage_context"

**`description`:** string  
Purpose: Human-readable explanation for LLM extraction  
Example: "A specific frustration or problem the consumer faces"

**`llm_extraction_prompt`:** string  
Purpose: Guidance for LLM to identify this node type in responses  
Example: "Extract concrete frustrations. Look for 'hard to', 'annoying', 'frustrating when'"

**`richness_weight`:** float  
Purpose: Score contribution when this node type is added  
Default: 1.0  
Range: [0.1, 5.0]  
Impact: Higher → encourages system to seek more of this type  
  - pain_point: 1.0 (baseline)  
  - benefit: 1.5 (valuable insights)  
  - value: 2.0 (deep motivation)

**`validation_regex`:** string (optional)  
Purpose: Format validation for node labels  
Example: "^[a-z_]{3,40}$" (lowercase, underscores, 3-40 chars)  
Impact: Rejects malformed extractions early

---

### Edge Type Parameters (per type)

**`name`:** string  
Purpose: Unique identifier for edge type  
Example: "causes", "mitigates", "contradicts"

**`description`:** string  
Purpose: Human-readable relationship meaning  
Example: "A causes or directly brings about B"

**`valid_sources`:** List[string]  
Purpose: Allowed source node types for this edge  
Example: ["pain_point", "action"]  
Impact: Validation rejects edges from wrong source types

**`valid_targets`:** List[string]  
Purpose: Allowed target node types for this edge  
Example: ["emotion", "outcome"]  
Impact: Validation rejects edges to wrong target types

**`richness_boost`:** float  
Purpose: Score contribution when this edge type is added  
Default: 1.0  
Range: [0.5, 5.0]  
Impact: Higher → prioritizes finding these relationships  
  - happens_before: 0.5 (temporal links common)  
  - causes: 2.0 (causal links rare and valuable)  
  - contradicts: 3.0 (reveals tensions)

---

### Richness Scoring Parameters

**`formula`:** string  
Purpose: Define scoring calculation  
Default: "sum(new_nodes * weight) + sum(new_edges * boost)"  
Options: Can include bonuses like "+ centrality_bonus * 0.5"

**`thresholds`:** dict  
Purpose: Classify response richness levels  
Default:
  - shallow: 0.5  
  - moderate: 2.0  
  - rich: 5.0  
Impact: Affects dead-end detection and reporting

---

### Interview Controller Parameters

**`max_turns`:** int  
Purpose: Hard limit on interview length  
Default: 30  
Range: [10, 100]  
Impact: Longer allows deeper exploration but risks participant fatigue

**`turn_timeout_seconds`:** float  
Purpose: Max wait time for participant response before timeout  
Default: 300.0 (5 minutes)  
Range: [60, 1800]  
Impact: Shorter = assume disconnect faster

**`latency_budget_seconds`:** float  
Purpose: Target response time per turn  
Default: 3.0  
Range: [1.0, 10.0]  
Impact: Affects LLM timeout settings and retry logic

---

### Interview Manager Parameters

**`coverage_priority_boost`:** int  
Purpose: Extra priority for unexplored seed nodes in early turns  
Default: 10  
Range: [1, 20]  
Impact: Higher → ensures all topics touched before going deep

**`depth_priority_boost`:** int  
Purpose: Extra priority for deep diving in later turns  
Default: 8  
Range: [1, 20]  
Impact: Higher → encourages thorough exploration over breadth

**`phase_transition_turn`:** int  
Purpose: Turn number to switch from coverage to depth priority  
Default: 5  
Range: [3, 10]  
Impact: Earlier → faster specialization; Later → more balanced coverage

**`dead_end_threshold`:** float  
Purpose: Avg richness below which a node becomes dead-end  
Default: 0.5  
Range: [0.1, 2.0]  
Impact: Lower → marks more nodes as exhausted; Higher → keeps probing longer

**`dead_end_probe_count`:** int  
Purpose: Minimum probes before considering dead-end  
Default: 3  
Range: [2, 5]  
Impact: Higher → gives nodes more chances before abandoning

---

### Question Generator Parameters

**`template_usage_probability`:** float  
Purpose: Chance to use template vs. LLM for question rendering  
Default: 0.7  
Range: [0.0, 1.0]  
Impact: Higher → faster and cheaper; Lower → more varied questions

**`repetition_threshold`:** float  
Purpose: Similarity score above which a question is considered repetitive  
Default: 0.7  
Range: [0.5, 0.9]  
Impact: Higher → tolerates more similarity; Lower → enforces more variety

**`llm_max_tokens`:** int  
Purpose: Token limit for LLM-generated questions  
Default: 50  
Range: [20, 100]  
Impact: Higher → more elaborate questions but slower

---

## Section 8: Error Handling Strategy

### Expected Failure Modes

#### **Input Errors**

**Error:** Participant provides non-textual response (e.g., just emoji, gibberish)  
**Cause:** Misunderstanding, testing system, accidental input  
**Detection:** Empty string after whitespace stripping, or no alphabetic characters  
**Recovery:**  
- Log as invalid response  
- Send clarification question: "I didn't quite catch that. Could you elaborate?"  
- Don't increment turn counter (give them a retry)  
- If 3 consecutive invalid → end interview gracefully

---

**Error:** Response is extremely long (>2000 words)  
**Cause:** Copy-paste from document, participant misunderstanding format  
**Detection:** Word count exceeds threshold  
**Recovery:**  
- Truncate to first 2000 words  
- Log truncation warning  
- Process truncated version  
- Add system note in next question: "That's a lot of detail—let me focus on..."

---

#### **Processing Errors**

**Error:** LLM extraction returns malformed JSON  
**Cause:** LLM hallucination, prompt confusion, API error  
**Detection:** JSON parsing failure  
**Recovery:**  
- Retry once with simplified prompt (remove context)  
- If retry fails → return empty GraphDelta  
- Use template fallback question ("Tell me more about your experience with this")  
- Log incident for schema prompt review

---

**Error:** Extracted graph delta violates schema rules (invalid edge source→target)  
**Cause:** LLM misinterprets ontology, participant describes novel relationship  
**Detection:** Schema validator raises ValidationError  
**Recovery:**  
- Log validation failure with specific rule violated  
- Reject entire delta (don't partially apply)  
- Use clarification question targeting the problematic part  
- If 2 consecutive validation failures → signal potential schema gap, continue with generic question

---

**Error:** Coreference resolution fails (pronoun links to nothing)  
**Cause:** No prior context, ambiguous reference  
**Detection:** Coreference algorithm returns None  
**Recovery:**  
- Extract pronoun phrase as new ambiguous node (type: "unclear_mention")  
- Generate clarification question: "When you said '{phrase}', what were you referring to?"  
- Next turn should resolve and replace ambiguous node

---

#### **Resource Errors**

**Error:** LLM API timeout (>10 seconds no response)  
**Cause:** API overload, network issue  
**Detection:** Request timeout exception  
**Recovery:**  
- Retry once with shorter context (remove history)  
- If retry fails → use cached previous question + variation  
- Log timeout incident  
- If 2 consecutive timeouts → pause interview, ask participant to wait 30s

---

**Error:** Graph exceeds size limit (>10,000 nodes)  
**Cause:** Runaway extraction, participant providing lists  
**Detection:** Node count exceeds threshold  
**Recovery:**  
- Force interview termination  
- Flag session as anomalous (possible spam or misuse)  
- Export graph snapshot before termination  
- Return error to operator, not participant

---

**Error:** Memory pressure (system RAM >90%)  
**Cause:** Large graph, multiple concurrent interviews, memory leak  
**Detection:** System monitoring  
**Recovery:**  
- Log warning  
- Disable graph snapshots (keep only final)  
- Continue interview if possible  
- If OOM imminent → gracefully end interview, export partial graph

---

#### **External Errors**

**Error:** LLM API rate limit exceeded  
**Cause:** Too many requests in short time  
**Detection:** HTTP 429 response  
**Recovery:**  
- Pause interview  
- Notify participant: "Experiencing technical delay, please wait..."  
- Wait for rate limit reset (read from headers)  
- Resume from last turn  
- If wait >2 minutes → offer to end interview and resume later

---

**Error:** Schema file not found or corrupted  
**Cause:** Configuration error, file system issue  
**Detection:** File not found or YAML parse error at startup  
**Recovery:**  
- Abort interview initialization  
- Return clear error to operator: "Schema file {path} missing or invalid"  
- Do not attempt to continue with partial schema  
- Fix file, restart

---

### Recovery Strategies Summary

**Retry:** For transient errors (LLM timeouts, API failures)  
- Max retries: 2  
- Backoff: 1 second, then 3 seconds

**Fallback:** For extraction failures  
- Use template questions (safe, generic)  
- Continue interview without graph update from problematic turn

**Degrade:** For resource constraints  
- Reduce graph snapshot frequency  
- Simplify LLM prompts (less context)  
- Switch to template-only mode

**Abort:** For critical failures  
- Schema invalid → Can't continue  
- Graph size explosion → Safety limit  
- Multiple consecutive failures → System unstable

**Fail-Safe Question:** "Tell me more about what matters most to you with this product."  
- Used when all else fails  
- Generic but always valid  
- Gives participant voice, might yield useful data anyway

---

## Section 9: Testing Strategy

### Unit Tests

**Schema Manager:**
- Valid schema loads successfully
- Invalid schema (missing fields) raises ValidationError with details
- Duplicate node type names rejected
- Edge type with non-existent source/target node types rejected
- Richness weight extraction returns correct float per type

**Interview Graph:**
- Adding node increments node count
- Adding duplicate node updates visit_count, doesn't duplicate
- Adding edge with non-existent nodes raises error
- Centrality calculation correct for star graph (center = 1.0, leaves = 0)
- Empty graph export produces valid empty GraphML

**Response Processor:**
- Valid response with clear entities extracts correct GraphDelta
- Ambiguous response returns partial delta with uncertainty flags
- Response with no entities returns empty delta (not error)
- Validation catches invalid edge type
- Coreference resolution links pronoun to last mentioned node

**Interview Manager:**
- Graph with unexplored seed node generates introduce_topic opportunity
- Graph with shallow recent node generates dig_deeper opportunity
- Dead-end node (3 probes, low richness) gets deprioritized
- Empty opportunity list returns termination signal
- Focus stack push/pop operates correctly

**Question Generator:**
- dig_deeper opportunity renders from template correctly
- Template variable substitution includes all node names
- LLM fallback triggers when no template matches
- Repetition check flags similar questions (>0.7 threshold)
- Generated question under 100 characters (reasonable length)

---

### Integration Tests

**Turn Loop:**
- Start interview → load schema → initialize graph with seeds → generate first question
- Process valid response → extract delta → update graph → generate follow-up → verify turn log
- Process 3 consecutive invalid responses → system uses fallback → continues
- Graph reaches coverage of all seeds after N turns (expected ~10-15 for 5 seeds)

**Schema Versioning:**
- Load schema v1.0 → process interview → export graph with version metadata
- Load schema v2.0 (different node types) → new interview uses new ontology
- Verify v1.0 and v2.0 graphs store separate versioning info

**Error Recovery:**
- Trigger LLM timeout (mock) → system retries → uses fallback → interview continues
- Trigger validation failure → system logs error → asks clarification → continues
- Trigger 2 consecutive failures → system logs incident → continues with safe question

**Dead-End Path:**
- Probe node 3 times with shallow responses → marked as dead-end → deprioritized
- Participant spontaneously mentions dead-end node → removed from dead-end set → re-prioritized

---

### Performance Tests

**Latency Benchmarks:**
- Measure end-to-end turn time for 10 consecutive turns (target: <3s 90th percentile)
- Profile individual components: extraction (target <2s), question gen (target <1s)
- Test with 100-word response (typical), 500-word response (large), 10-word response (minimal)

**Memory Usage:**
- Monitor graph size growth over 30-turn interview (expect ~50-200 nodes)
- Detect memory leaks (run 10 interviews sequentially, check for linear growth)
- Test with extreme case (participant lists 50 items in one response) → should not crash

**Scalability:**
- Run 5 concurrent interviews (simulate multi-participant study)
- Verify latency doesn't degrade >20%
- Check for resource contention (LLM API rate limits)

---

### Validation Tests

**Graph Quality:**
- Compare graphs from 5 interviews on same concept → check for consistent core nodes
- Human expert review: Does graph capture participant's key points?
- Blind test: Given transcript + graph, can researcher identify main themes?

**Interview Coherence:**
- Human eval: Do generated questions feel natural? (5-point scale)
- Repetition check: Measure % of questions flagged as similar
- Topic relevance: Do questions stay on-topic? (track off-topic rate)

**Richness Validity:**
- Correlate system richness scores with human expert ratings
- Test whether higher richness corresponds to more actionable insights
- Compare richness across different schema versions (which elicits more?)

---

### Regression Tests

**Core Functionality:**
- Re-run 10 "golden" interview transcripts through system
- Compare generated graphs to stored expected outputs
- Flag any deviations for review

**Schema Migration:**
- Load old interview (schema v1.0) → attempt to process with schema v2.0 → should fail gracefully
- Verify backward compatibility: v2.0 system can read v1.0 graphs

**Error Cases:**
- Intentionally trigger each error mode → verify correct recovery behavior
- Check logs contain sufficient debug info

---

## Section 10: Implementation Notes

### Recommended Libraries/Frameworks

**Core Stack:**
- **Python 3.11+**: Native `asyncio` for concurrent LLM calls
- **NetworkX 3.2+**: Proven graph library, rich algorithms
- **Pydantic 2.5+**: Strict validation, auto-generates schemas from manifests
- **PyYAML 6.0+**: Industry standard, stable
- **anthropic 0.18+** or **openai 1.10+**: LLM API clients with function calling support

**Optional Enhancements:**
- **Redis 5.0+**: Cache embeddings for node similarity (Phase 2)
- **Streamlit 1.30+**: Build debugging dashboard to visualize graph in real-time
- **pytest 7.4+**: Testing framework
- **loguru 0.7+**: Better logging than stdlib

**Rationale:**
- NetworkX chosen over graph databases (Neo4j) for POC: in-memory is sufficient, faster iteration
- Pydantic over manual validation: reduces bugs, auto-docs
- Anthropic Claude recommended for extraction: stronger structured output adherence than GPT-4
- Streamlit for dashboard: fastest way to visualize, crucial for debugging

---

### Known Pitfalls

**Pitfall 1: Over-Reliance on LLM "Reasoning"**  
LLMs are good at extraction but not at strategy. The Interview Manager must be algorithmic (Python logic), not LLM-prompted ("decide what to ask next").  
**Avoid:** Passing full graph to LLM and asking "what should I probe?"  
**Do:** Use deterministic priority ranking, let LLM only render phrasing.

**Pitfall 2: Context Window Creep**  
Passing full conversation history to LLM for every extraction hits token limits (~8k-32k) and slows down.  
**Avoid:** Sending all 30 turns to LLM  
**Do:** Sliding window of last 2-3 turns + schema definitions only.

**Pitfall 3: Graph Validation vs. Correction**  
Attempting to "fix" malformed LLM extractions risks hallucination propagation.  
**Avoid:** Auto-correcting edge types or node names  
**Do:** Reject and retry, or use template fallback. Clean failure better than polluted graph.

**Pitfall 4: Dead-End Spiral**  
If dead-end detection is too aggressive, system might stop probing valid but initially shallow nodes.  
**Avoid:** Marking dead-end after 1-2 probes  
**Do:** Require 3+ probes AND consistently low richness before abandoning.

**Pitfall 5: Schema Brittleness**  
Hardcoding node types makes experimentation painful.  
**Avoid:** Python classes like `class PainPoint(Node)`  
**Do:** Meta-schema loaded from YAML, dynamic validation via Pydantic.

---

### Optimization Opportunities

**Hot Path: Response Extraction**  
- Current: ~2 seconds (LLM call)  
- **Optimization:** Use smaller, faster model (e.g., GPT-4o-mini, Claude Haiku) for extraction only  
- **Trade-off:** Slightly lower accuracy for 3x speed gain  
- **Implementation:** Config flag `extraction_model` separate from `question_gen_model`

**Caching: Embeddings for Node Similarity**  
- Current: Compute similarity on-the-fly (slow)  
- **Optimization:** Pre-compute embeddings for all nodes, store in Redis, cosine similarity in <50ms  
- **Impact:** Faster coreference resolution, better duplicate detection  
- **Implementation:** Phase 1.5 (after POC validates core loop)

**Parallelization: Opportunity Ranking**  
- Current: Sequential evaluation of opportunities  
- **Optimization:** Parallel compute of priority scores (Python `concurrent.futures`)  
- **Impact:** Useful only if >50 opportunities (rare)  
- **Implementation:** Low priority unless profiling shows bottleneck

**Template Pre-compilation**  
- Current: Load templates from file every turn  
- **Optimization:** Load once at startup, store in memory  
- **Impact:** ~10ms saved per turn  
- **Implementation:** Week 1

---

### Development Phases

**Week 1-2: Core Infrastructure**
- Implement Schema Manager + Interview Graph
- Unit tests for graph operations
- Manual graph construction (hardcoded deltas)
- Deliverable: Graph can be built and queried

**Week 3-4: Extraction Pipeline**
- Implement Response Processor with LLM
- Test with 10 synthetic responses
- Validate against toy schema
- Deliverable: Response → GraphDelta working

**Week 5: Interview Logic**
- Implement Interview Manager (opportunity ranking)
- Implement Question Generator (templates + LLM)
- Deliverable: Can run 1 turn manually (paste response, get question)

**Week 6: Integration**
- Implement Interview Controller (full loop)
- Run 3 end-to-end synthetic interviews
- Deliverable: System can conduct full interview unattended

**Week 7-8: Refinement**
- Human testing (5 real participants)
- Tune richness scoring based on expert review
- Add Streamlit dashboard for graph visualization
- Deliverable: POC ready for pilot study

---

### References

**Graph Construction from Text:**
- Hogan et al. (2021): "Knowledge Graphs" - Survey of graph extraction methods
- Zhang et al. (2022): "LLM-based Information Extraction" - Function calling for structured outputs

**Interview Design:**
- Reynolds & Gutman (1988): "Laddering Theory" - Means-End Chain methodology (classic marketing research)
- Zaltman (1997): "Elicitation Techniques" - Deep metaphor exploration

**LLM Tool Usage:**
- Anthropic Docs: Function Calling - https://docs.anthropic.com/claude/docs/functions
- OpenAI Docs: Structured Outputs - https://platform.openai.com/docs/guides/structured-outputs

**Graph Analysis:**
- NetworkX Docs: https://networkx.org/documentation/stable/
- Hagberg et al. (2008): "Exploring Network Structure" - Algorithms reference

---

## Appendices

### Appendix A: Example Schema Manifests

Two complete example schemas are included in the original document:
1. **Means-End Chain Lens** (`schema_mec_v0.1.yaml`)
2. **Usage Script Lens** (`schema_script_v0.1.yaml`)

These should be implemented as starter templates in `/schemas/examples/`.

---

### Appendix B: Sample Turn Log Format

```json
{
  "session_id": "abc123",
  "turn_number": 5,
  "timestamp": "2025-11-28T10:35:22Z",
  "schema_version": "1.0.0",
  "participant_response": "I like the price, but I'm worried about quality...",
  "graph_delta": {
    "nodes_added": [
      {"type": "benefit", "label": "affordable_price", "quote": "I like the price"},
      {"type": "concern", "label": "quality_doubt", "quote": "worried about quality"}
    ],
    "edges_added": [
      {"type": "contradicts", "source": "affordable_price", "target": "quality_doubt"}
    ],
    "richness_score": 3.2
  },
  "interview_state": {
    "graph_size": {"nodes": 12, "edges": 8},
    "coverage_pct": 0.6,
    "top_opportunity": {
      "action": "dig_deeper",
      "target": "quality_doubt",
      "priority": 9.5
    }
  },
  "question_generated": "You mentioned concerns about quality. What specifically makes you worry about that?",
  "question_method": "template"
}
```

---

### Appendix C: Glossary

- **Graph Delta:** The set of new nodes and edges extracted from a single response
- **Opportunity:** A potential direction for the next question (e.g., probe node X, connect Y and Z)
- **Richness:** Quantitative measure of how much new information a response added to the graph
- **Dead-End:** A node that has been probed multiple times with diminishing returns
- **Focus Stack:** A LIFO data structure tracking the current inquiry thread
- **Seed Node:** Initial concept from research objectives that must be explored
- **Schema Manifest:** YAML file defining the graph ontology (node/edge types, prompts, weights)
- **Turn:** One cycle of participant response → system question
- **Coreference:** Linking pronouns/vague references to specific graph nodes

---

## Document Status

**Review Checklist:**
- [ ] All components have clear interfaces
- [ ] Error handling covers expected failure modes
- [ ] Algorithms described conceptually (not as code)
- [ ] Parameters have justifications and ranges
- [ ] Testing strategy covers critical paths
- [ ] Known pitfalls documented
- [ ] References support key decisions

**Next Steps:**
1. Review by domain expert (marketing research)
2. Review by ML engineer (implementation feasibility)
3. Create schema examples for 3 FMCG categories
4. Set up development environment
5. Begin Week 1 implementation

---

**Document End**