# AI Interview System - Implementation Plan

**Version:** 1.0
**Last Updated:** 2025-11-28
**Current Phase:** Phase 0 Complete ✅ → Ready for Phase 1

---

## Overview

This document tracks the phased implementation of the AI-based Graph Interviewing System. The system conducts adaptive interviews while building knowledge graphs that represent participants' mental models.

### Key Architecture Decisions

- **Dual LLM Architecture**: Kimi K2 for fast graph extraction + Claude Sonnet for natural question generation
- **Schema-Driven Design**: YAML-based mental model definitions (Means-End Chain, Usage Script)
- **Graph Processing**: NetworkX for knowledge graph operations
- **UI Framework**: Gradio 6.x for HuggingFace Space deployment
- **Package Management**: uv for fast dependency management

---

## Phase Status

| Phase | Status | Completion Date | Description |
|-------|--------|----------------|-------------|
| Phase 0 | ✅ Complete | 2025-11-28 | Foundation & Skeleton |
| Phase 1 | 🔄 Next | - | Core Infrastructure |
| Phase 2 | ⏸️ Planned | - | Extraction Pipeline |
| Phase 3 | ⏸️ Planned | - | Interview Logic |
| Phase 4 | ⏸️ Planned | - | Integration & UI |

---

## Phase 0: Foundation ✅ COMPLETE

**Goal:** Establish project structure, configuration files, and UI skeleton

### Completed Tasks

#### 1. Project Structure
- ✅ Created HuggingFace Space-compatible directory layout
- ✅ Organized code into `src/core/`, `src/interview/`, `src/llm/`, `src/ui/`
- ✅ Set up `schemas/`, `configs/`, `prompts/`, `tests/` directories

**Files Created:**
- Root structure with `app.py` entry point
- `.gitignore` with project-specific exclusions
- `README.md` with HF frontmatter

#### 2. Schema YAMLs
- ✅ Enhanced `schemas/means_end_chain_v0.1.yaml`
- ✅ Enhanced `schemas/usage_script_v0.1.yaml`

**Key Features:**
- Schema version tracking
- Node types with richness weights and probing prompts
- Edge types with validation rules
- Interview configuration (opening questions, probing strategies)
- Richness scoring formulas
- LLM extraction prompts embedded in schema

#### 3. Data Models
- ✅ Created `src/core/data_models.py` with Pydantic v2 models

**Models Implemented:**
- `Node` - Graph node with metadata
- `Edge` - Graph edge with quotes
- `GraphDelta` - Response extraction results
- `InterviewState` - Session state tracking
- `OpportunityScore` - Question selection ranking
- `InterviewConfig` - Configuration settings

#### 4. Configuration Files
- ✅ `configs/model_config.yaml` - LLM provider configuration
- ✅ `configs/default_interview.yaml` - Interview parameters
- ✅ `.env.example` - Environment template

**LLM Configuration:**
- Graph processing: Kimi K2 (moonshot-v1-32k)
- Question generation: Claude Sonnet 4.5
- Fallback models configured

#### 5. Prompt Templates
- ✅ `prompts/extraction_prompts.yaml` - Graph extraction prompts
- ✅ `prompts/question_templates.yaml` - Question generation templates

**Template Types:**
- `dig_deeper` - Probing existing concepts
- `connect_concepts` - Exploring relationships
- `introduce_topic` - New concept introduction

#### 6. Dependency Management
- ✅ `requirements.txt` - HF Space compatible
- ✅ `pyproject.toml` - uv configuration with hatchling

**Key Dependencies:**
- gradio>=6.0.0,<7.0.0
- pydantic>=2.11.10,<=2.12.4
- anthropic>=0.40.0
- openai>=1.54.0
- networkx>=3.2.1

**Fixes Applied:**
- Added `[tool.hatch.build.targets.wheel]` with `packages = ["src"]`
- Migrated from `tool.uv.dev-dependencies` to `[dependency-groups]`

#### 7. Gradio Interface Skeleton
- ✅ `src/ui/gradio_app.py` - Full UI with placeholder logic
- ✅ `app.py` - HF Space entry point

**UI Features:**
- Chat interface with history
- Turn counter display
- Graph statistics panel (placeholder)
- Session info display
- Responsive layout (2:1 column ratio)

**Gradio 6.x Compatibility Fixes:**
- Removed unsupported `theme` parameter from `gr.Blocks()`
- Removed unsupported `css` parameter
- Removed `type="messages"` from `gr.Chatbot()`
- Removed deprecated `show_copy_button` parameter
- Uses minimal compatible parameters: `title`, `label`, `height`, `autoscroll`

#### 8. Test Fixtures
- ✅ `tests/fixtures/sample_responses.json`

**Sample Data:**
- Means-End Chain examples (4 turns)
- Usage Script examples (3 turns)
- Edge cases (empty responses, ambiguous text, off-topic)

### Phase 0 Acceptance Criteria ✅

- [x] Project structure matches HF Space requirements
- [x] `uv sync` completes without errors
- [x] `uv run python app.py` launches Gradio interface
- [x] User can type responses and see placeholder questions
- [x] No runtime errors in UI interaction
- [x] All configuration files load successfully
- [x] README includes deployment instructions

### Known Limitations (Expected)

- No real LLM integration (placeholder questions only)
- No graph construction from responses
- No intelligent question selection
- Graph statistics show zeros
- No session persistence

---

## Phase 1: Core Infrastructure 🔄 NEXT

**Goal:** Implement schema management and graph operations

**Estimated Effort:** 2-3 days

### Tasks

#### 1.1 Schema Manager
**File:** `src/core/schema_manager.py`

**Responsibilities:**
- Load and validate YAML schemas
- Provide typed access to node types, edge types, rules
- Load interview configuration
- Load richness scoring formulas
- Cache parsed schemas

**Key Methods:**
```python
class SchemaManager:
    def __init__(self, schema_path: str)
    def load_schema(self) -> Dict
    def validate_schema(self, schema: Dict) -> bool
    def get_node_types(self) -> List[NodeType]
    def get_edge_types(self) -> List[EdgeType]
    def get_interview_config(self) -> InterviewConfig
    def get_richness_weights(self) -> Dict[str, float]
    def get_probing_prompt(self, node_type: str) -> str
```

**Dependencies:**
- `pyyaml` for YAML parsing
- `pydantic` for validation
- `src/core/data_models.py` for type definitions

**Acceptance Criteria:**
- [ ] Loads `means_end_chain_v0.1.yaml` without errors
- [ ] Validates schema structure
- [ ] Returns typed node/edge configurations
- [ ] Handles missing/malformed schemas gracefully
- [ ] Unit tests cover happy path + error cases

#### 1.2 Interview Graph
**File:** `src/core/interview_graph.py`

**Responsibilities:**
- Wrap NetworkX graph with domain-specific operations
- Add/update nodes and edges
- Calculate coverage metrics
- Compute richness scores
- Identify expansion opportunities
- Export graph state

**Key Methods:**
```python
class InterviewGraph:
    def __init__(self, schema_manager: SchemaManager)
    def add_node(self, node: Node) -> bool
    def add_edge(self, edge: Edge) -> bool
    def apply_delta(self, delta: GraphDelta) -> None
    def get_coverage(self) -> Dict[str, float]
    def calculate_richness(self) -> float
    def get_expansion_opportunities(self) -> List[OpportunityScore]
    def get_node(self, node_id: str) -> Optional[Node]
    def get_neighbors(self, node_id: str) -> List[Node]
    def to_dict(self) -> Dict
    def from_dict(self, data: Dict) -> None
```

**Graph Operations:**
- Node deduplication (merge similar concepts)
- Edge validation (respect schema rules)
- Coverage calculation (% of node types filled)
- Richness scoring (weighted sum of nodes + edges)
- Opportunity ranking (unexplored nodes, shallow branches)

**Dependencies:**
- `networkx` for graph storage
- `src/core/schema_manager.py` for validation rules
- `src/core/data_models.py` for type definitions

**Acceptance Criteria:**
- [ ] Creates empty graph from schema
- [ ] Adds nodes and edges with validation
- [ ] Calculates coverage accurately
- [ ] Computes richness scores per schema weights
- [ ] Identifies expansion opportunities
- [ ] Serializes/deserializes graph state
- [ ] Unit tests cover all operations

#### 1.3 Unit Tests
**Files:**
- `tests/test_schema_manager.py`
- `tests/test_interview_graph.py`

**Test Coverage:**
- Schema loading (valid, invalid, missing)
- Graph operations (add, update, query)
- Richness calculations
- Coverage metrics
- Opportunity ranking
- Edge case handling

**Acceptance Criteria:**
- [ ] All tests pass with `pytest`
- [ ] Coverage >80% for core modules
- [ ] Edge cases documented and tested

### Phase 1 Deliverables

- [ ] `src/core/schema_manager.py` - Full implementation
- [ ] `src/core/interview_graph.py` - Full implementation
- [ ] `tests/test_schema_manager.py` - Comprehensive tests
- [ ] `tests/test_interview_graph.py` - Comprehensive tests
- [ ] Documentation in docstrings
- [ ] Update this plan with completion status

### Phase 1 Dependencies

**Prerequisites:**
- ✅ Phase 0 complete
- ✅ Schema YAMLs defined
- ✅ Data models implemented

**Blocks:**
- Phase 2 (needs graph to populate)
- Phase 3 (needs graph for opportunity ranking)

---

## Phase 2: Extraction Pipeline ⏸️ PLANNED

**Goal:** Implement LLM-powered graph extraction from user responses

**Estimated Effort:** 3-4 days

### Tasks

#### 2.1 LLM Client Factory
**File:** `src/llm/client_factory.py`

**Responsibilities:**
- Abstract LLM provider differences
- Load configuration from `configs/model_config.yaml`
- Instantiate appropriate client (Kimi, Claude, OpenAI)
- Handle fallbacks

**Key Methods:**
```python
class LLMClientFactory:
    @staticmethod
    def create_client(provider: str, task: str) -> BaseLLMClient
    @staticmethod
    def load_config() -> Dict
```

#### 2.2 Kimi Client
**File:** `src/llm/kimi_client.py`

**Responsibilities:**
- Implement Moonshot Kimi API integration
- Optimized for fast extraction tasks
- Handle structured output (JSON)

**Key Methods:**
```python
class KimiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str)
    async def extract_graph_delta(
        self,
        response: str,
        current_graph: InterviewGraph,
        extraction_prompt: str
    ) -> GraphDelta
```

#### 2.3 Anthropic Client
**File:** `src/llm/anthropic_client.py`

**Responsibilities:**
- Implement Anthropic Claude API integration
- Optimized for natural question generation
- Handle streaming responses (optional)

**Key Methods:**
```python
class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str)
    async def generate_question(
        self,
        template: str,
        context: Dict,
        temperature: float = 0.7
    ) -> str
```

#### 2.4 Response Processor
**File:** `src/interview/response_processor.py`

**Responsibilities:**
- Orchestrate LLM extraction
- Parse LLM output into GraphDelta
- Validate extracted nodes/edges against schema
- Handle extraction failures gracefully

**Key Methods:**
```python
class ResponseProcessor:
    def __init__(self, llm_client: BaseLLMClient, schema_manager: SchemaManager)
    async def process_response(
        self,
        user_response: str,
        current_graph: InterviewGraph,
        turn_number: int
    ) -> GraphDelta
```

#### 2.5 Integration Tests
**Files:**
- `tests/test_llm_clients.py`
- `tests/test_response_processor.py`

**Test Strategy:**
- Mock LLM APIs for unit tests
- Use fixture responses for integration tests
- Test error handling (API failures, malformed output)

### Phase 2 Deliverables

- [ ] `src/llm/client_factory.py`
- [ ] `src/llm/kimi_client.py`
- [ ] `src/llm/anthropic_client.py`
- [ ] `src/interview/response_processor.py`
- [ ] Integration tests with sample responses
- [ ] Error handling and logging
- [ ] API key validation

### Phase 2 Dependencies

**Prerequisites:**
- ✅ Phase 0 complete
- ✅ Prompt templates defined
- ✅ GraphDelta model implemented
- [ ] Phase 1 complete (needs InterviewGraph)

**Blocks:**
- Phase 4 (needs response processing)

---

## Phase 3: Interview Logic ⏸️ PLANNED

**Goal:** Implement intelligent question selection and generation

**Estimated Effort:** 2-3 days

### Tasks

#### 3.1 Interview Manager
**File:** `src/interview/manager.py`

**Responsibilities:**
- Rank expansion opportunities in graph
- Select next question strategy (dig deeper, connect, introduce)
- Maintain focus stack (current exploration path)
- Detect interview completion

**Key Methods:**
```python
class InterviewManager:
    def __init__(self, graph: InterviewGraph, schema_manager: SchemaManager)
    def rank_opportunities(self) -> List[OpportunityScore]
    def select_next_strategy(self) -> QuestionStrategy
    def should_continue(self) -> bool
    def get_interview_summary(self) -> Dict
```

**Opportunity Ranking Algorithm:**
1. **Coverage Score**: Favor underexplored node types
2. **Depth Score**: Balance breadth vs depth
3. **Recency Score**: Penalize recently visited nodes
4. **Focus Bonus**: Reward staying on current topic

#### 3.2 Question Generator
**File:** `src/interview/question_generator.py`

**Responsibilities:**
- Load question templates
- Fill templates with graph context
- Use LLM to naturalize questions
- Handle special cases (opening, closing, transitions)

**Key Methods:**
```python
class QuestionGenerator:
    def __init__(self, llm_client: BaseLLMClient, schema_manager: SchemaManager)
    async def generate_question(
        self,
        strategy: QuestionStrategy,
        opportunity: OpportunityScore,
        graph: InterviewGraph
    ) -> str
    def get_opening_question(self) -> str
    def get_closing_question(self) -> str
```

**Question Templates:**
- `dig_deeper`: "What specifically about {concept} matters to you?"
- `connect_concepts`: "How does {concept_a} relate to {concept_b}?"
- `introduce_topic`: "Tell me about {new_topic}."

#### 3.3 Unit Tests
**Files:**
- `tests/test_interview_manager.py`
- `tests/test_question_generator.py`

### Phase 3 Deliverables

- [ ] `src/interview/manager.py`
- [ ] `src/interview/question_generator.py`
- [ ] Opportunity ranking algorithm
- [ ] Question template system
- [ ] Unit tests for ranking and generation
- [ ] Documentation of strategy selection logic

### Phase 3 Dependencies

**Prerequisites:**
- ✅ Phase 0 complete
- ✅ Question templates defined
- [ ] Phase 1 complete (needs InterviewGraph)
- [ ] Phase 2 complete (needs LLM clients)

**Blocks:**
- Phase 4 (needs question generation)

---

## Phase 4: Integration & UI ⏸️ PLANNED

**Goal:** Wire up all components and complete end-to-end interview flow

**Estimated Effort:** 2-3 days

### Tasks

#### 4.1 Interview Controller
**File:** `src/interview/controller.py`

**Responsibilities:**
- Orchestrate entire interview turn loop
- Initialize graph and managers
- Process responses and generate questions
- Manage session state
- Handle errors and edge cases

**Key Methods:**
```python
class InterviewController:
    def __init__(self, schema_path: str, config_path: str)
    async def start_interview(self) -> str  # Returns first question
    async def process_turn(self, user_response: str) -> Tuple[GraphDelta, str]
    def get_graph_stats(self) -> Dict
    def export_session(self) -> Dict
    def is_complete(self) -> bool
```

**Turn Loop:**
1. User submits response
2. ResponseProcessor extracts GraphDelta
3. InterviewGraph applies delta
4. InterviewManager ranks opportunities
5. QuestionGenerator creates next question
6. Return question to UI

#### 4.2 Gradio UI Integration
**File:** `src/ui/gradio_app.py` (update)

**Changes:**
- Replace placeholder logic with InterviewController
- Wire up graph stats display
- Add session export button
- Display richness score
- Show coverage visualization (optional)

**Updated Methods:**
```python
def __init__(self):
    self.controller = InterviewController(
        schema_path=os.getenv("DEFAULT_SCHEMA"),
        config_path=os.getenv("DEFAULT_INTERVIEW_CONFIG")
    )

async def process_response(self, user_response: str, history: List):
    graph_delta, next_question = await self.controller.process_turn(user_response)
    stats = self.controller.get_graph_stats()
    # Update UI with stats and question
```

#### 4.3 End-to-End Testing
**Files:**
- `tests/test_interview_controller.py`
- `tests/test_e2e_interview.py`

**Test Scenarios:**
- Full interview flow with fixture responses
- Graph construction over multiple turns
- Error recovery (LLM failures, network issues)
- Session export/import
- Interview completion detection

#### 4.4 Session Management
**Enhancements:**
- Save interview sessions to `data/interviews/`
- Export graph as JSON
- Export conversation transcript
- Generate summary report

### Phase 4 Deliverables

- [ ] `src/interview/controller.py` - Full implementation
- [ ] Updated `src/ui/gradio_app.py` - Integrated with controller
- [ ] Real-time graph stats display
- [ ] Session export functionality
- [ ] End-to-end tests
- [ ] Error handling and logging
- [ ] Performance optimization (caching, async)

### Phase 4 Dependencies

**Prerequisites:**
- ✅ Phase 0 complete
- [ ] Phase 1 complete
- [ ] Phase 2 complete
- [ ] Phase 3 complete

**Enables:**
- Production deployment to HuggingFace Spaces
- User testing and feedback collection

---

## Testing Strategy

### Unit Tests (Per Phase)
- Each module has dedicated test file
- Coverage target: >80%
- Run with: `pytest tests/`

### Integration Tests (Phase 2, 4)
- Test component interactions
- Mock external APIs (LLMs)
- Use fixture data from `tests/fixtures/`

### End-to-End Tests (Phase 4)
- Full interview simulation
- Multi-turn conversations
- Graph construction validation

### Manual Testing Checklist (Phase 4)
- [ ] Complete 5+ full interviews with different products
- [ ] Test edge cases (empty responses, off-topic)
- [ ] Verify graph construction accuracy
- [ ] Check question quality and relevance
- [ ] Test session export/import
- [ ] Verify HF Space deployment

---

## Deployment Checklist

### Pre-Deployment
- [ ] All Phase 0-4 tests passing
- [ ] Manual testing complete
- [ ] README updated with usage instructions
- [ ] API keys configured as HF Secrets
- [ ] Performance benchmarks acceptable

### HuggingFace Space Deployment
1. Create HF Space with Gradio SDK
2. Clone Space repository
3. Copy files:
   - `app.py`
   - `requirements.txt`
   - `README.md` (with HF frontmatter)
   - `src/`
   - `schemas/`
   - `configs/`
   - `prompts/`
4. Configure secrets in Space Settings:
   - `KIMI_API_KEY`
   - `ANTHROPIC_API_KEY`
5. Commit and push
6. Monitor Space logs for errors
7. Test deployed interface

### Post-Deployment
- [ ] Test public Space URL
- [ ] Monitor LLM API usage and costs
- [ ] Collect user feedback
- [ ] Iterate based on usage patterns

---

## Risk Mitigation

### Technical Risks

**Risk:** LLM extraction quality varies
- **Mitigation:** Use structured prompts with examples, validate against schema
- **Fallback:** Allow manual graph editing (future feature)

**Risk:** API rate limits or costs
- **Mitigation:** Implement caching, request throttling, cost monitoring
- **Fallback:** Switch to fallback models (Claude Haiku, GPT-4o-mini)

**Risk:** Graph becomes too large/complex
- **Mitigation:** Set max nodes limit, prune low-confidence nodes
- **Fallback:** Focus on coverage rather than exhaustive depth

### Process Risks

**Risk:** Phase interdependencies cause delays
- **Mitigation:** This implementation plan clarifies dependencies
- **Fallback:** Can parallelize some Phase 2 and Phase 3 work

**Risk:** Scope creep during implementation
- **Mitigation:** Stick to minimal POC scope, defer nice-to-haves
- **Fallback:** Maintain backlog of future enhancements

---

## Future Enhancements (Post-Phase 4)

### Near-Term (After MVP)
- Graph visualization (D3.js or Plotly)
- Multi-schema support (switch between Means-End and Usage Script)
- Interview replay and analysis
- Export to research formats (CSV, GraphML)

### Medium-Term
- Interviewer personality customization
- Multi-language support
- Voice input/output
- Collaborative interviewing (multiple participants)

### Long-Term
- Automated insight generation from graphs
- Cross-interview pattern analysis
- Integration with research platforms
- Custom schema builder UI

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-28 | Initial plan created, Phase 0 marked complete | POC Team |

---

## References

- Original Design Document: `context/design_doc.md`
- README: `README.md`
- Schema Definitions: `schemas/`
- Sample Responses: `tests/fixtures/sample_responses.json`

---

**Next Action:** Begin Phase 1 implementation - start with `src/core/schema_manager.py`
