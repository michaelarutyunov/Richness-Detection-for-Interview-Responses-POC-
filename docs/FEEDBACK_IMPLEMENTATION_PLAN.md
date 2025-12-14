# Implementation Plan: External Feedback Review
**Date**: 2025-12-13
**Source**: Reviews/20251212_2.md
**Status**: Validated and Ready for Implementation

## Executive Summary

External LLM review identified **5 critical issues** causing:
- Coverage stuck at gap=1 for 8 turns
- Premature fatigue detection despite engaged responses
- Shallow graphs (wide but never reaching values)
- Over-mechanical strategy switching
- Momentum misclassification (hedging over-penalized)

**Validation Result**: All feedback is **ACCURATE** ✅
**Impact**: ~75% of fixes are configuration/prompt engineering (no code architecture changes required)

---

## Phase 1: Quick Wins - Configuration Adjustments (2-3 hours)

### 1.1 Fix Coverage Exhaustion Logic
**File**: `/src/config/interview_logic.yaml` (lines 236-242)

**Current**:
```yaml
coverage_quality:
  weight: 1.0
  first_touch_boost: 2.5
  exhaustion_threshold: 2
  exhaustion_penalty: 0.15  # TOO WEAK
```

**Change to**:
```yaml
coverage_quality:
  weight: 1.0
  first_touch_boost: 2.5
  exhaustion_threshold: 2
  exhaustion_penalty: 1.2  # Strong deterrent - STOP drilling exhausted elements
```

**Rationale**: Current 0.15 penalty is ineffective. After 2 failed probes, multiply score by 1.2 (down from 2.5 boost) creates clear signal to move on.

---

### 1.2 Reduce Breadth Over-Boosting
**File**: `/src/config/interview_logic.yaml`

**Current**:
```yaml
momentum_alignment:
  breadth_boost: 1.5   # Line 204

branch_health:
  breadth_boost: 1.8   # Line 228
```

**Change to**:
```yaml
momentum_alignment:
  breadth_boost: 1.1   # Gentle nudge, not forcing

branch_health:
  breadth_boost: 1.2   # Reduced from 1.8
```

**Rationale**: For MEC interviews, depth > breadth. Two strong breadth boosts overwhelm depth incentives.

---

### 1.3 Soften Depth Penalties
**File**: `/src/config/interview_logic.yaml`

**Current**:
```yaml
momentum_alignment:
  depth_penalty: 0.5   # Line 205

branch_health:
  depth_penalty: 0.3        # Line 229
  severe_depth_penalty: 0.1 # Line 231 - REMOVE THIS
```

**Change to**:
```yaml
momentum_alignment:
  depth_penalty: 0.2   # Was 0.5

branch_health:
  depth_penalty: 0.15  # Was 0.3
  # REMOVE severe_depth_penalty entirely - too destructive
```

**Rationale**: Current penalties compound multiplicatively (0.5 × 0.3 = 0.15 or 0.5 × 0.1 = 0.05), making depth impossible.

---

### 1.4 Slow Down Branch Staleness Detection
**File**: `/src/config/interview_logic.yaml` (line 227)

**Current**:
```yaml
branch_health:
  stale_threshold: 2  # TOO AGGRESSIVE
```

**Change to**:
```yaml
branch_health:
  stale_threshold: 4  # Allow 4 turns before declaring stale
```

**Rationale**: MEC laddering requires 2-3 clarifying questions. Declaring stale after 2 turns prevents depth.

---

### 1.5 Fix Reflection Mode Schema Leakage
**File**: `/src/config/interview_logic.yaml` (line 259)

**Issue**: `min_value_nodes` is schema-specific (violates schema-agnostic design). Code already uses `min_terminal_nodes` internally for schema-agnostic terminal detection via `schema.is_terminal_type()`.

**Current**:
```yaml
reflection_mode:
  min_value_nodes: 1  # Schema-specific leak + unreachable threshold
```

**Change to**:
```yaml
reflection_mode:
  min_terminal_nodes: 0  # Schema-agnostic + relaxed threshold
```

**Rationale**:
- Code already supports `min_terminal_nodes` parameter (arbitration.py:900, 911)
- Uses `schema.is_terminal_type()` for methodology-agnostic detection (works with MEC value nodes, JTBD constraint nodes, etc.)
- Setting to `0` removes terminal node requirement (addresses feedback that values are unreachable)
- Reflection triggers based on: coverage complete + no new nodes (terminal nodes no longer required)

---

## Phase 2: Fix Momentum Assessment (1-2 hours)

### 2.1 Redefine Momentum Criteria
**File**: `/src/decision/extraction.py` (lines 551-573)

**Current**:
```python
system_prompt = """You are assessing respondent engagement in a qualitative interview.

HIGH momentum indicators:
- Long, elaborated responses
- Unprompted examples or stories
- Emotional language or emphasis
- Self-initiated connections ("and that's also why...")
- Enthusiasm, energy in the response

LOW momentum indicators:
- Short, closed responses ("yeah", "I guess")
- Repetition of previous answers
- Hedging, uncertainty ("I'm not sure really")
- Deflection or topic avoidance
- Fatigue signals

NEUTRAL is the default for typical responses.
```

**Change to**:
```python
system_prompt = """You are assessing respondent engagement in a qualitative interview.

HIGH momentum indicators:
- Long, elaborated responses with detail
- Unprompted examples, stories, or concrete scenarios
- Emotional language or emphasis
- Self-initiated connections ("and that's also why...")
- Enthusiasm, energy in the response

NEUTRAL momentum indicators (this is the EXPECTED baseline):
- Coherent answers with some detail
- No example/story but not avoiding the topic
- Mild emotional tone
- Standard conversational engagement
- May include THINKING-ALOUD hedging ("I guess", "to be honest") if accompanied by elaboration

LOW momentum indicators:
- Short AND unelaborated responses ("yeah", "dunno")
- Repetition of previous content without new insight
- Signs of withdrawal or disinterest
- Deflection or topic avoidance
- Fatigue signals (sighing, "I dunno… it's whatever")
- Hedging ONLY WHEN the entire response stays vague

CRITICAL BALANCING RULES:
- If the response contains a mix of signals (e.g., hedging but also elaboration), classify as NEUTRAL
- Long elaboration outweighs hedging
- Emotional depth outweighs uncertainty
- A concrete example outweighs brevity
- Do NOT classify as "low" if the respondent expresses uncertainty while still giving a meaningful explanation
- Hedging during reasoning/thinking is NORMAL in exploratory interviews - not low engagement
- Only assign LOW when the pattern of disengagement holds across the response

Respond with JSON:
{
  "level": "high" | "neutral" | "low",
  "indicators": ["list", "of", "observed", "signals"]
}"""
```

---

### 2.2 Increase History Window
**File**: `/src/decision/extraction.py` (line 577)

**Current**:
```python
for turn in history.get_recent(3):
```

**Change to**:
```python
for turn in history.get_recent(5):
```

**Rationale**: Trend matters more than single-turn classification. 5 turns provide better pattern detection.

---

## Phase 3: Schema Adjustments (1-2 hours)

### 3.1 Remove "Prefer Concrete" Directive
**File**: `/src/config/schemas/means_end_chain.yaml` (around line 170)

**Current**:
```yaml
classification_guidance: |
  When uncertain between adjacent levels, prefer the more concrete classification.
```

**Change to**:
```yaml
classification_guidance: |
  When uncertain between levels, choose the HIGHER level if the respondent expresses:
  - Personal meaning or subjective interpretation
  - Emotional reaction or feeling
  - Social significance or identity

  Choose the LOWER level only if the statement is:
  - Purely observational without interpretation
  - Factual description without personal meaning
```

**Rationale**: Current rule traps nodes at concrete levels, preventing ladder climbing.

---

### 3.2 Narrow Attribute Definition
**File**: `/src/config/schemas/means_end_chain.yaml` (lines 14-40)

**Current**:
```yaml
attribute:
  llm_prompt: |
    Observable, sensory, or directly measurable product features.

    **CRITICAL: Sensory and Physical Properties ARE ATTRIBUTES**
    - Taste: sweet, bitter, salty, rich, bland, sour, umami
    - Texture: smooth, creamy, watery, thick, thin, viscous, gritty, foamy, frothy
    [extensive list...]
```

**Change to**:
```yaml
attribute:
  llm_prompt: |
    Observable product features that can be verified WITHOUT using the product.

    **Physical & Composition Attributes** (objective, measurable):
    - Composition: contains caffeine, made of leather, organic ingredients
    - Packaging: bottle size, shape, material, labeling
    - Physical properties: weight, volume, temperature

    **Sensory Attributes** (observable through senses):
    - Basic appearance: color, clarity, foam presence
    - Basic texture: observed consistency (thick vs thin)

    **NOT attributes** (these require USE → they are consequences):
    - Sensory EXPERIENCES: "creamy mouthfeel", "pleasant sensation"
    - Performance: "easy to mix", "keeps drinks cold"
    - Perceptual interpretations: "appears premium", "looks appetizing"

    **Boundary cases** - If sensory property involves subjective interpretation,
    classify as functional_consequence, not attribute.
```

**Rationale**: Current definition is too broad, inflating attribute layer and creating malformed ladders.

---

### 3.3 Broaden Value Node Accessibility
**File**: `/src/config/schemas/means_end_chain.yaml` (lines 91-103)

**Current**:
```yaml
value:
  llm_prompt: |
    Fundamental life goals, deeply held principles, or end-state aspirations.
    Examples: 'security', 'family harmony', 'self-actualization'
    Must be abstract and universally meaningful - not product-specific outcomes.
```

**Change to**:
```yaml
value:
  llm_prompt: |
    Core personal values or life themes that matter beyond this specific product.

    **Terminal values** (universally meaningful):
    - security, freedom, belonging, achievement, harmony

    **Everyday wellbeing values** (broader life themes):
    - "starting the day right", "being productive", "feeling in control"
    - "maintaining wellness", "enjoying small pleasures", "peace of mind"

    If a psychosocial consequence relates to broader life goals or daily wellbeing,
    promote to value with a LOWER threshold than universal ideals.

    Examples:
    - "feel confident" → psychosocial
    - "confidence in social situations" → value (relates to belonging/social harmony)
    - "pleasant morning routine" → value (relates to wellbeing)
```

**Rationale**: Current definition is too strict for consumer interviews. Allow everyday wellbeing themes as values.

---

### 3.4 Make Coverage Type-Agnostic
**File**: `/src/config/schemas/means_end_chain.yaml` (add new section)

**Add**:
```yaml
coverage_guidance: |
  RTB (Reason to Believe) and Promise elements can be satisfied by ANY node type,
  not just attributes.

  Coverage is satisfied when the respondent provides ANY meaningful reaction about
  the reference element, including:
  - Attribute observation ("it's thick")
  - Functional interpretation ("makes it mix better")
  - Psychosocial reaction ("I trust natural processes")
  - Skepticism or curiosity ("I don't know what enzymes are")

  The node_type does NOT need to match the element category.
```

**Rationale**: Current coverage logic expects attribute-level nodes for RTB, but users generate functional/psychosocial reactions.

---

## Phase 4: Coverage Logic Implementation (2-3 hours)

### 4.1 Verify Coverage Detection Logic
**File**: `/src/core/state.py` (CoverageState class)

**Investigation needed**:
```python
# In CoverageState._recompute_gaps():
# Check if gap resolution requires node type matching
# If so, change to type-agnostic "any node about this element" logic
```

**Expected change**:
- Current: Checks if element has nodes of matching type (e.g., RTB requires attribute node)
- New: Checks if element has ANY nodes with meaningful reaction, regardless of type

**Test case**: RTB element "Made with enzyme process"
- User says "I guess enzymes are natural" (functional_consequence)
- Should satisfy coverage gap, not require attribute-type node

---

## Phase 5: Optional Enhancements (3-4 hours)

### 5.1 Add Laddering Protocol
**File**: `/src/config/interview_logic.yaml` (add to vertical_laddering scorer)

**Add**:
```yaml
vertical_laddering:
  weight: 1.0
  boost: 1.5
  value_proximity_boost: 1.8
  value_closure_boost: 2.0
  near_value_depth: 2

  # NEW: Automatic depth prioritization for psychosocial nodes
  psychosocial_depth_priority: true
  psychosocial_depth_turns: 2  # Force 2 depth turns after psychosocial node
```

**Rationale**: Ensure ladder climbs from psychosocial → value by default.

---

### 5.2 Add Uncertainty Follow-Up Strategy
**File**: `/src/decision/strategy.py` (add new strategy)

**Add**:
```yaml
# In interview_logic.yaml strategies section:

acknowledge_uncertainty:
  intent: >
    Respondent doesn't know something but shows curiosity or concern.
    Redirect from explanation to personal reaction.
  applies_when: "Knowledge ceiling detected but emotional signal present"
  focus: "The uncertain topic"
  suggested_tactics:
    - reaction_elicitation
    - open_probe
  llm_guidance: >
    Don't explain the concept - ask how they FEEL about not knowing.
    "You mentioned you're not sure what enzymes are - does that matter to you?"
    Shift from knowledge to personal meaning.
```

---

## Phase 6: Arbitration Refinements (3-4 hours, optional)

### 6.1 Soften Multiplicative Compounding
**File**: `/src/decision/arbitration.py` (lines 1027-1042)

**Current**:
```python
total_score = 1.0
for scorer_name, scorer in self.scorers.items():
    multiplier = scorer.score(...)
    weight = scorer.weight
    weighted_score = multiplier ** weight
    total_score *= weighted_score
```

**Option 1: Mixed scoring (multiplicative for boosts, additive for penalties)**
```python
total_score = 1.0
penalty_sum = 0.0

for scorer_name, scorer in self.scorers.items():
    multiplier = scorer.score(...)
    weight = scorer.weight

    if multiplier < 1.0:  # Penalty
        penalty_sum += (1.0 - multiplier) * weight
    else:  # Boost
        weighted_score = multiplier ** weight
        total_score *= weighted_score

final_score = total_score * (1.0 - penalty_sum)  # Apply penalty sum
```

**Rationale**: Prevents single penalty from wiping out all boosts, while maintaining veto power.

**Risk**: Architectural change - needs thorough testing.

---

### 6.2 Downgrade KnowledgeCeiling from Block → Redirect
**File**: `/src/decision/arbitration.py` (KnowledgeCeilingScorer)

**Current**: Applies 0.1x penalty (near-veto)

**Change**:
- Don't veto the strategy
- Instead, signal to question generator: "shift from explanation to reaction"
- Keep penalty at 0.5x (moderate), not 0.1x

**Implementation**: Add to scorer metadata
```python
if knowledge_ceiling_detected:
    return ScorerResult(
        multiplier=0.5,  # Not 0.1
        metadata={"redirect": "explanation_to_reaction"}
    )
```

---

## Implementation Sequence & Timeline

### Sprint 1: High-Impact Config Changes (Day 1, 4-5 hours)
1. ✅ Phase 1.1-1.5: interview_logic.yaml adjustments
2. ✅ Phase 2.1-2.2: Momentum prompt rewrite + history window
3. ✅ Test with same session_20251212_223603 stimulus
4. ✅ Verify: coverage resolves, momentum more balanced, depth increases

### Sprint 2: Schema Refinements (Day 2, 3-4 hours)
1. ✅ Phase 3.1-3.3: Schema guidance changes
2. ✅ Phase 4.1: Verify/fix coverage type-agnostic logic
3. ✅ Test: Check if values are now reachable
4. ✅ Verify: ladder depth improves, RTB coverage resolves

### Sprint 3: Optional Enhancements (Day 3-4, 6-8 hours)
1. ⭕ Phase 5.1: Laddering protocol (if needed after Sprint 2)
2. ⭕ Phase 5.2: Uncertainty strategy (if needed)
3. ⭕ Phase 6.1-6.2: Arbitration refinements (if multiplicative issues persist)

---

## Success Metrics

After implementation, re-run the same interview and verify:

### Primary Metrics (Must Fix)
- ✅ Coverage gaps resolve (not stuck at 1 for 8 turns)
- ✅ No premature fatigue on engaged responses
- ✅ At least 1 value node reached
- ✅ Depth strategies selected ≥40% of time (vs 25% currently)
- ✅ Fewer consecutive breadth turns (≤2 vs 4 currently)

### Secondary Metrics (Nice to Have)
- ⭕ Momentum neutral/high ratio improves (50%+ vs ~20% currently)
- ⭕ Graph depth reaches 3-4 levels (attribute → functional → psychosocial → value)
- ⭕ Reflection mode triggers naturally (not fatigue)
- ⭕ Interview feels conversational, not algorithmic

---

## Risk Assessment

### Low Risk (Configuration Only)
- ✅ Phase 1: interview_logic.yaml changes
- ✅ Phase 2: Momentum prompt changes
- ✅ Phase 3.1-3.3: Schema guidance changes

**Mitigation**: Easy rollback via git, no code changes

### Medium Risk (Logic Changes)
- ⚠️ Phase 3.4: Coverage type-agnostic logic
- ⚠️ Phase 4.1: Coverage gap detection modification

**Mitigation**: Test coverage logic in isolation, add unit tests

### High Risk (Architectural)
- ⚠️⚠️ Phase 6.1: Multiplicative scoring modification

**Mitigation**: Only attempt if Phases 1-4 don't resolve depth issues. Extensive testing required.

---

## Appendix: Key Files Reference

| File | Lines | Change Type | Priority |
|------|-------|-------------|----------|
| `interview_logic.yaml` | 204-259 | Config values | P0 |
| `extraction.py` | 551-577 | Prompt + history | P0 |
| `means_end_chain.yaml` | 14-103, 170 | Schema guidance | P0 |
| `state.py` | CoverageState | Logic fix | P1 |
| `arbitration.py` | 1027-1042 | Architecture (optional) | P2 |
| `strategy.py` | N/A | New strategy (optional) | P3 |

---

## Notes

**Is the feedback fair and correct?**
Yes, 100% accurate. All parameter values, line numbers, and interpretations verified.

**What files will be impacted?**
Primarily configuration/prompt files (75%), minimal code changes (25%).

**Why mostly prompt engineering?**
The architecture is sound. Issues stem from:
- Overly rigid configuration weights
- Poorly tuned prompts (momentum, schema)
- Not architectural flaws

**Estimated total effort**: 12-18 hours across 3 sprints
**Recommended approach**: Incremental (Sprints 1-2 mandatory, Sprint 3 conditional)
