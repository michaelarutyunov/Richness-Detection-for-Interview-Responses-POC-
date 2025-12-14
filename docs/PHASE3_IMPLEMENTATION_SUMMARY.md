# Phase 3: Prompt Construction Improvements - Implementation Summary

**Date**: 2025-12-12
**Plan**: `/home/mikhailarutyunov/.claude/plans/witty-crunching-bengio.md`
**Modified File**: `src/decision/extraction.py`
**Status**: ✅ COMPLETED

---

## Overview

Successfully implemented all 5 tasks from Phase 3 of the Extraction Module Improvement Plan. The changes focus on improving prompt quality by:

1. Prioritizing edge extraction
2. Strengthening ambiguity detection
3. De-emphasizing forced element mappings
4. Refining atomicity rules to prevent over-fragmentation
5. Making element_mapping optional in the schema

---

## Implementation Details

### Task 1: Add Edge Priority Section ✅

**Location**: Line 669-685 (before Extraction Rules)

**Added Section**:
```
## CRITICAL: Edge Extraction Priority

Edges are AS IMPORTANT as nodes. For every concept, ask: "Why?" or "What does this lead to?"

**Extract edges when respondent uses:**
- Causal language: "because", "leads to", "results in", "causes"
- Explanations: "X is important because it gives me Y"
- Connections: "X is related to Y", "X enables Y"
- Sequences: "first X happens, then Y"

**Minimum expectation:** 1 edge per 2 nodes
If you extract 4 nodes, extract at least 2 edges connecting them.

**Edge validation checklist:**
1. Both nodes exist or are being created in this extraction
2. Relationship is explicitly stated or strongly implied
3. Relation type follows schema adjacency rules
```

**Impact**:
- Elevates edge extraction to same priority as nodes
- Provides concrete linguistic markers for edge detection
- Sets quantitative expectation (1 edge per 2 nodes minimum)

---

### Task 2: Enhanced Ambiguity Detection ✅

**Location**: Line 709-719 (Rule 4)

**Before**:
```
4. **Mark ambiguous concepts**
   - If meaning is unclear, set is_ambiguous: true
   - These will be clarified before further use
```

**After**:
```
4. **Ambiguity Detection (Mark Liberally)**

Set **is_ambiguous: true** when:
1. **Type uncertainty**: "smooth" (texture attribute? ease-of-use functional?)
2. **Scope ambiguity**: "it works well" (what is "it"? what aspect "works"?)
3. **Implicit reference**: "that", "the other one" without clear antecedent
4. **Polysemy**: "security" (physical? data? financial?)
5. **Vague intensifiers**: "very good", "quite nice" without specifics

**Default when uncertain: FLAG IT**
It's better to flag 10 nodes and clarify 2 than miss 1 critical ambiguity.
```

**Impact**:
- Provides 5 concrete ambiguity types with examples
- Encourages liberal flagging over conservative acceptance
- Reduces risk of accepting unclear concepts

---

### Task 3: De-emphasize Element Mapping ✅

**Location**: Line 721-724 (Rule 5)

**Before**:
```
5. **IMPORTANT: Map to reference elements**
   - For EVERY node, you MUST provide an element_mapping value
   - If a node discusses, evaluates, or relates to a reference element, use that element's ID
   - Only use null if the node truly doesn't relate to any reference element
   - This mapping is essential for coverage tracking
```

**After**:
```
5. **Map to reference elements (when clear)**
   - Provide element_mapping ONLY if the node clearly discusses that element
   - When uncertain, use null (don't force mappings)
   - Concept extraction takes priority over element mapping
```

**Impact**:
- Removes pressure to force element mappings
- Prioritizes concept extraction quality
- Allows null when uncertain

---

### Task 4: Refined Atomicity Rule ✅

**Location**: Line 689-699 (Rule 1)

**Before**:
```
1. **Extract ATOMIC concepts (CRITICAL)**
   - Each node must represent ONE concept that could vary independently
   - Break compound concepts into separate nodes:
     * "thicker and creamier" → two nodes: "thicker", "creamier"
     * "fast and reliable" → two nodes: "fast", "reliable"
     * "smooth and sweet taste" → two nodes: "smooth taste", "sweet taste"
   - Test: If two attributes could have different causes or consequences, they MUST be separate nodes
   - Test: If you can imagine someone having one without the other, they're separate
```

**After**:
```
1. **Extract ATOMIC concepts (avoid over-fragmentation)**
   - Each node = ONE concept that could vary independently
   - Break compounds ONLY if respondent treats them separately:
     * "thick and creamy" → ONE node if treated as single quality
     * "thick and creamy" → TWO nodes if discussed separately

   Test: Would respondent have mentioned one without the other?

   Examples:
   - "smooth creamy texture" → ONE node (single gestalt quality)
   - "smooth texture, plus it's also creamy" → TWO nodes (distinct attributes)
```

**Impact**:
- Prevents over-fragmentation of gestalt concepts
- Provides clearer guidance on when to split vs. keep together
- Uses respondent's treatment as the key decision criterion

---

### Task 5: Update EXTRACTION_TOOL Schema ✅

**Location**: Line 37

**Before**:
```python
"required": ["label", "node_type", "quote", "element_mapping"],
```

**After**:
```python
"required": ["label", "node_type", "quote"],
```

**Impact**:
- element_mapping is now optional in function calling schema
- LLM will not be forced to provide mappings
- Aligns schema with de-emphasized prompt guidance

---

## Verification Results

### Code Changes Verified ✅

All 5 tasks successfully implemented:

1. ✅ Edge Priority Section added at line 669
2. ✅ Ambiguity Detection enhanced at line 709
3. ✅ Element Mapping de-emphasized at line 721
4. ✅ Atomicity Rule refined at line 689
5. ✅ EXTRACTION_TOOL schema updated at line 37

### Old Text Removal Verified ✅

Confirmed removal of problematic text:

- ❌ "thicker and creamier → two nodes" (NOT FOUND - removed)
- ❌ "For EVERY node, you MUST provide an element_mapping" (NOT FOUND - removed)
- ❌ Simple ambiguity text (NOT FOUND - removed)

### Prompt Structure Verified ✅

The `_build_extraction_system_prompt()` method now follows this improved structure:

1. Schema prompt (from YAML)
2. Reference elements section (if applicable)
3. Sentiment guidance (if applicable)
4. **NEW: Edge Extraction Priority** (Task 1)
5. Extraction Rules:
   - Rule 1: Refined atomicity (Task 4)
   - Rule 2: Quote support
   - Rule 3: Use existing nodes
   - Rule 4: Enhanced ambiguity detection (Task 2)
   - Rule 5: De-emphasized element mapping (Task 3)
6. Reaction Detection
7. Function call instruction

---

## Expected Benefits

### 1. Improved Edge Extraction
- **Target**: Edge-to-node ratio >0.5 (from ~0.3)
- **Mechanism**: Explicit priority, linguistic markers, quantitative expectation

### 2. Better Ambiguity Detection
- **Target**: Ambiguity flagging rate 5-10% (from <2%)
- **Mechanism**: 5 concrete types, liberal flagging encouraged

### 3. Reduced Forced Mappings
- **Target**: More null element_mappings when appropriate
- **Mechanism**: Optional schema field, de-emphasized prompt guidance

### 4. Less Fragmentation
- **Target**: Fewer over-fragmented compound concepts
- **Mechanism**: Gestalt quality recognition, respondent-treatment test

### 5. Cleaner Prompt
- **Target**: More focused, structured prompt
- **Mechanism**: Prioritized sections, better organization

---

## Testing Recommendations

### 1. Integration Testing

Run extraction pipeline with sample responses to verify:

```bash
# If you have integration tests
pytest tests/test_extraction_pipeline_v2.py -v

# If you have sample responses
python scripts/test_extraction.py --sample-responses data/test_responses.json
```

### 2. Metrics to Monitor

Track these metrics before and after:

- **Edge-to-node ratio**: Should increase from ~0.3 to >0.5
- **Ambiguity flagging rate**: Should increase from <2% to 5-10%
- **Element mapping null rate**: Should increase (more selective mappings)
- **Average nodes per response**: Should decrease slightly (less fragmentation)
- **Extraction quality**: Manual review of sample extractions

### 3. A/B Testing

Consider running parallel extractions:
- Control: Old prompt
- Treatment: New prompt (Phase 3)

Compare quality metrics and select the better performing version.

---

## Next Steps

1. **Test the Changes**
   - Run integration tests if available
   - Extract from sample responses
   - Compare with previous extractions

2. **Monitor Metrics**
   - Log edge-to-node ratios
   - Track ambiguity flagging rates
   - Monitor element mapping null rates

3. **Iterate Based on Results**
   - Adjust edge minimum expectation if needed (currently 1 per 2 nodes)
   - Tune ambiguity detection thresholds
   - Refine atomicity examples based on observed patterns

4. **Proceed to Phase 4 (Optional)**
   - Implement schema validation feedback loop
   - Add post-extraction quality checks
   - Log warnings for low edge density, isolated nodes, etc.

---

## Files Modified

- `src/decision/extraction.py` (lines 37, 669-724)

---

## Related Work

This Phase 3 implementation builds on:

- **Phase 1**: Schema enhancements (already completed in `means_end_chain.yaml` and `jobs_to_be_done.yaml`)
- **Phase 2**: Semantic deduplication (already implemented with Jaccard + embeddings)
- **Phase 4**: Enhanced schema validation (optional, not yet implemented)

All phases work together to improve extraction quality and reduce fragmentation.

---

## Success Criteria

✅ All 5 tasks implemented
✅ Old problematic text removed
✅ Prompt structure verified
⏳ Integration testing pending
⏳ Metrics collection pending

**Overall Status**: Implementation complete, ready for testing and deployment.
