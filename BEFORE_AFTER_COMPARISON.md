# Before/After Comparison - LLM Manager Refactoring

## Configuration Structure

### BEFORE (String-based)
```yaml
providers:
  kimi:
    api_key_env: "KIMI_API_KEY"
    base_url: "https://api.moonshot.ai/v1"
    request_timeout: 30          # ⚠️ NOT USED - hardcoded to 30s
    models:
      graph_extraction: "kimi-k2-turbo-preview"        # Simple string
      question_generation: "kimi-k2-turbo-preview"     # Simple string
```

**Problems**:
- ❌ No cost tracking
- ❌ `request_timeout` ignored (hardcoded to 30s)
- ❌ No per-model timeout configuration
- ❌ Combined token count (can't calculate accurate costs)

### AFTER (Nested with Pricing)
```yaml
providers:
  kimi:
    api_key_env: "KIMI_API_KEY"
    base_url: "https://api.moonshot.ai/v1"
    request_timeout: 30          # ✅ NOW USED as default
    models:
      graph_extraction:
        name: "kimi-k2-turbo-preview"
        request_timeout: 25      # ✅ Model-specific override
        cost_input: 1.15         # ✅ Input token pricing
        cost_output: 8.00        # ✅ Output token pricing
      question_generation:
        name: "kimi-k2-turbo-preview"
        request_timeout: 30      # ✅ Use provider default
        cost_input: 1.15
        cost_output: 8.00
```

**Benefits**:
- ✅ Automatic cost calculation
- ✅ Dynamic timeout configuration
- ✅ Per-model timeout overrides
- ✅ Separate input/output token tracking
- ✅ Still supports old format (backward compatible)

---

## Code Structure

### BEFORE

#### LLMResponse
```python
class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    tokens_used: int          # ⚠️ Combined tokens - can't calculate accurate cost
    latency_ms: int
    success: bool
    error: Optional[str]
    function_call: Optional[Dict[str, Any]]
```

**Usage**:
```python
response = llm_manager.complete(...)
print(f"Tokens: {response.tokens_used}")  # Only total
# ❌ No way to calculate cost
# ❌ No way to see input vs output split
```

#### API Call (hardcoded timeout)
```python
def _call_openai_compatible(self, client, provider_name, model,
                           system_prompt, user_prompt,
                           temperature, max_tokens):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=30,              # ⚠️ HARDCODED - config ignored!
        messages=[...]
    )

    # ⚠️ Combined tokens
    tokens = response.usage.total_tokens if response.usage else 0
    return {"content": content, "tokens": tokens}
```

### AFTER

#### Enhanced LLMResponse
```python
class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str
    input_tokens: int         # ✅ Separate input tokens
    output_tokens: int        # ✅ Separate output tokens
    latency_ms: int
    success: bool
    error: Optional[str]
    function_call: Optional[Dict[str, Any]]
    cost_input_per_1m: Optional[float]   # ✅ Input pricing
    cost_output_per_1m: Optional[float]  # ✅ Output pricing

    @property
    def tokens_used(self) -> int:
        """✅ Backward compatibility."""
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> Optional[float]:
        """✅ Automatic cost calculation."""
        if self.cost_input_per_1m and self.cost_output_per_1m:
            input_cost = (self.input_tokens / 1_000_000) * self.cost_input_per_1m
            output_cost = (self.output_tokens / 1_000_000) * self.cost_output_per_1m
            return input_cost + output_cost
        return None
```

**Usage**:
```python
response = llm_manager.complete(...)

# ✅ Backward compatible
print(f"Total tokens: {response.tokens_used}")

# ✅ New detailed tracking
print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")

# ✅ Automatic cost calculation
if response.cost_usd:
    print(f"Cost: ${response.cost_usd:.4f}")
```

#### Dynamic Timeout API Call
```python
def _call_openai_compatible(self, client, provider_name, model,
                           system_prompt, user_prompt,
                           temperature, max_tokens, timeout):  # ✅ Dynamic
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,         # ✅ DYNAMIC - from config!
        messages=[...]
    )

    # ✅ Split tokens
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    return {"content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens}
```

---

## Usage Examples

### BEFORE - Cost Tracking
```python
# ❌ Manual tracking required
total_tokens = 0
for item in dataset:
    response = llm_manager.complete(...)
    total_tokens += response.tokens_used

# ❌ Can't calculate cost accurately:
# - Don't know input vs output split
# - Different pricing for input/output
# - Manual calculation error-prone
estimated_cost = total_tokens * 0.000005  # Rough estimate
```

### AFTER - Automatic Cost Tracking
```python
# ✅ Automatic, accurate cost tracking
total_cost = 0.0
for item in dataset:
    response = llm_manager.complete(...)

    if response.cost_usd:
        total_cost += response.cost_usd
        print(f"Request cost: ${response.cost_usd:.4f}")
        print(f"  Input: {response.input_tokens} tokens")
        print(f"  Output: {response.output_tokens} tokens")

print(f"Total session cost: ${total_cost:.2f}")
```

---

## Timeout Configuration

### BEFORE
```python
# In config
request_timeout: 45  # ⚠️ IGNORED!

# In code
response = client.chat.completions.create(
    ...
    timeout=30,  # ⚠️ Always 30 seconds, regardless of config
)
```

**Problem**: No way to configure timeouts differently per model/task

### AFTER
```python
# In config
providers:
  anthropic:
    request_timeout: 45  # ✅ Default for provider
    models:
      graph_extraction:
        name: "claude-haiku"
        request_timeout: 20  # ✅ Fast extraction
      question_generation:
        name: "claude-sonnet"
        request_timeout: 60  # ✅ Longer for generation

# In code (automatic resolution)
if isinstance(model_config, str):
    timeout = provider_config.request_timeout  # ✅ Provider default
else:
    timeout = model_config.request_timeout or provider_config.request_timeout  # ✅ Override or default

response = client.chat.completions.create(
    ...
    timeout=timeout,  # ✅ Dynamic from config
)
```

---

## Token Extraction

### BEFORE - Combined Tokens
```python
# Anthropic
tokens = response.usage.input_tokens + response.usage.output_tokens  # ⚠️ Combined

# OpenAI
tokens = response.usage.total_tokens  # ⚠️ Combined

return {"content": content, "tokens": tokens}
```

**Problem**: Can't distinguish input from output tokens

### AFTER - Split Tokens
```python
# Anthropic
input_tokens = response.usage.input_tokens   # ✅ Separate
output_tokens = response.usage.output_tokens # ✅ Separate

# OpenAI
input_tokens = response.usage.prompt_tokens      # ✅ Separate
output_tokens = response.usage.completion_tokens # ✅ Separate

return {
    "content": content,
    "input_tokens": input_tokens,    # ✅ Split
    "output_tokens": output_tokens   # ✅ Split
}
```

**Benefit**: Accurate cost calculation (output tokens cost 3-5x more)

---

## Migration Path

### Existing Configs (No Changes Needed) ✅
```yaml
# This still works exactly as before!
providers:
  deepseek:
    api_key_env: "DEEPSEEK_API_KEY"
    models:
      graph_extraction: "deepseek-chat"  # ✅ String format still supported
```

### Gradual Migration ✅
```yaml
# Mix old and new formats
providers:
  openai:
    models:
      graph_extraction:
        name: "gpt-4o-mini"
        cost_input: 0.15      # ✅ Add pricing when ready
        cost_output: 0.60
      question_generation: "gpt-4o"  # ✅ Keep string for now
```

### Full Feature Adoption ✅
```yaml
# Use all new features
providers:
  anthropic:
    request_timeout: 45
    models:
      graph_extraction:
        name: "claude-haiku-4-5-20251001"
        request_timeout: 30    # ✅ Per-model timeout
        cost_input: 0.80       # ✅ Accurate pricing
        cost_output: 4.00
```

---

## Summary of Changes

| Feature | Before | After |
|---------|--------|-------|
| Model Config | String only | String OR nested object |
| Token Tracking | Combined total | Split input/output |
| Cost Calculation | Manual | Automatic |
| Timeout Config | Hardcoded 30s | Dynamic from config |
| Per-Model Timeout | ❌ Not possible | ✅ Supported |
| Pricing Info | ❌ Not tracked | ✅ Co-located with model |
| Backward Compat | N/A | ✅ Fully maintained |

---

## Real-World Impact

### Cost Transparency Example
```python
# Using Anthropic Claude Sonnet
# Input: 100k tokens ($3 per 1M)  = $0.30
# Output: 20k tokens ($15 per 1M) = $0.30
# Total: $0.60

response = llm_manager.complete(...)
print(f"Cost: ${response.cost_usd:.2f}")  # Shows: $0.60

# WITHOUT this refactoring:
# - Would only see: "120k tokens"
# - Can't calculate accurate cost
# - Would guess: 120k * $3 = $0.36 (WRONG by 67%!)
```

### Timeout Optimization Example
```yaml
providers:
  anthropic:
    request_timeout: 60  # Conservative default
    models:
      graph_extraction:
        name: "claude-haiku"
        request_timeout: 15  # ✅ Fast model, fast timeout
      question_generation:
        name: "claude-opus"
        request_timeout: 90  # ✅ Powerful model, longer timeout
```

**Before**: All requests timeout at 30s (hardcoded)
**After**: Optimized per model (faster responses, better UX)

---

## Verification

All changes verified with:
```bash
python3 verify_refactoring.py
```

✅ All checks passed:
- ModelConfig class structure
- Union type support
- Token splitting
- Timeout activation
- Cost calculation
- Backward compatibility
