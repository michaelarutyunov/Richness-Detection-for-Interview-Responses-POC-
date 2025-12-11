# LLM Manager Refactoring - Nested Model Configuration with Integrated Pricing

## Overview

The LLM Manager has been refactored to support nested model configuration with integrated pricing information. This allows for better cost tracking, per-model timeout configuration, and maintains full backward compatibility with existing string-based model configurations.

## Key Changes

### 1. New `ModelConfig` Class

A new `ModelConfig` class has been added to represent detailed model configuration:

```python
class ModelConfig(BaseModel):
    """Configuration for a specific model assignment."""
    name: str                           # Model name (e.g., "gpt-4o-mini")
    request_timeout: Optional[int] = None    # Override provider timeout
    cost_input: Optional[float] = None       # USD per 1M input tokens
    cost_output: Optional[float] = None      # USD per 1M output tokens
```

### 2. Updated `ProviderConfig.models` Field

The `models` field now accepts both string and `ModelConfig` objects:

```python
# Before:
models: Dict[str, str] = Field(default_factory=dict)

# After:
models: Dict[str, Union[str, ModelConfig]] = Field(default_factory=dict)
```

**Backward Compatibility**: String models are automatically converted to `ModelConfig` objects during validation.

### 3. Enhanced `LLMResponse` with Token Splitting and Cost Calculation

The `LLMResponse` class has been enhanced to track input and output tokens separately and calculate costs:

```python
class LLMResponse(BaseModel):
    # ... existing fields ...
    input_tokens: int           # NEW: Input tokens used
    output_tokens: int          # NEW: Output tokens used
    cost_input_per_1m: Optional[float] = None   # NEW: Input cost per 1M tokens
    cost_output_per_1m: Optional[float] = None  # NEW: Output cost per 1M tokens

    @property
    def tokens_used(self) -> int:
        """Backward compatibility: returns total tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> Optional[float]:
        """Calculate cost if pricing available."""
        if self.cost_input_per_1m and self.cost_output_per_1m:
            input_cost = (self.input_tokens / 1_000_000) * self.cost_input_per_1m
            output_cost = (self.output_tokens / 1_000_000) * self.cost_output_per_1m
            return input_cost + output_cost
        return None
```

### 4. Dynamic Timeout Resolution

Timeouts are now properly resolved from configuration with model-level overrides:

```python
# Resolution priority:
# 1. Model-specific timeout (if ModelConfig used)
# 2. Provider-level timeout (fallback)

if isinstance(model_config, str):
    timeout = provider_config.request_timeout
else:
    timeout = model_config.request_timeout or provider_config.request_timeout
```

**IMPORTANT**: The hardcoded `timeout=30` has been removed. All API calls now use the configured timeout values.

### 5. Updated Token Extraction

All provider API calls now extract and return input/output tokens separately:

- **Anthropic**: `response.usage.input_tokens` and `response.usage.output_tokens`
- **OpenAI/Compatible**: `response.usage.prompt_tokens` and `response.usage.completion_tokens`

## Configuration Examples

### Old Format (Still Supported)

```yaml
providers:
  deepseek:
    api_key_env: "DEEPSEEK_API_KEY"
    base_url: "https://api.deepseek.com/"
    request_timeout: 30
    models:
      graph_extraction: "deepseek-chat"        # Simple string
      question_generation: "deepseek-chat"     # Simple string
```

### New Format (With Pricing)

```yaml
providers:
  kimi:
    api_key_env: "KIMI_API_KEY"
    base_url: "https://api.moonshot.ai/v1"
    request_timeout: 30
    models:
      graph_extraction:
        name: "kimi-k2-turbo-preview"
        request_timeout: 25              # Override: faster for extraction
        cost_input: 1.15                 # USD per 1M input tokens
        cost_output: 8.00                # USD per 1M output tokens
      question_generation:
        name: "kimi-k2-turbo-preview"
        request_timeout: 30              # Use default
        cost_input: 1.15
        cost_output: 8.00

  anthropic:
    api_key_env: "ANTHROPIC_API_KEY"
    request_timeout: 45
    models:
      graph_extraction:
        name: "claude-haiku-4-5-20251001"
        request_timeout: 40
        cost_input: 0.80                 # Haiku pricing
        cost_output: 4.00
      question_generation:
        name: "claude-sonnet-4-5"
        request_timeout: 60              # Longer timeout for generation
        cost_input: 3.00                 # Sonnet pricing
        cost_output: 15.00
```

### Mixed Format (Backward Compatible)

```yaml
providers:
  openai:
    api_key_env: "OPENAI_API_KEY"
    request_timeout: 30
    models:
      graph_extraction:
        name: "gpt-4o-mini"
        cost_input: 0.15                 # With pricing
        cost_output: 0.60
      question_generation: "gpt-4o"      # Old string format still works!
```

## Usage Examples

### Accessing Cost Information

```python
# After making an LLM call
response = llm_manager.complete(
    task=TaskType.GRAPH_EXTRACTION,
    system_prompt="...",
    user_prompt="..."
)

# Access token information
print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")
print(f"Total tokens: {response.tokens_used}")  # Backward compatible

# Calculate cost (if pricing configured)
if response.cost_usd:
    print(f"Cost: ${response.cost_usd:.4f}")
else:
    print("Cost information not available")
```

### Session Cost Tracking

```python
total_cost = 0.0
responses = []

for item in dataset:
    response = llm_manager.complete(...)
    responses.append(response)

    if response.cost_usd:
        total_cost += response.cost_usd

print(f"Total session cost: ${total_cost:.2f}")
```

## Benefits

1. **Cost Transparency**: Track exact costs per request when pricing is configured
2. **Flexible Timeouts**: Configure different timeouts per model/task
3. **Better Token Tracking**: Separate input/output tokens for detailed analysis
4. **Backward Compatible**: Existing configs work without changes
5. **Opt-in Pricing**: Pricing info is optional - not required to use the system

## Migration Guide

### For Existing Users

**No action required!** Your existing configurations will continue to work exactly as before.

### To Enable Pricing Tracking

1. Update your `llm_config.yaml` to use nested model configurations
2. Add `cost_input` and `cost_output` fields with pricing per 1M tokens
3. Access `response.cost_usd` to get calculated costs

### To Customize Timeouts

Add `request_timeout` to specific models to override provider defaults:

```yaml
models:
  graph_extraction:
    name: "claude-haiku-4-5-20251001"
    request_timeout: 20  # Override: faster timeout for this model
```

## Files Modified

- `/home/mikhailarutyunov/projects/Richness-Detection-for-Interview-Responses-POC-/src/utils/llm_manager.py`
  - Added `ModelConfig` class (line 77-82)
  - Updated `ProviderConfig` (line 85-103)
  - Enhanced `LLMResponse` (line 40-66)
  - Updated all API call methods to handle timeouts and split tokens
  - Updated model resolution logic in `complete()` method (line 259-273)

## Example Configuration File

See: `/home/mikhailarutyunov/projects/Richness-Detection-for-Interview-Responses-POC-/src/config/llm_config_with_pricing_example.yaml`

This file demonstrates:
- Nested model configurations with pricing (kimi, anthropic, openai)
- Per-model timeout overrides
- Backward compatible string models (deepseek)
- All provider types (Anthropic, OpenAI, OpenAI-compatible)

## Verification

Run the verification script to confirm all changes:

```bash
python3 verify_refactoring.py
```

This checks:
- ModelConfig class structure
- ProviderConfig.models type
- LLMResponse token splitting
- Dynamic timeout usage
- Token extraction from all providers

## Breaking Changes

**None.** All changes are backward compatible. Existing code and configurations will work without modification.

## Implementation Notes

### Critical Fix: request_timeout Activation

The `request_timeout` configuration field existed in the previous implementation but was **not used**. All API calls used a hardcoded `timeout=30`.

This refactoring **activates** the timeout configuration:
- Removes hardcoded `timeout=30` from all API calls
- Passes resolved timeout from configuration
- Supports model-level overrides via `ModelConfig.request_timeout`

### Token Split Rationale

Input and output tokens often have different pricing (output tokens typically cost 3-5x more). Tracking them separately enables:
- Accurate cost calculation
- Better understanding of prompt vs response token usage
- Optimization opportunities (e.g., reduce output tokens if they dominate cost)

### Optional Pricing

Pricing information is **optional**. The system works without it:
- `cost_usd` property returns `None` if pricing not configured
- No errors if `cost_input` or `cost_output` are missing
- Allows gradual adoption of cost tracking
