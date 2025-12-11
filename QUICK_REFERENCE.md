# LLM Manager Refactoring - Quick Reference

## 🚀 Quick Start

### New Configuration Format
```yaml
providers:
  provider_name:
    api_key_env: "API_KEY"
    request_timeout: 30        # Provider default timeout
    models:
      graph_extraction:
        name: "model-name"
        request_timeout: 25    # Optional: override timeout
        cost_input: 1.15       # Optional: USD per 1M input tokens
        cost_output: 8.00      # Optional: USD per 1M output tokens
```

### Backward Compatible (Old Format Still Works)
```yaml
providers:
  provider_name:
    models:
      graph_extraction: "model-name"  # ✅ String format still supported
```

---

## 📊 Using Cost Tracking

### Basic Usage
```python
response = llm_manager.complete(
    task=TaskType.GRAPH_EXTRACTION,
    system_prompt="...",
    user_prompt="..."
)

# Access cost info
if response.cost_usd:
    print(f"Cost: ${response.cost_usd:.4f}")
```

### Session Tracking
```python
total_cost = 0.0
for item in dataset:
    response = llm_manager.complete(...)
    if response.cost_usd:
        total_cost += response.cost_usd

print(f"Total: ${total_cost:.2f}")
```

---

## 🔧 Token Information

### New Fields
```python
response.input_tokens    # Input tokens used
response.output_tokens   # Output tokens used
response.tokens_used     # Total (backward compatible)
```

### Example
```python
print(f"Input: {response.input_tokens} tokens")
print(f"Output: {response.output_tokens} tokens")
print(f"Total: {response.tokens_used} tokens")
```

---

## ⏱️ Timeout Configuration

### Provider-Level Default
```yaml
providers:
  anthropic:
    request_timeout: 45  # Used if model doesn't override
```

### Model-Level Override
```yaml
providers:
  anthropic:
    request_timeout: 45
    models:
      graph_extraction:
        name: "claude-haiku"
        request_timeout: 20  # ✅ Faster timeout for extraction
```

---

## 💰 Adding Pricing to Existing Config

### Step 1: Keep existing config working
```yaml
# Your current config continues to work as-is
```

### Step 2: Add pricing to one model
```yaml
providers:
  your_provider:
    models:
      graph_extraction:
        name: "your-model"         # Change from string
        cost_input: 1.0            # Add pricing
        cost_output: 5.0
      question_generation: "old-model"  # Keep others as strings
```

### Step 3: Use cost tracking
```python
response = llm_manager.complete(...)
if response.cost_usd:
    print(f"Cost: ${response.cost_usd:.4f}")
```

---

## 🎯 Common Patterns

### Pattern 1: Cost Monitoring
```python
def track_costs(func):
    """Decorator to track LLM costs."""
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        if response.cost_usd:
            logger.info(f"LLM cost: ${response.cost_usd:.4f}")
        return response
    return wrapper
```

### Pattern 2: Budget Control
```python
MAX_BUDGET = 1.00  # $1 limit
session_cost = 0.0

for item in dataset:
    response = llm_manager.complete(...)

    if response.cost_usd:
        session_cost += response.cost_usd
        if session_cost > MAX_BUDGET:
            raise BudgetExceededError(f"Exceeded ${MAX_BUDGET}")
```

### Pattern 3: Cost Analysis
```python
costs = []
for item in dataset:
    response = llm_manager.complete(...)
    if response.cost_usd:
        costs.append({
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
            'cost': response.cost_usd
        })

# Analyze
avg_cost = sum(c['cost'] for c in costs) / len(costs)
print(f"Average cost per request: ${avg_cost:.4f}")
```

---

## 📝 Field Reference

### ModelConfig Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | str | ✅ Yes | Model identifier |
| `request_timeout` | int | ❌ No | Override provider timeout |
| `cost_input` | float | ❌ No | USD per 1M input tokens |
| `cost_output` | float | ❌ No | USD per 1M output tokens |

### LLMResponse Fields (New/Changed)
| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | int | Input tokens used |
| `output_tokens` | int | Output tokens used |
| `tokens_used` | int | Total (property, backward compat) |
| `cost_input_per_1m` | float | Input pricing (from config) |
| `cost_output_per_1m` | float | Output pricing (from config) |
| `cost_usd` | float | Calculated cost (property) |

---

## 🔍 Debugging

### Check Configuration
```python
# View loaded config
manager = LLMManager.from_config_file("llm_config.yaml")

# Check provider config
provider_config = manager.config.providers["kimi"]
model_config = provider_config.models["graph_extraction"]

if isinstance(model_config, ModelConfig):
    print(f"Model: {model_config.name}")
    print(f"Timeout: {model_config.request_timeout}")
    print(f"Pricing: ${model_config.cost_input}/${model_config.cost_output}")
else:
    print(f"Model: {model_config} (string format)")
```

### Enable Cost Logging
```python
import logging
logging.basicConfig(level=logging.INFO)

# Will log:
# [LLM Call] graph_extraction via kimi/kimi-k2-turbo-preview
# [LLM Result] 1234ms, 567 tokens (in:400, out:167), success=True
```

---

## 📚 Configuration Examples

### Example 1: Minimal (Backward Compatible)
```yaml
providers:
  deepseek:
    api_key_env: "DEEPSEEK_API_KEY"
    base_url: "https://api.deepseek.com/"
    models:
      graph_extraction: "deepseek-chat"
```

### Example 2: With Pricing
```yaml
providers:
  kimi:
    api_key_env: "KIMI_API_KEY"
    base_url: "https://api.moonshot.ai/v1"
    models:
      graph_extraction:
        name: "kimi-k2-turbo-preview"
        cost_input: 1.15
        cost_output: 8.00
```

### Example 3: Full Featured
```yaml
providers:
  anthropic:
    api_key_env: "ANTHROPIC_API_KEY"
    request_timeout: 45
    models:
      graph_extraction:
        name: "claude-haiku-4-5-20251001"
        request_timeout: 30      # Override
        cost_input: 0.80
        cost_output: 4.00
      question_generation:
        name: "claude-sonnet-4-5"
        request_timeout: 60      # Different timeout
        cost_input: 3.00
        cost_output: 15.00
```

### Example 4: Mixed Format
```yaml
providers:
  openai:
    api_key_env: "OPENAI_API_KEY"
    models:
      graph_extraction:
        name: "gpt-4o-mini"
        cost_input: 0.15         # New format with pricing
        cost_output: 0.60
      question_generation: "gpt-4o"  # Old format (no pricing)
```

---

## ✅ Verification

### Check Your Installation
```bash
# Verify syntax
python3 -m py_compile src/utils/llm_manager.py

# Run verification script
python3 verify_refactoring.py
```

---

## 📖 More Information

- **Full Documentation**: See `LLM_MANAGER_REFACTORING.md`
- **Before/After Comparison**: See `BEFORE_AFTER_COMPARISON.md`
- **Complete Summary**: See `REFACTORING_COMPLETION_SUMMARY.md`
- **Example Config**: See `src/config/llm_config_with_pricing_example.yaml`

---

## 🆘 Common Issues

### Issue: Cost is None
**Cause**: Pricing not configured
**Solution**: Add `cost_input` and `cost_output` to model config

### Issue: Timeout still 30s
**Cause**: Using old string format
**Solution**: Convert to nested format with `request_timeout` field

### Issue: Configuration not loading
**Cause**: Syntax error in YAML
**Solution**: Check YAML indentation and field names

---

## 💡 Tips

1. **Start Simple**: Keep existing config, add pricing gradually
2. **Test First**: Try on one provider before updating all
3. **Monitor Costs**: Enable logging to see costs in real-time
4. **Use Properties**: `response.cost_usd` and `response.tokens_used` are convenient
5. **Backward Compatible**: Old configs work forever - no rush to migrate
