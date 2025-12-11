# LLM Manager Architecture - After Refactoring

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM Manager                              │
│  Unified interface for multiple LLM providers                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LLMConfig                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  graph_extraction_model: "kimi"                         │   │
│  │  question_generation_model: "kimi"                      │   │
│  │  extraction_specs: {...}                                │   │
│  │  providers: {...}  ◄─────────────────────────┐         │   │
│  │  retry_config: {...}                         │         │   │
│  └──────────────────────────────────────────────┼─────────┘   │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │
                              ┌─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────────────┐
        │           ProviderConfig                           │
        │  ┌────────────────────────────────────────────┐   │
        │  │  api_key_env: "KIMI_API_KEY"              │   │
        │  │  base_url: "https://api.moonshot.ai/v1"   │   │
        │  │  request_timeout: 30                       │   │
        │  │  models: {                                 │   │
        │  │    "graph_extraction": ModelConfig,       │   │
        │  │    "question_generation": ModelConfig     │   │
        │  │  }                                         │   │
        │  └────────────────────────────────────────────┘   │
        └───────────────────────────────────────────────────┘
                              │
                              │ contains
                              ▼
        ┌───────────────────────────────────────────────────┐
        │           ModelConfig (NEW)                        │
        │  ┌────────────────────────────────────────────┐   │
        │  │  name: "kimi-k2-turbo-preview"            │   │
        │  │  request_timeout: 25 (optional override)   │   │
        │  │  cost_input: 1.15 (USD per 1M tokens)     │   │
        │  │  cost_output: 8.00 (USD per 1M tokens)    │   │
        │  └────────────────────────────────────────────┘   │
        └───────────────────────────────────────────────────┘
```

---

## Request Flow

```
1. Application
      │
      │ llm_manager.complete(task, system_prompt, user_prompt)
      │
      ▼
2. LLMManager.complete()
      │
      │ Resolve provider, spec, model config
      │
      ├─► Get provider name (e.g., "kimi")
      ├─► Get extraction spec (temperature, max_tokens)
      ├─► Get model config:
      │       - If string: use as-is, default timeout
      │       - If ModelConfig: extract name, timeout, pricing
      │
      ▼
3. LLMManager._execute_with_retry()
      │
      │ Retry loop with exponential backoff
      │
      ├─► Determine API call method based on provider
      │
      ▼
4. Provider-specific API call
      │
      ├─► _call_anthropic(timeout=25, ...)
      │       └─► Extract: input_tokens, output_tokens
      │
      ├─► _call_openai_compatible(timeout=25, ...)
      │       └─► Extract: prompt_tokens, completion_tokens
      │
      └─► _call_*_with_tools(timeout=25, ...)
              └─► Extract tokens + function call
      │
      ▼
5. Build LLMResponse
      │
      │ LLMResponse(
      │   content=...,
      │   input_tokens=400,
      │   output_tokens=167,
      │   cost_input_per_1m=1.15,
      │   cost_output_per_1m=8.00
      │ )
      │
      ├─► Calculate: tokens_used = 567 (property)
      ├─► Calculate: cost_usd = $0.0014 (property)
      │
      ▼
6. Return to Application
      │
      └─► Access: response.cost_usd, response.tokens_used, etc.
```

---

## Configuration Resolution

```
┌─────────────────────────────────────────────────────────────┐
│  Task: GRAPH_EXTRACTION                                     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Get provider name                                  │
│  config.graph_extraction_model → "kimi"                     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Get extraction spec                                │
│  config.extraction_specs["graph_extraction"]                │
│    → temperature: 0.3, max_tokens: 1000                     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Get model config                                   │
│  provider_config = config.providers["kimi"]                 │
│  model_config = provider_config.models["graph_extraction"]  │
│                                                              │
│  if isinstance(model_config, str):                          │
│    model = model_config                                     │
│    timeout = provider_config.request_timeout                │
│    cost_input = None                                        │
│    cost_output = None                                       │
│  else:  # ModelConfig object                                │
│    model = model_config.name                                │
│    timeout = model_config.request_timeout OR                │
│              provider_config.request_timeout                │
│    cost_input = model_config.cost_input                     │
│    cost_output = model_config.cost_output                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  Result:                                                     │
│    model = "kimi-k2-turbo-preview"                          │
│    timeout = 25                                             │
│    cost_input = 1.15                                        │
│    cost_output = 8.00                                       │
│    temperature = 0.3                                        │
│    max_tokens = 1000                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Token and Cost Calculation

```
API Response
    │
    ├─► Anthropic Format
    │       response.usage.input_tokens = 400
    │       response.usage.output_tokens = 167
    │
    └─► OpenAI Format
            response.usage.prompt_tokens = 400
            response.usage.completion_tokens = 167
    │
    ▼
Extract Tokens
    input_tokens = 400
    output_tokens = 167
    │
    ▼
Build LLMResponse
    LLMResponse(
        input_tokens=400,
        output_tokens=167,
        cost_input_per_1m=1.15,
        cost_output_per_1m=8.00
    )
    │
    ▼
Property Calculations
    │
    ├─► tokens_used (property)
    │       return input_tokens + output_tokens
    │       = 400 + 167
    │       = 567 tokens
    │
    └─► cost_usd (property)
            if cost_input_per_1m and cost_output_per_1m:
                input_cost = (400 / 1_000_000) × 1.15
                           = 0.00046
                output_cost = (167 / 1_000_000) × 8.00
                            = 0.001336
                total = 0.00046 + 0.001336
                      = 0.001796
                return $0.0018 (rounded)
            return None
```

---

## Class Relationships

```
LLMConfig
    │
    ├─► extraction_specs: Dict[str, ExtractionSpec]
    │       └─► ExtractionSpec
    │               ├─► temperature: float
    │               ├─► max_tokens: int
    │               └─► timeout_seconds: int
    │
    ├─► providers: Dict[str, ProviderConfig]
    │       └─► ProviderConfig
    │               ├─► api_key_env: str
    │               ├─► base_url: Optional[str]
    │               ├─► request_timeout: int
    │               ├─► organization: Optional[str]
    │               └─► models: Dict[str, Union[str, ModelConfig]]
    │                       ├─► "graph_extraction": ModelConfig
    │                       │       ├─► name: str
    │                       │       ├─► request_timeout: Optional[int]
    │                       │       ├─► cost_input: Optional[float]
    │                       │       └─► cost_output: Optional[float]
    │                       │
    │                       └─► "question_generation": str OR ModelConfig
    │
    └─► retry_config: RetryConfig
            ├─► max_retries: int
            ├─► initial_delay_seconds: float
            ├─► backoff_multiplier: float
            └─► max_delay_seconds: float
```

---

## LLMResponse Structure

```
LLMResponse
    │
    ├─► Core Fields
    │       ├─► content: str
    │       ├─► model: str
    │       ├─► provider: str
    │       ├─► success: bool
    │       ├─► error: Optional[str]
    │       └─► function_call: Optional[Dict]
    │
    ├─► Token Fields (NEW)
    │       ├─► input_tokens: int
    │       └─► output_tokens: int
    │
    ├─► Pricing Fields (NEW)
    │       ├─► cost_input_per_1m: Optional[float]
    │       └─► cost_output_per_1m: Optional[float]
    │
    ├─► Timing Fields
    │       └─► latency_ms: int
    │
    ├─► Properties (Computed)
    │       ├─► tokens_used: int
    │       │       = input_tokens + output_tokens
    │       │       (Backward compatibility)
    │       │
    │       └─► cost_usd: Optional[float]
    │               = (input_tokens / 1M) × cost_input_per_1m +
    │                 (output_tokens / 1M) × cost_output_per_1m
    │
    └─► Usage Examples
            response.tokens_used       → 567
            response.input_tokens      → 400
            response.output_tokens     → 167
            response.cost_usd          → 0.0018
            response.latency_ms        → 1234
```

---

## Provider Support Matrix

```
┌──────────────┬──────────┬────────────┬──────────┬──────────┐
│ Provider     │ Timeout  │ Pricing    │ Tools    │ Status   │
├──────────────┼──────────┼────────────┼──────────┼──────────┤
│ Anthropic    │ ✅ Yes   │ ✅ Yes     │ ✅ Yes   │ ✅ Full  │
│ OpenAI       │ ✅ Yes   │ ✅ Yes     │ ✅ Yes   │ ✅ Full  │
│ Kimi         │ ✅ Yes   │ ✅ Yes     │ ✅ Yes   │ ✅ Full  │
│ DeepSeek     │ ✅ Yes   │ ✅ Yes     │ ✅ Yes   │ ✅ Full  │
└──────────────┴──────────┴────────────┴──────────┴──────────┘

Legend:
  ✅ Yes = Fully supported
  Timeout = Dynamic timeout from config
  Pricing = Cost tracking with cost_input/cost_output
  Tools = Function calling support
  Status = Overall implementation status
```

---

## Configuration Backward Compatibility

```
OLD FORMAT (Still Supported)
┌─────────────────────────────────────┐
│ models:                             │
│   graph_extraction: "model-name"    │ ◄── String
│   question_generation: "model"      │ ◄── String
└─────────────────────────────────────┘
          │
          │ Automatic conversion
          ▼
┌─────────────────────────────────────┐
│ models:                             │
│   graph_extraction:                 │
│     ModelConfig(                    │
│       name="model-name",            │
│       request_timeout=None,         │
│       cost_input=None,              │
│       cost_output=None              │
│     )                               │
└─────────────────────────────────────┘

NEW FORMAT (Enhanced)
┌─────────────────────────────────────┐
│ models:                             │
│   graph_extraction:                 │
│     name: "model-name"              │
│     request_timeout: 25             │ ◄── Override
│     cost_input: 1.15                │ ◄── Pricing
│     cost_output: 8.00               │ ◄── Pricing
└─────────────────────────────────────┘
          │
          │ Direct use
          ▼
┌─────────────────────────────────────┐
│ ModelConfig(                        │
│   name="model-name",                │
│   request_timeout=25,               │
│   cost_input=1.15,                  │
│   cost_output=8.00                  │
│ )                                   │
└─────────────────────────────────────┘
```

---

## Timeline Comparison

```
BEFORE REFACTORING
─────────────────────────────────────────────────────────────
Request → API Call (timeout=30 HARDCODED) → Response
                                               │
                                               ├─► content
                                               ├─► tokens_used (combined)
                                               └─► ❌ NO COST INFO

AFTER REFACTORING
─────────────────────────────────────────────────────────────
Request → Resolve Config → API Call (timeout=DYNAMIC) → Response
              │                                           │
              ├─► timeout from config                    ├─► content
              ├─► pricing from config                    ├─► input_tokens
              └─► model name                             ├─► output_tokens
                                                          ├─► cost_usd (calc)
                                                          └─► ✅ FULL TRACKING
```

---

## Summary

This refactoring adds:

1. **ModelConfig Class**: Nested configuration with pricing
2. **Token Splitting**: Separate input/output tokens
3. **Cost Calculation**: Automatic USD cost calculation
4. **Dynamic Timeout**: Activated from configuration
5. **Backward Compatibility**: Old configs still work

All changes are non-breaking and fully backward compatible!
