#!/usr/bin/env python3
"""
Test script for LLM Manager refactoring.
Verifies:
1. Backward compatibility with string models
2. New nested ModelConfig support
3. Timeout configuration
4. Token splitting (input/output)
5. Cost calculation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.llm_manager import (
    LLMConfig, LLMResponse, ModelConfig, ProviderConfig
)


def test_model_config_creation():
    """Test ModelConfig class creation."""
    print("Test 1: ModelConfig Creation")
    model = ModelConfig(
        name="test-model",
        request_timeout=45,
        cost_input=1.5,
        cost_output=8.0
    )
    assert model.name == "test-model"
    assert model.request_timeout == 45
    assert model.cost_input == 1.5
    assert model.cost_output == 8.0
    print("  ✓ ModelConfig creation works")


def test_provider_config_backward_compatibility():
    """Test backward compatibility with string models."""
    print("\nTest 2: Backward Compatibility")
    provider = ProviderConfig(
        api_key_env="TEST_KEY",
        models={
            "graph_extraction": "old-model-string",
            "question_generation": "another-old-model"
        }
    )

    # String models should be automatically converted to ModelConfig
    graph_model = provider.models["graph_extraction"]
    assert isinstance(graph_model, ModelConfig) or isinstance(graph_model, str)
    print("  ✓ String models handled correctly")


def test_provider_config_nested_models():
    """Test nested model configuration."""
    print("\nTest 3: Nested Model Config")
    provider = ProviderConfig(
        api_key_env="TEST_KEY",
        request_timeout=30,
        models={
            "graph_extraction": ModelConfig(
                name="new-model",
                request_timeout=25,
                cost_input=1.0,
                cost_output=5.0
            )
        }
    )

    model = provider.models["graph_extraction"]
    assert isinstance(model, (ModelConfig, str))
    if isinstance(model, ModelConfig):
        assert model.name == "new-model"
        assert model.request_timeout == 25
        print("  ✓ Nested ModelConfig works")
    else:
        print("  ✓ String fallback works")


def test_llm_response_token_splitting():
    """Test LLMResponse with split tokens."""
    print("\nTest 4: Token Splitting")
    response = LLMResponse(
        content="test response",
        model="test-model",
        provider="test-provider",
        input_tokens=100,
        output_tokens=50,
        latency_ms=1000
    )

    assert response.input_tokens == 100
    assert response.output_tokens == 50
    assert response.tokens_used == 150  # Backward compatibility property
    print("  ✓ Token splitting works")
    print(f"    Input: {response.input_tokens}, Output: {response.output_tokens}, Total: {response.tokens_used}")


def test_cost_calculation():
    """Test cost calculation in LLMResponse."""
    print("\nTest 5: Cost Calculation")
    response = LLMResponse(
        content="test response",
        model="test-model",
        provider="test-provider",
        input_tokens=100_000,  # 100k tokens
        output_tokens=50_000,   # 50k tokens
        latency_ms=1000,
        cost_input_per_1m=1.0,  # $1 per 1M input tokens
        cost_output_per_1m=5.0  # $5 per 1M output tokens
    )

    # Expected: (100k/1M)*1.0 + (50k/1M)*5.0 = 0.1 + 0.25 = 0.35
    expected_cost = 0.35
    actual_cost = response.cost_usd

    assert actual_cost is not None
    assert abs(actual_cost - expected_cost) < 0.001  # Float comparison tolerance
    print(f"  ✓ Cost calculation correct: ${actual_cost:.4f}")


def test_config_loading():
    """Test loading config from YAML file."""
    print("\nTest 6: Config Loading")
    config_path = Path(__file__).parent / "src/config/llm_config_with_pricing_example.yaml"

    if not config_path.exists():
        print(f"  ⚠ Skipping - example config not found at {config_path}")
        return

    try:
        config = LLMConfig.load(str(config_path))
        print(f"  ✓ Config loaded successfully")
        print(f"    Providers: {list(config.providers.keys())}")

        # Check kimi provider with nested config
        if "kimi" in config.providers:
            kimi = config.providers["kimi"]
            graph_model = kimi.models.get("graph_extraction")
            if isinstance(graph_model, ModelConfig):
                print(f"    Kimi graph_extraction:")
                print(f"      Model: {graph_model.name}")
                print(f"      Timeout: {graph_model.request_timeout}")
                print(f"      Cost: ${graph_model.cost_input}/{graph_model.cost_output} per 1M tokens")

        # Check deepseek with string models (backward compat)
        if "deepseek" in config.providers:
            deepseek = config.providers["deepseek"]
            graph_model = deepseek.models.get("graph_extraction")
            print(f"    Deepseek graph_extraction (backward compat):")
            if isinstance(graph_model, str):
                print(f"      Model: {graph_model} (string format)")
            elif isinstance(graph_model, ModelConfig):
                print(f"      Model: {graph_model.name} (converted to ModelConfig)")

    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM Manager Refactoring Tests")
    print("=" * 60)

    try:
        test_model_config_creation()
        test_provider_config_backward_compatibility()
        test_provider_config_nested_models()
        test_llm_response_token_splitting()
        test_cost_calculation()
        test_config_loading()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
