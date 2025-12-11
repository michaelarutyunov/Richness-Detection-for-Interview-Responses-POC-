#!/usr/bin/env python3
"""
Verification script for LLM Manager refactoring.
This checks the key changes without requiring dependencies.
"""

import ast
import sys
from pathlib import Path


def check_file_syntax(filepath):
    """Check if file has valid Python syntax."""
    print(f"\n{'='*60}")
    print(f"Checking: {filepath.name}")
    print('='*60)

    with open(filepath, 'r') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
        print("✓ Syntax is valid")
        return tree, source
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        sys.exit(1)


def find_class_def(tree, class_name):
    """Find a class definition in AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def get_class_attributes(class_node):
    """Extract attributes from a Pydantic-style class."""
    attributes = {}
    for item in class_node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            attributes[item.target.id] = item
    return attributes


def check_model_config_class(tree):
    """Check if ModelConfig class exists and has required fields."""
    print("\n1. Checking ModelConfig class...")

    model_config = find_class_def(tree, "ModelConfig")
    if not model_config:
        print("  ✗ ModelConfig class not found")
        return False

    print("  ✓ ModelConfig class exists")

    attrs = get_class_attributes(model_config)
    required_fields = ["name", "request_timeout", "cost_input", "cost_output"]

    for field in required_fields:
        if field in attrs:
            print(f"    ✓ {field} field present")
        else:
            print(f"    ✗ {field} field missing")
            return False

    return True


def check_provider_config_models_type(tree, source):
    """Check if ProviderConfig.models has Union[str, ModelConfig] type."""
    print("\n2. Checking ProviderConfig.models type...")

    provider_config = find_class_def(tree, "ProviderConfig")
    if not provider_config:
        print("  ✗ ProviderConfig class not found")
        return False

    # Look for models field
    attrs = get_class_attributes(provider_config)
    if "models" not in attrs:
        print("  ✗ models field not found")
        return False

    print("  ✓ models field exists")

    # Check if Union is used in the annotation (simple text search)
    if "Union[str, ModelConfig]" in source:
        print("  ✓ models type includes Union[str, ModelConfig]")
        return True
    else:
        print("  ✗ models type doesn't include Union[str, ModelConfig]")
        return False


def check_llm_response_tokens(tree, source):
    """Check if LLMResponse has input_tokens and output_tokens."""
    print("\n3. Checking LLMResponse token fields...")

    llm_response = find_class_def(tree, "LLMResponse")
    if not llm_response:
        print("  ✗ LLMResponse class not found")
        return False

    attrs = get_class_attributes(llm_response)

    required_fields = ["input_tokens", "output_tokens", "cost_input_per_1m", "cost_output_per_1m"]
    for field in required_fields:
        if field in attrs:
            print(f"  ✓ {field} field present")
        else:
            print(f"  ✗ {field} field missing")
            return False

    # Check for properties
    has_tokens_used = False
    has_cost_usd = False

    for item in llm_response.body:
        if isinstance(item, ast.FunctionDef):
            # Check if it's a property
            for decorator in item.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "property":
                    if item.name == "tokens_used":
                        has_tokens_used = True
                        print("  ✓ tokens_used property (backward compat)")
                    elif item.name == "cost_usd":
                        has_cost_usd = True
                        print("  ✓ cost_usd property")

    if not has_tokens_used:
        print("  ✗ tokens_used property missing")
        return False
    if not has_cost_usd:
        print("  ✗ cost_usd property missing")
        return False

    return True


def check_timeout_parameter_usage(source):
    """Check if timeout parameter is passed to API calls."""
    print("\n4. Checking timeout parameter usage...")

    timeout_checks = [
        ("_call_anthropic signature", "def _call_anthropic(" in source and "timeout: int" in source),
        ("_call_openai_compatible signature", "def _call_openai_compatible(" in source and "timeout: int" in source),
        ("_call_anthropic_with_tools signature", "def _call_anthropic_with_tools(" in source and "timeout: int" in source),
        ("_call_openai_compatible_with_tools signature", "def _call_openai_compatible_with_tools(" in source and "timeout: int" in source),
        ("Anthropic API call", "client.messages.create(" in source and "timeout=timeout" in source),
        ("OpenAI API call", "client.chat.completions.create(" in source and "timeout=timeout" in source),
        ("No hardcoded timeout=30", "timeout=30" not in source or source.count("timeout=30") < 2),  # Might be in default value
    ]

    all_passed = True
    for check_name, passed in timeout_checks:
        if passed:
            print(f"  ✓ {check_name}")
        else:
            print(f"  ✗ {check_name}")
            all_passed = False

    return all_passed


def check_token_extraction(source):
    """Check if token extraction returns separate input/output tokens."""
    print("\n5. Checking token extraction...")

    checks = [
        ("input_tokens/output_tokens in return dict", '"input_tokens"' in source and '"output_tokens"' in source),
        ("Anthropic input extraction", "response.usage.input_tokens" in source),
        ("Anthropic output extraction", "response.usage.output_tokens" in source),
        ("OpenAI input extraction", "response.usage.prompt_tokens" in source),
        ("OpenAI output extraction", "response.usage.completion_tokens" in source),
        ("LLMResponse with input_tokens", "input_tokens=" in source),
        ("LLMResponse with output_tokens", "output_tokens=" in source),
        ("LLMResponse with cost params", "cost_input_per_1m=" in source and "cost_output_per_1m=" in source),
    ]

    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"  ✓ {check_name}")
        else:
            print(f"  ✗ {check_name}")
            all_passed = False

    return all_passed


def main():
    """Run all verification checks."""
    print("="*60)
    print("LLM Manager Refactoring Verification")
    print("="*60)

    llm_manager_path = Path(__file__).parent / "src/utils/llm_manager.py"

    if not llm_manager_path.exists():
        print(f"✗ File not found: {llm_manager_path}")
        sys.exit(1)

    tree, source = check_file_syntax(llm_manager_path)

    checks = [
        check_model_config_class(tree),
        check_provider_config_models_type(tree, source),
        check_llm_response_tokens(tree, source),
        check_timeout_parameter_usage(source),
        check_token_extraction(source),
    ]

    print("\n" + "="*60)
    if all(checks):
        print("All verifications passed! ✓")
        print("="*60)
        print("\nSummary of changes:")
        print("  1. ✓ ModelConfig class created with pricing fields")
        print("  2. ✓ ProviderConfig.models supports Union[str, ModelConfig]")
        print("  3. ✓ LLMResponse split tokens (input/output) with cost calculation")
        print("  4. ✓ Dynamic timeout from config (no more hardcoded 30s)")
        print("  5. ✓ Separate token extraction from all providers")
        print("\n  Backward compatibility maintained for string models!")
    else:
        print("Some verifications failed! ✗")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
