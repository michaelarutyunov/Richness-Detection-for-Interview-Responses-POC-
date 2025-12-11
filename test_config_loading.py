#!/usr/bin/env python3
"""Test script to validate LLM config loading with new structure."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Direct import to avoid __init__ issues
from src.utils.llm_manager import LLMManager

def test_config_loading():
    """Test loading the updated config with nested model configs."""
    config_path = project_root / 'src/config/llm_config.yaml'

    print("Testing LLM config loading...")
    manager = LLMManager.from_config_file(str(config_path))

    print('✓ Config loaded successfully')
    print(f'✓ Graph extraction model: {manager.config.graph_extraction_model}')
    print(f'✓ Question generation model: {manager.config.question_generation_model}')

    # Test ModelConfig handling
    for provider_name, provider_config in manager.config.providers.items():
        print(f'\n{provider_name}:')
        for task_name, model_config in provider_config.models.items():
            if hasattr(model_config, 'name'):
                print(f'  {task_name}:')
                print(f'    - model: {model_config.name}')
                print(f'    - timeout: {model_config.request_timeout}s')
                print(f'    - cost: ${model_config.cost_input:.2f}/${model_config.cost_output:.2f} per 1M tokens')
            else:
                print(f'  {task_name}: {model_config} (legacy string format)')

    print('\n✓ All providers validated')
    print('\n✓ Test passed!')

if __name__ == '__main__':
    test_config_loading()
