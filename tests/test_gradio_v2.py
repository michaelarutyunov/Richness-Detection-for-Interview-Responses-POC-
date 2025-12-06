#!/usr/bin/env python3
"""
Test script for Gradio UI v2 with LLM integration.
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui.gradio_app import launch_app


def test_basic_functionality():
    """Test basic Gradio functionality."""
    print("🧪 Testing Gradio UI v2 Basic Functionality")
    
    # Check environment
    print("📋 Environment Check:")
    print(f"   Python: {sys.version}")
    print(f"   Gradio: {getattr(__import__('gradio'), '__version__', 'unknown')}")
    
    # Check API keys
    providers = {
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "OpenAI": os.getenv("OPENAI_API_KEY"), 
        "Kimi": os.getenv("KIMI_API_KEY")
    }
    
    print("🔑 API Keys:")
    for provider, key in providers.items():
        status = "✅ Configured" if key else "❌ Not set"
        print(f"   {provider}: {status}")
    
    return any(providers.values())


def test_gradio_imports():
    """Test that all Gradio imports work."""
    print("\n📦 Testing Gradio Imports...")
    
    try:
        import gradio as gr
        print("   ✅ Gradio imported successfully")
        
        # Test basic components
        test_components = [
            gr.Textbox,
            gr.Button, 
            gr.Chatbot,
            gr.JSON,
            gr.Dataframe,
            gr.Plot,
            gr.File,
            gr.Markdown,
            gr.Row,
            gr.Column,
            gr.Tabs,
            gr.TabItem,
            gr.Accordion
        ]
        
        for component in test_components:
            try:
                component()  # Try to instantiate
                print(f"   ✅ {component.__name__} works")
            except Exception as e:
                print(f"   ❌ {component.__name__} failed: {e}")
                
        return True
        
    except ImportError as e:
        print(f"   ❌ Gradio import failed: {e}")
        return False


def test_v2_components():
    """Test that v2 components can be imported."""
    print("\n🔧 Testing v2 Component Imports...")
    
    try:
        from src.core.models import GraphState, InterviewState
        print("   ✅ Core models imported")
        
        from src.interview.core import ConfigurableGraphDrivenOrchestrator
        print("   ✅ ConfigurableGraphDrivenOrchestrator imported")
        
        from src.interview.tactics import ConfigurableQuestionGenerator
        print("   ✅ ConfigurableQuestionGenerator imported")
        
        from src.interview.tactics import SchemaDrivenTacticLoader as TacticLoader
        print("   ✅ TacticLoader imported")
        
        from src.llm.factory import LLMClientFactory
        print("   ✅ LLMClientFactory imported")
        
        from src.ui.gradio_app import InterviewUI
        print("   ✅ InterviewUI imported")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ v2 component import failed: {e}")
        return False


def test_orchestrator_creation():
    """Test that orchestrator can be created."""
    print("\n🎯 Testing Orchestrator Creation...")
    
    try:
        from src.interview.core import ConfigurableGraphDrivenOrchestrator
        from src.interview.tactics import ConfigurableQuestionGenerator
        from src.config.interview_config_loader import InterviewConfigLoader
        from src.interview.extraction.graph_extraction_orchestrator import GraphExtractionOrchestrator
        from src.interview.extraction import ExtractionPromptBuilder, ExtractionValidator, ResponseProcessor, ConceptExtractor
        
        # Create a mock extraction orchestrator for testing
        llm_client = None
        schema_path = "schemas/means_end_chain_v0.2.yaml"
        prompt_builder = ExtractionPromptBuilder(schema_path)
        validator = ExtractionValidator(schema_path)
        response_processor = ResponseProcessor(llm_client, prompt_builder, validator)
        concept_extractor = ConceptExtractor(llm_client, prompt_builder, validator)
        
        extraction_orchestrator = GraphExtractionOrchestrator(
            response_processor=response_processor,
            concept_extractor=concept_extractor
        )
        
        # Create configuration loader
        config_loader = InterviewConfigLoader()
        
        orchestrator = ConfigurableGraphDrivenOrchestrator(
            extraction_orchestrator=extraction_orchestrator,
            config_loader=config_loader
        )
        
        print("   ✅ Orchestrator created successfully (template mode)")
        print(f"   📋 Components: {type(orchestrator).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Orchestrator creation failed: {e}")
        return False


def test_gradio_ui_creation():
    """Test that Gradio UI can be created."""
    print("\n🎨 Testing Gradio UI Creation...")
    
    try:
        from src.ui.gradio_app import InterviewUI
        
        ui = InterviewUI()
        print("   ✅ InterviewUI created successfully")
        
        # Test basic methods
        assert hasattr(ui, 'setup_llm_client')
        assert hasattr(ui, 'start_interview_with_concept')
        assert hasattr(ui, 'process_response')
        assert hasattr(ui, 'build_interface')
        
        print("   ✅ All required methods present")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Gradio UI creation failed: {e}")
        return False


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality."""
    print("\n⚡ Testing Async Functionality...")
    
    try:
        from src.interview.core import ConfigurableGraphDrivenOrchestrator
        from src.interview.tactics import ConfigurableQuestionGenerator
        from src.core.models import GraphState, InterviewState
        from src.config.interview_config_loader import InterviewConfigLoader
        from src.interview.extraction.graph_extraction_orchestrator import GraphExtractionOrchestrator
        from src.interview.extraction import ExtractionPromptBuilder, ExtractionValidator, ResponseProcessor, ConceptExtractor
        
        # Create a mock extraction orchestrator for testing
        llm_client = None
        schema_path = "schemas/means_end_chain_v0.2.yaml"
        prompt_builder = ExtractionPromptBuilder(schema_path)
        validator = ExtractionValidator(schema_path)
        response_processor = ResponseProcessor(llm_client, prompt_builder, validator)
        concept_extractor = ConceptExtractor(llm_client, prompt_builder, validator)
        
        extraction_orchestrator = GraphExtractionOrchestrator(
            response_processor=response_processor,
            concept_extractor=concept_extractor
        )
        
        # Create configuration loader
        config_loader = InterviewConfigLoader()
        
        orchestrator = ConfigurableGraphDrivenOrchestrator(
            extraction_orchestrator=extraction_orchestrator,
            config_loader=config_loader
        )
        
        # Create test state
        graph_state = GraphState()
        interview_state = InterviewState(session_id="test_session")
        
        # Test async question generation
        question = await orchestrator.next_question(
            graph_state=graph_state,
            interview_state=interview_state,
            available_tactics=[]
        )
        
        print(f"   ✅ Async question generation works: {question}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Async functionality failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test of the system."""
    print("🚀 AI Interview System v2 - Comprehensive Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test 2: Gradio imports
    results.append(("Gradio Imports", test_gradio_imports()))
    
    # Test 3: v2 components
    results.append(("v2 Components", test_v2_components()))
    
    # Test 4: Orchestrator creation
    results.append(("Orchestrator Creation", test_orchestrator_creation()))
    
    # Test 5: Gradio UI creation
    results.append(("Gradio UI Creation", test_gradio_ui_creation()))
    
    # Test 6: Async functionality
    results.append(("Async Functionality", asyncio.run(test_async_functionality())))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System is ready for Gradio UI.")
        print("\n💡 Next steps:")
        print("   1. Set up API keys: export ANTHROPIC_API_KEY='your-key'")
        print("   2. Run: python app_v2_with_llm.py")
        print("   3. Open browser to http://localhost:7860")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please fix issues before proceeding.")
        return False


def main():
    """Main test function."""
    success = run_comprehensive_test()
    
    if success:
        print("\n🎯 Ready to launch Gradio UI!")
        print("\nTo launch the interface:")
        print("   python app_v2_with_llm.py")
        print("\nFor interactive mode:")
        print("   python app_v2_with_llm.py --interactive")
    else:
        print("\n🔧 Please fix the failed tests before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()