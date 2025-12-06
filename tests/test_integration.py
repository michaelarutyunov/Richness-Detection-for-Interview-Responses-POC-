#!/usr/bin/env python3
"""
Integration Test Script

Tests the complete integration including extraction components.
This verifies that the system can handle real interview scenarios.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.models import GraphState, InterviewState, Node, Edge
from src.interview.core.configurable_orchestrator import ConfigurableGraphDrivenOrchestrator
from src.interview.tactics.loader import SchemaDrivenTacticLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_interview_scenario():
    """Test a complete interview scenario."""
    logger.info("Starting integration test...")
    
    try:
        # Initialize components
        # Create configurable orchestrator with proper dependencies
        from src.config.interview_config_loader import InterviewConfigLoader
        from src.interview.extraction.graph_extraction_orchestrator import GraphExtractionOrchestrator
        from src.interview.extraction import ExtractionPromptBuilder, ExtractionValidator, ResponseProcessor, ConceptExtractor
        
        # Create a mock extraction orchestrator for testing
        llm_client = None  # Will use template mode
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
        tactic_loader = SchemaDrivenTacticLoader()
        
        # Create initial interview state
        interview_state = InterviewState(
            session_id="integration_test_001",
            turn_number=0,
            phase="COVERAGE"
        )
        
        # Create initial graph state (empty)
        graph_state = GraphState(turn_number=0)
        
        # Load available tactics
        available_tactics = tactic_loader.load_tactics()
        logger.info(f"Loaded {len(available_tactics)} tactics")
        
        # Simulate first question generation
        logger.info("Generating first question...")
        first_question = await orchestrator.next_question(
            graph_state=graph_state,
            interview_state=interview_state,
            available_tactics=available_tactics
        )
        
        logger.info(f"First question: {first_question}")
        
        # Simulate participant response and graph update
        logger.info("Simulating participant response...")
        
        # Add some concepts to the graph (simulating extraction)
        test_nodes = [
            Node(id="concept1", label="Product Quality", type="attribute", creation_turn=1),
            Node(id="concept2", label="Customer Satisfaction", type="functional_consequence", creation_turn=1),
        ]
        
        for node in test_nodes:
            graph_state.add_node(node)
            
        test_edges = [
            Edge(id="edge1", type="leads_to", source="concept1", target="concept2", creation_turn=1),
        ]
        
        for edge in test_edges:
            graph_state.add_edge(edge)
        
        # Update interview state
        interview_state.increment_turn()
        interview_state.add_question(first_question)
        
        # Generate follow-up question
        logger.info("Generating follow-up question based on updated graph...")
        follow_up_question = await orchestrator.next_question(
            graph_state=graph_state,
            interview_state=interview_state,
            available_tactics=available_tactics
        )
        
        logger.info(f"Follow-up question: {follow_up_question}")
        
        # Verify graph state
        logger.info(f"Graph now has {graph_state.get_node_count()} nodes and {graph_state.get_edge_count()} edges")
        
        # Test needs detection on updated graph (using configurable detector)
        from src.interview.core.configurable_graph_needs_detector import ConfigurableGraphNeedsDetector
        from src.config.interview_config_loader import InterviewConfigLoader, InterviewConfig
        
        # Create default config for needs detection
        config_loader = InterviewConfigLoader()
        config = config_loader.load_config()
        needs_detector = ConfigurableGraphNeedsDetector(config)
        needs = needs_detector.detect_productive_needs(graph_state)
        logger.info(f"Detected needs: {[str(need) for need in needs]}")
        
        logger.info("✓ Integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}", exc_info=True)
        return False

async def main():
    """Main integration test execution."""
    logger.info("Starting Integration Test Suite")
    
    success = await test_interview_scenario()
    
    if success:
        logger.info("🎉 Integration test passed! System is working correctly.")
        return 0
    else:
        logger.error("❌ Integration test failed.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)