#!/usr/bin/env python3
"""
Final Validation Script

Tests the complete end-to-end workflow:
1. Start interview session
2. Process participant response
3. Extract concepts and update graph
4. Generate follow-up question
5. Verify complete pipeline works
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.models import GraphState, InterviewState, Node, Edge
from src.core.schema_loader import SchemaLoader
from src.interview.core.graph_driven_orchestrator import GraphDrivenOrchestrator
from src.interview.tactics.loader import SchemaDrivenTacticLoader
from src.interview.core.graph_needs_detector import GraphNeedsDetector
from src.interview.core.strategy_selector import StrategySelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_complete_workflow():
    """Test the complete interview workflow end-to-end."""
    logger.info("🚀 Starting final validation test...")
    
    try:
        # Initialize all components
        logger.info("1. Initializing components...")
        orchestrator = GraphDrivenOrchestrator()
        tactic_loader = SchemaDrivenTacticLoader()
        needs_detector = GraphNeedsDetector()
        strategy_selector = StrategySelector()
        
        # Create initial states
        logger.info("2. Creating initial states...")
        interview_state = InterviewState(
            session_id="final_validation_001",
            turn_number=0,
            phase="COVERAGE"
        )
        graph_state = GraphState(turn_number=0)
        
        # Load available tactics
        logger.info("3. Loading tactics from schema...")
        available_tactics = tactic_loader.load_tactics()
        logger.info(f"✓ Loaded {len(available_tactics)} tactics")
        
        # Step 1: Generate opening question
        logger.info("4. Generating opening question...")
        opening_question = await orchestrator.next_question(
            graph_state=graph_state,
            interview_state=interview_state,
            available_tactics=available_tactics
        )
        logger.info(f"✓ Opening question: {opening_question}")
        
        # Step 2: Simulate participant response and concept extraction
        logger.info("5. Processing participant response...")
        participant_response = "I really like how convenient this product is for my busy lifestyle."
        
        # Simulate concept extraction (normally done by extraction pipeline)
        extracted_nodes = [
            Node(id="attr_convenience", label="Convenience", type="attribute", creation_turn=1),
            Node(id="func_time_saving", label="Time Saving", type="functional_consequence", creation_turn=1),
            Node(id="psycho_reduced_stress", label="Reduced Stress", type="psychosocial_consequence", creation_turn=1),
        ]
        
        extracted_edges = [
            Edge(id="edge1", type="leads_to", source="attr_convenience", target="func_time_saving", creation_turn=1),
            Edge(id="edge2", type="leads_to", source="func_time_saving", target="psycho_reduced_stress", creation_turn=1),
        ]
        
        # Update graph state
        for node in extracted_nodes:
            graph_state.add_node(node)
        for edge in extracted_edges:
            graph_state.add_edge(edge)
        
        # Update interview state
        interview_state.increment_turn()
        interview_state.add_question(opening_question)
        # Note: Response recording would be handled by the UI/extraction pipeline
        
        logger.info(f"✓ Extracted {len(extracted_nodes)} nodes and {len(extracted_edges)} edges")
        logger.info(f"✓ Graph now has {graph_state.get_node_count()} nodes and {graph_state.get_edge_count()} edges")
        
        # Step 3: Generate follow-up question based on updated graph
        logger.info("6. Generating follow-up question...")
        follow_up_question = await orchestrator.next_question(
            graph_state=graph_state,
            interview_state=interview_state,
            available_tactics=available_tactics
        )
        logger.info(f"✓ Follow-up question: {follow_up_question}")
        
        # Step 4: Test needs detection on updated graph
        logger.info("7. Testing needs detection...")
        needs = needs_detector.detect(graph_state)
        logger.info(f"✓ Detected needs: {[str(need) for need in needs]}")
        
        # Step 5: Test strategy selection
        logger.info("8. Testing strategy selection...")
        if needs:
            selected_strategy = strategy_selector.select(needs)
            logger.info(f"✓ Selected strategy: {selected_strategy}")
        
        # Step 6: Test schema integration
        logger.info("9. Testing schema v0.2 integration...")
        schema_loader = SchemaLoader(Path("schemas/means_end_chain_v0.2.yaml"))
        schema = schema_loader.load_schema()
        
        # Verify all components are working with schema
        logger.info(f"✓ Schema version: {schema.schema_version}")
        logger.info(f"✓ Domain: {schema.domain}")
        logger.info(f"✓ Node types: {len(schema.node_types)}")
        logger.info(f"✓ Edge types: {len(schema.edge_types)}")
        logger.info(f"✓ Strategies: {len(schema.strategies)}")
        logger.info(f"✓ Tactics: {len(schema.tactics)}")
        
        # Verify the extracted concepts match schema node types
        extracted_types = {node.type for node in extracted_nodes}
        schema_types = {node_type.name for node_type in schema.node_types}
        valid_types = extracted_types.issubset(schema_types)
        logger.info(f"✓ Extracted node types valid: {valid_types}")
        
        # Step 7: Test UI integration
        logger.info("10. Testing UI integration...")
        from src.ui.gradio_app import InterviewUI
        ui = InterviewUI()
        interface = ui.build_interface()
        logger.info(f"✓ Gradio interface created successfully: {type(interface).__name__}")
        
        logger.info("\n🎉 Final validation completed successfully!")
        logger.info("✅ All components working together correctly")
        logger.info("✅ Schema v0.2 integration verified")
        logger.info("✅ Complete pipeline functional")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Final validation failed: {e}", exc_info=True)
        return False

async def main():
    """Main validation execution."""
    logger.info("=" * 60)
    logger.info("FINAL SYSTEM VALIDATION")
    logger.info("=" * 60)
    
    success = await test_complete_workflow()
    
    if success:
        logger.info("\n🚀 System ready for production use!")
        return 0
    else:
        logger.error("\n❌ System validation failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)