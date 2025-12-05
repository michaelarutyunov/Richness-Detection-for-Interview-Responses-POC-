#!/usr/bin/env python3
"""
Core Functionality Test Script

Tests the critical path after reorganization:
GraphDrivenOrchestrator → GraphNeedsDetector → StrategySelector → 
SchemaDrivenTacticSelector → QuestionGenerator

Also tests schema v0.2 integration and component communication.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all core components
from src.core.models import GraphState, InterviewState, Node, Edge, NeedName, StrategyName
from src.core.schema_loader import SchemaLoader
from src.interview.core.graph_driven_orchestrator import GraphDrivenOrchestrator
from src.interview.core.graph_needs_detector import GraphNeedsDetector
from src.interview.core.strategy_selector import StrategySelector
from src.interview.tactics.selector import SchemaDrivenTacticSelector
from src.interview.tactics.loader import SchemaDrivenTacticLoader
from src.interview.tactics.question_generator import QuestionGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoreFunctionalityTester:
    """Test suite for core graph-driven interview functionality."""
    
    def __init__(self):
        self.test_results = {}
        self.components = {}
        
    async def setup_components(self) -> bool:
        """Initialize all components needed for testing."""
        logger.info("Setting up test components...")
        
        try:
            # Initialize core components
            self.components['orchestrator'] = GraphDrivenOrchestrator()
            self.components['needs_detector'] = GraphNeedsDetector()
            self.components['strategy_selector'] = StrategySelector()
            self.components['tactic_selector'] = SchemaDrivenTacticSelector()
            self.components['question_generator'] = QuestionGenerator()
            self.components['schema_loader'] = SchemaLoader()
            self.components['tactic_loader'] = SchemaDrivenTacticLoader()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            return False
    
    def create_test_graph_state(self) -> GraphState:
        """Create a test graph state with sample data."""
        graph = GraphState(turn_number=5)
        
        # Add some test nodes
        nodes = [
            Node(id="node1", label="Product Quality", type="attribute", creation_turn=1),
            Node(id="node2", label="Customer Satisfaction", type="functional_consequence", creation_turn=2),
            Node(id="node3", label="Brand Loyalty", type="psychosocial_consequence", creation_turn=3),
            Node(id="node4", label="Personal Values", type="value", creation_turn=4),
            Node(id="node5", label="Price Sensitivity", type="attribute", creation_turn=5, visit_count=3),
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        # Add some test edges
        edges = [
            Edge(id="edge1", type="leads_to", source="node1", target="node2", creation_turn=2),
            Edge(id="edge2", type="leads_to", source="node2", target="node3", creation_turn=3),
            Edge(id="edge3", type="leads_to", source="node3", target="node4", creation_turn=4),
        ]
        
        for edge in edges:
            graph.add_edge(edge)
            
        return graph
    
    def create_test_interview_state(self) -> InterviewState:
        """Create a test interview state."""
        return InterviewState(
            session_id="test_session_123",
            turn_number=5,
            question_history=[
                "What do you think about the product quality?",
                "How does that affect your satisfaction?",
                "Can you tell me more about that relationship?"
            ],
            tactic_usage={
                "emotional_contrast": 2,
                "relationship_dynamics": 1
            }
        )
    
    async def test_imports(self) -> bool:
        """Test that all components can be imported."""
        logger.info("Testing imports...")
        
        try:
            # Test basic imports
            from src.core.models import GraphState, InterviewState
            from src.interview.core.graph_driven_orchestrator import GraphDrivenOrchestrator
            from src.interview.core.graph_needs_detector import GraphNeedsDetector
            from src.interview.core.strategy_selector import StrategySelector
            from src.interview.tactics.selector import SchemaDrivenTacticSelector
            from src.interview.tactics.question_generator import QuestionGenerator
            
            logger.info("✓ All imports successful")
            return True
            
        except ImportError as e:
            logger.error(f"✗ Import failed: {e}")
            return False
    
    async def test_orchestrator_instantiation(self) -> bool:
        """Test that GraphDrivenOrchestrator can be instantiated."""
        logger.info("Testing GraphDrivenOrchestrator instantiation...")
        
        try:
            orchestrator = self.components['orchestrator']
            
            # Test basic properties
            assert orchestrator.needs_detector is not None
            assert orchestrator.strategy_selector is not None
            assert orchestrator.tactic_selector is not None
            assert orchestrator.question_generator is not None
            
            # Test validation
            assert orchestrator.validate_components() is True
            
            logger.info("✓ GraphDrivenOrchestrator instantiated and validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Orchestrator instantiation failed: {e}")
            return False
    
    async def test_needs_detection(self) -> bool:
        """Test GraphNeedsDetector functionality."""
        logger.info("Testing GraphNeedsDetector...")
        
        try:
            detector = self.components['needs_detector']
            graph_state = self.create_test_graph_state()
            
            # Test needs detection
            needs = detector.detect(graph_state)
            
            assert isinstance(needs, list)
            assert len(needs) > 0
            
            # Check that needs have valid properties
            for need in needs:
                assert hasattr(need, 'name')
                assert hasattr(need, 'score')
                assert hasattr(need, 'context')
                assert 0 <= need.score <= 1.0
                assert need.name in NeedName
            
            logger.info(f"✓ Detected {len(needs)} needs: {[str(need) for need in needs]}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Needs detection failed: {e}")
            return False
    
    async def test_strategy_selection(self) -> bool:
        """Test StrategySelector functionality."""
        logger.info("Testing StrategySelector...")
        
        try:
            selector = self.components['strategy_selector']
            graph_state = self.create_test_graph_state()
            detector = self.components['needs_detector']
            
            # Get needs first
            needs = detector.detect(graph_state)
            assert len(needs) > 0
            
            # Test strategy selection
            strategy = selector.select(needs)
            
            assert strategy in StrategyName
            
            # Test strategy description
            description = selector.get_strategy_description(strategy)
            assert isinstance(description, str)
            assert len(description) > 0
            
            logger.info(f"✓ Selected strategy: {strategy.value} - {description}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Strategy selection failed: {e}")
            return False
    
    async def test_tactic_loading(self) -> bool:
        """Test SchemaDrivenTacticLoader functionality."""
        logger.info("Testing SchemaDrivenTacticLoader...")
        
        try:
            loader = self.components['tactic_loader']
            
            # Test tactic loading
            tactics = loader.load_tactics()
            
            assert isinstance(tactics, list)
            assert len(tactics) > 0
            
            # Check that tactics have required properties
            for tactic in tactics:
                assert hasattr(tactic, 'id')
                assert hasattr(tactic, 'name')
                assert hasattr(tactic, 'templates')
                assert hasattr(tactic, 'min_turn')
                assert hasattr(tactic, 'max_visit_count')
            
            logger.info(f"✓ Loaded {len(tactics)} tactics from schema")
            return True
            
        except Exception as e:
            logger.error(f"✗ Tactic loading failed: {e}")
            return False
    
    async def test_tactic_selection(self) -> bool:
        """Test SchemaDrivenTacticSelector functionality."""
        logger.info("Testing SchemaDrivenTacticSelector...")
        
        try:
            selector = self.components['tactic_selector']
            loader = self.components['tactic_loader']
            
            # Load tactics
            tactics = loader.load_tactics()
            assert len(tactics) > 0
            
            # Test with a specific strategy
            strategy = StrategyName.DEPTH_COMPLETION
            interview_state = self.create_test_interview_state()
            
            # Test tactic selection
            selected_tactic = selector.select(strategy, interview_state, tactics)
            
            assert selected_tactic is not None
            assert hasattr(selected_tactic, 'id')
            assert hasattr(selected_tactic, 'templates')
            
            logger.info(f"✓ Selected tactic: {selected_tactic.id} for strategy {strategy.value}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Tactic selection failed: {e}")
            return False
    
    async def test_question_generation(self) -> bool:
        """Test QuestionGenerator functionality."""
        logger.info("Testing QuestionGenerator...")
        
        try:
            generator = self.components['question_generator']
            
            # Create test context
            graph_state = self.create_test_graph_state()
            interview_state = self.create_test_interview_state()
            
            # Get a tactic for testing
            loader = self.components['tactic_loader']
            tactics = loader.load_tactics()
            tactic = tactics[0] if tactics else None
            
            if not tactic:
                logger.error("No tactics available for question generation test")
                return False
            
            # Test question generation (using fallback templates)
            question = await generator.generate_question(tactic, graph_state, interview_state)
            
            assert isinstance(question, str)
            assert len(question) > 0
            assert question.endswith('?')
            
            logger.info(f"✓ Generated question: {question}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Question generation failed: {e}")
            return False
    
    async def test_full_pipeline(self) -> bool:
        """Test the complete pipeline: orchestrator → needs → strategy → tactic → question."""
        logger.info("Testing complete pipeline...")
        
        try:
            orchestrator = self.components['orchestrator']
            
            # Create test context
            graph_state = self.create_test_graph_state()
            interview_state = self.create_test_interview_state()
            
            # Load available tactics
            loader = self.components['tactic_loader']
            available_tactics = loader.load_tactics()
            
            # Test the complete pipeline
            question = await orchestrator.next_question(
                graph_state=graph_state,
                interview_state=interview_state,
                available_tactics=available_tactics
            )
            
            assert isinstance(question, str)
            assert len(question) > 0
            assert question.endswith('?')
            
            logger.info(f"✓ Complete pipeline generated question: {question}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Complete pipeline test failed: {e}")
            return False
    
    async def test_schema_integration(self) -> bool:
        """Test that schema v0.2 integration is working."""
        logger.info("Testing schema v0.2 integration...")
        
        try:
            schema_loader = self.components['schema_loader']
            
            # Test loading schema (it's already loaded during initialization)
            schema_config = schema_loader.load_schema()
            assert schema_config is not None
            
            # Test schema components
            assert len(schema_config.node_types) > 0
            assert len(schema_config.edge_types) > 0
            assert len(schema_config.strategies) > 0
            assert len(schema_config.tactics) > 0
            
            # Test getting specific components
            node_types = schema_loader.get_node_types()
            assert len(node_types) > 0
            
            edge_types = schema_loader.get_edge_types()
            assert len(edge_types) > 0
            
            strategies = schema_loader.get_all_strategies()
            assert len(strategies) > 0
            
            tactics = schema_loader.get_all_tactics()
            assert len(tactics) > 0
            
            logger.info(f"✓ Schema v0.2 integration working - loaded schema with {len(node_types)} node types, {len(edge_types)} edge types, {len(strategies)} strategies, and {len(tactics)} tactics")
            return True
            
        except Exception as e:
            logger.error(f"✗ Schema integration test failed: {e}")
            return False
    
    async def test_component_communication(self) -> bool:
        """Test that components can communicate with each other properly."""
        logger.info("Testing component communication...")
        
        try:
            # Test orchestrator state
            orchestrator = self.components['orchestrator']
            state = orchestrator.get_orchestrator_state()
            
            assert isinstance(state, dict)
            assert 'needs_detector' in state
            assert 'strategy_selector' in state
            assert 'tactic_selector' in state
            
            # Test that components are properly initialized
            assert state['components_initialized'] is True
            
            logger.info("✓ Component communication working properly")
            return True
            
        except Exception as e:
            logger.error(f"✗ Component communication test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("Starting core functionality tests...")
        
        # Setup components first
        if not await self.setup_components():
            logger.error("Failed to setup components, aborting tests")
            return {}
        
        # Define test sequence
        tests = [
            ("Imports", self.test_imports),
            ("Orchestrator Instantiation", self.test_orchestrator_instantiation),
            ("Needs Detection", self.test_needs_detection),
            ("Strategy Selection", self.test_strategy_selection),
            ("Tactic Loading", self.test_tactic_loading),
            ("Tactic Selection", self.test_tactic_selection),
            ("Question Generation", self.test_question_generation),
            ("Schema Integration", self.test_schema_integration),
            ("Component Communication", self.test_component_communication),
            ("Full Pipeline", self.test_full_pipeline),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running: {test_name} ---")
            try:
                result = await test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✓ {test_name}: PASSED")
                else:
                    logger.error(f"✗ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"✗ {test_name}: FAILED with exception: {e}")
                results[test_name] = False
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Print test summary."""
        logger.info("\n" + "="*60)
        logger.info("CORE FUNCTIONALITY TEST SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name:<30} {status}")
        
        logger.info("-" * 60)
        logger.info(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            logger.info("🎉 All tests passed! Core functionality is working correctly.")
        else:
            logger.warning(f"⚠️  {total - passed} tests failed. Review the logs above.")
        
        return passed == total

async def main():
    """Main test execution."""
    logger.info("Starting Core Functionality Test Suite")
    logger.info("Testing critical path: GraphDrivenOrchestrator → GraphNeedsDetector → StrategySelector → SchemaDrivenTacticSelector → QuestionGenerator")
    
    tester = CoreFunctionalityTester()
    results = await tester.run_all_tests()
    
    if results:
        success = tester.print_summary(results)
        return 0 if success else 1
    else:
        logger.error("No test results available - setup failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)