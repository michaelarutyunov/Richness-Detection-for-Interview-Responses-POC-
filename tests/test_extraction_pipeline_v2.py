#!/usr/bin/env python3
"""
Test script for the complete extraction pipeline with schema v0.2.
Tests all components working together: SchemaLoader, ExtractionPromptBuilder, 
ExtractionValidator, and ResponseProcessor.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import the extraction pipeline components
from src.core.schema_loader import SchemaLoader
from src.interview.extraction import ExtractionPromptBuilder, ExtractionValidator, ResponseProcessor
from src.llm.client import BaseLLMClient, LLMResponse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without real API calls."""
    
    def __init__(self):
        """Initialize mock client."""
        super().__init__(model="mock-model", temperature=0.0)
        self.provider = "mock"
        self.call_count = 0
        
    async def generate_completion(self, messages: List[Dict[str, str]], 
                                system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate mock completion."""
        self.call_count += 1
        
        # Extract the participant response from messages
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        
        # Simple mock response based on the input
        if "packaging" in user_message.lower():
            content = "I understand you're talking about packaging. Let me extract the relevant concepts."
        else:
            content = "I can help analyze that response."
            
        return LLMResponse(
            content=content,
            provider=self.provider,
            model=self.model,
            model_used=self.model,
            latency_ms=100,
            tokens_used=50
        )
    
    async def generate_completion_with_function_call(
        self,
        messages: List[Dict[str, str]],
        function_schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate mock function call response."""
        self.call_count += 1
        
        # Extract the participant response from messages
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        
        # Mock extraction based on the sample response
        if "packaging" in user_message.lower() and "convenient" in user_message.lower():
            # Expected extraction for the test response
            mock_extraction = {
                "name": "extract_graph_delta",
                "arguments": {
                    "nodes_added": [
                        {
                            "type": "attribute",
                            "label": "packaging",
                            "quote": "I really like how convenient the packaging is"
                        },
                        {
                            "type": "attribute", 
                            "label": "resealable_feature",
                            "quote": "The resealable feature keeps everything fresh"
                        },
                        {
                            "type": "functional_consequence",
                            "label": "saves_time",
                            "quote": "It saves me time in the morning when I'm getting ready for work"
                        },
                        {
                            "type": "functional_consequence",
                            "label": "keeps_fresh",
                            "quote": "keeps everything fresh"
                        }
                    ],
                    "edges_added": [
                        {
                            "type": "leads_to",
                            "source": "packaging",
                            "target": "saves_time",
                            "quote": "convenient the packaging is. It saves me time",
                            "confidence": 0.9
                        },
                        {
                            "type": "leads_to",
                            "source": "resealable_feature", 
                            "target": "keeps_fresh",
                            "quote": "resealable feature keeps everything fresh",
                            "confidence": 1.0
                        }
                    ]
                }
            }
        else:
            # Default empty extraction
            mock_extraction = {
                "name": "extract_graph_delta",
                "arguments": {
                    "nodes_added": [],
                    "edges_added": []
                }
            }
        
        return LLMResponse(
            content="Extraction complete",
            provider=self.provider,
            model=self.model,
            model_used=self.model,
            latency_ms=150,
            tokens_used=75,
            function_call=mock_extraction
        )
    
    def validate_config(self) -> bool:
        """Mock validation always passes."""
        return True


async def test_schema_loader():
    """Test SchemaLoader with v0.2 schema."""
    logger.info("=== Testing SchemaLoader ===")
    
    schema_loader = SchemaLoader()
    schema_config = schema_loader.load_schema()
    
    logger.info(f"Schema version: {schema_config.schema_version}")
    logger.info(f"Domain: {schema_config.domain}")
    logger.info(f"Node types: {len(schema_config.node_types)}")
    logger.info(f"Edge types: {len(schema_config.edge_types)}")
    logger.info(f"Tactics: {len(schema_config.tactics)}")
    logger.info(f"Strategies: {len(schema_config.strategies)}")
    
    # Test node type validation
    valid_node_types = ["attribute", "functional_consequence", "psychosocial_consequence", "value"]
    for node_type in valid_node_types:
        is_valid = schema_loader.validate_node_type(node_type)
        logger.info(f"Node type '{node_type}' validation: {is_valid}")
        assert is_valid, f"Node type '{node_type}' should be valid"
    
    # Test invalid node type
    is_valid = schema_loader.validate_node_type("invalid_type")
    logger.info(f"Invalid node type validation: {is_valid}")
    assert not is_valid, "Invalid node type should fail validation"
    
    # Test edge type validation
    is_valid = schema_loader.validate_edge_type("leads_to", "attribute", "functional_consequence")
    logger.info(f"Valid edge validation: {is_valid}")
    assert is_valid, "Valid edge should pass validation"
    
    # Test invalid edge type
    is_valid = schema_loader.validate_edge_type("invalid_edge", "attribute", "functional_consequence")
    logger.info(f"Invalid edge type validation: {is_valid}")
    assert not is_valid, "Invalid edge type should fail validation"
    
    logger.info("✓ SchemaLoader tests passed")
    return schema_loader


async def test_extraction_prompt_builder(schema_loader: SchemaLoader):
    """Test ExtractionPromptBuilder."""
    logger.info("=== Testing ExtractionPromptBuilder ===")
    
    prompt_builder = ExtractionPromptBuilder()
    
    # Test prompt building
    sample_response = "I really like how convenient the packaging is. It saves me time in the morning when I'm getting ready for work. The resealable feature keeps everything fresh."
    history = []
    existing_nodes = []
    
    messages, function_schema = prompt_builder.build_prompt(
        response=sample_response,
        history=history,
        existing_nodes=existing_nodes
    )
    
    logger.info(f"Generated {len(messages)} messages")
    logger.info(f"Function schema name: {function_schema.get('name', 'unknown')}")
    
    # Check system prompt contains schema context
    system_message = messages[0]["content"]
    assert "NODE TYPES:" in system_message, "System prompt should contain node types"
    assert "EDGE TYPES:" in system_message, "System prompt should contain edge types"
    
    # Check user prompt contains the response
    user_message = messages[1]["content"]
    assert sample_response in user_message, "User prompt should contain the response"
    
    logger.info("✓ ExtractionPromptBuilder tests passed")
    return prompt_builder


async def test_extraction_validator():
    """Test ExtractionValidator."""
    logger.info("=== Testing ExtractionValidator ===")
    
    validator = ExtractionValidator()
    
    # Test valid extraction
    valid_extraction = {
        "nodes": [
            {
                "type": "attribute",
                "label": "packaging",
                "quote": "I really like how convenient the packaging is"
            },
            {
                "type": "functional_consequence",
                "label": "saves_time",
                "quote": "It saves me time in the morning"
            }
        ],
        "edges": [
            {
                "type": "leads_to",
                "source": "packaging",
                "target": "saves_time",
                "quote": "convenient the packaging is. It saves me time",
                "confidence": 0.9
            }
        ]
    }
    
    result = validator.validate_extraction(valid_extraction, "test response")
    logger.info(f"Valid extraction result: {len(result['nodes'])} nodes, {len(result['edges'])} edges")
    logger.info(f"Validation errors: {result['errors']}")
    
    assert len(result["nodes"]) == 2, "Should have 2 valid nodes"
    assert len(result["edges"]) == 1, "Should have 1 valid edge"
    assert len(result["errors"]) == 0, "Should have no validation errors"
    
    # Test invalid extraction (missing required fields)
    invalid_extraction = {
        "nodes": [
            {
                "type": "invalid_type",
                "label": "test",
                "quote": "test quote"
            }
        ],
        "edges": []
    }
    
    result = validator.validate_extraction(invalid_extraction, "test response")
    logger.info(f"Invalid extraction errors: {result['errors']}")
    
    assert len(result["errors"]) > 0, "Should have validation errors for invalid extraction"
    
    logger.info("✓ ExtractionValidator tests passed")
    return validator


async def test_complete_pipeline():
    """Test the complete extraction pipeline."""
    logger.info("=== Testing Complete Extraction Pipeline ===")
    
    # Initialize components
    schema_loader = SchemaLoader()
    prompt_builder = ExtractionPromptBuilder()
    validator = ExtractionValidator()
    llm_client = MockLLMClient()
    
    response_processor = ResponseProcessor(llm_client, prompt_builder, validator)
    
    # Test data
    sample_response = "I really like how convenient the packaging is. It saves me time in the morning when I'm getting ready for work. The resealable feature keeps everything fresh."
    conversation_history = []
    existing_nodes = []
    turn_number = 1
    
    logger.info(f"Processing response: {sample_response}")
    
    # Process response
    result = await response_processor.process_response(
        participant_response=sample_response,
        conversation_history=conversation_history,
        existing_nodes=existing_nodes,
        turn_number=turn_number
    )
    
    logger.info(f"Extraction success: {result.success}")
    logger.info(f"Nodes added: {len(result.delta.nodes_added)}")
    logger.info(f"Edges added: {len(result.delta.edges_added)}")
    logger.info(f"Validation errors: {result.metadata.validation_errors}")
    logger.info(f"Latency: {result.metadata.latency_ms}ms")
    logger.info(f"Tokens used: {result.metadata.tokens_used}")
    
    # Validate results
    assert result.success, "Extraction should be successful"
    assert len(result.delta.nodes_added) > 0, "Should extract some nodes"
    assert len(result.delta.edges_added) > 0, "Should extract some edges"
    
    # Check extracted nodes
    node_types = [node.type for node in result.delta.nodes_added]
    expected_node_types = ["attribute", "functional_consequence"]
    
    for expected_type in expected_node_types:
        assert expected_type in node_types, f"Should extract {expected_type} nodes"
    
    # Check extracted edges
    edge_types = [edge.type for edge in result.delta.edges_added]
    expected_edge_types = ["leads_to"]
    
    for expected_type in expected_edge_types:
        assert expected_type in edge_types, f"Should extract {expected_type} edges"
    
    # Check confidence scores
    for edge in result.delta.edges_added:
        assert 0.0 <= edge.confidence <= 1.0, f"Edge confidence should be between 0 and 1, got {edge.confidence}"
        assert edge.quote, "Each edge should have a supporting quote"
    
    # Check node quotes
    for node in result.delta.nodes_added:
        assert node.quote, "Each node should have a supporting quote"
    
    # Print detailed results
    logger.info("\n=== Extracted Nodes ===")
    for node in result.delta.nodes_added:
        logger.info(f"- {node.type}: {node.label} (quote: '{node.quote}')")
    
    logger.info("\n=== Extracted Edges ===")
    for edge in result.delta.edges_added:
        logger.info(f"- {edge.type}: {edge.source} -> {edge.target} (confidence: {edge.confidence}, quote: '{edge.quote}')")
    
    logger.info("✓ Complete pipeline test passed")
    return result


async def test_schema_compliance():
    """Test that extraction follows schema v0.2 rules."""
    logger.info("=== Testing Schema v0.2 Compliance ===")
    
    schema_loader = SchemaLoader()
    validator = ExtractionValidator()
    
    # Get valid node and edge types from schema
    valid_node_types = {nt.name for nt in schema_loader.get_node_types()}
    valid_edge_types = {et.name for et in schema_loader.get_edge_types()}
    
    logger.info(f"Valid node types: {valid_node_types}")
    logger.info(f"Valid edge types: {valid_edge_types}")
    
    # Test that only valid types are used
    test_extractions = [
        {
            "name": "Valid extraction",
            "data": {
                "nodes": [
                    {"type": "attribute", "label": "test_attr", "quote": "test"},
                    {"type": "functional_consequence", "label": "test_func", "quote": "test"}
                ],
                "edges": [
                    {"type": "leads_to", "source": "test_attr", "target": "test_func", "quote": "test", "confidence": 0.8}
                ]
            },
            "should_pass": True
        },
        {
            "name": "Invalid node type",
            "data": {
                "nodes": [
                    {"type": "invalid_type", "label": "test", "quote": "test"}
                ],
                "edges": []
            },
            "should_pass": False
        },
        {
            "name": "Invalid edge type",
            "data": {
                "nodes": [
                    {"type": "attribute", "label": "test_attr", "quote": "test"}
                ],
                "edges": [
                    {"type": "invalid_edge", "source": "test_attr", "target": "test_attr", "quote": "test", "confidence": 0.5}
                ]
            },
            "should_pass": False
        }
    ]
    
    for test in test_extractions:
        result = validator.validate_extraction(test["data"], "test response")
        passed = len(result["errors"]) == 0
        
        logger.info(f"{test['name']}: {'PASS' if passed == test['should_pass'] else 'FAIL'}")
        if result["errors"]:
            logger.info(f"  Errors: {result['errors']}")
        
        assert passed == test["should_pass"], f"Test '{test['name']}' validation result mismatch"
    
    logger.info("✓ Schema compliance tests passed")


async def main():
    """Run all tests."""
    logger.info("Starting extraction pipeline tests with schema v0.2")
    logger.info(f"Test response: 'I really like how convenient the packaging is. It saves me time in the morning when I'm getting ready for work. The resealable feature keeps everything fresh.'")
    
    try:
        # Run individual component tests
        schema_loader = await test_schema_loader()
        prompt_builder = await test_extraction_prompt_builder(schema_loader)
        validator = await test_extraction_validator()
        
        # Run complete pipeline test
        pipeline_result = await test_complete_pipeline()
        
        # Run schema compliance tests
        await test_schema_compliance()
        
        logger.info("\n🎉 All tests passed! The extraction pipeline with schema v0.2 is working correctly.")
        
        # Summary
        logger.info("\n=== Test Summary ===")
        logger.info(f"✓ SchemaLoader: Loaded schema v{schema_loader.load_schema().schema_version}")
        logger.info(f"✓ ExtractionPromptBuilder: Built prompts with schema context")
        logger.info(f"✓ ExtractionValidator: Validated extractions against schema rules")
        logger.info(f"✓ ResponseProcessor: Extracted {len(pipeline_result.delta.nodes_added)} nodes and {len(pipeline_result.delta.edges_added)} edges")
        logger.info(f"✓ Schema Compliance: All node and edge types validated correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Create event loop and run tests
    result = asyncio.run(main())
    exit(0 if result else 1)