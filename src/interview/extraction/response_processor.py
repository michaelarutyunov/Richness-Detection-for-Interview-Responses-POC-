"""
Response Processor for extracting concepts from participant responses.
Streamlined single-stage extraction with essential validation only.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.core.extraction_models import (
    ExtractedNode, ExtractedEdge, GraphDelta, ExtractionMetadata, ExtractionResult
)
from src.interview.extraction.extraction_prompt_builder import ExtractionPromptBuilder
from src.interview.extraction.extraction_validator import ExtractionValidator
from src.llm.client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class ResponseProcessor:
    """Processes participant responses to extract graph concepts efficiently."""
    
    def __init__(self, llm_client: BaseLLMClient, prompt_builder: ExtractionPromptBuilder, 
                 validator: ExtractionValidator):
        """Initialize with dependencies."""
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.validator = validator
        logger.info(f"ResponseProcessor initialized with {type(llm_client).__name__}")
    
    async def process_response(self, participant_response: str, conversation_history: List[dict],
                             existing_nodes: List[str], turn_number: int) -> ExtractionResult:
        """
        Extract concepts from participant response with streamlined validation.
        
        Pipeline:
        1. Build schema-aware prompt
        2. Single LLM call with function extraction
        3. 2-stage validation (structure + schema)
        4. Build extraction result
        
        Args:
            participant_response: Latest response from participant
            conversation_history: Recent conversation turns (last 2-3)
            existing_nodes: Currently known concept labels
            turn_number: Current turn number
            
        Returns:
            ExtractionResult with concepts and metadata
        """
        logger.info(f"Processing response for turn {turn_number}")
        logger.debug(f"Response: {participant_response[:100]}...")
        logger.debug(f"Existing nodes: {len(existing_nodes)}")
        
        start_time = datetime.now()
        
        try:
            # Stage 1: Build extraction prompt with schema context
            messages, function_schema = self.prompt_builder.build_prompt(
                response=participant_response,
                history=conversation_history,
                existing_nodes=existing_nodes
            )
            
            # Stage 2: Single LLM call with function extraction
            llm_response = await self._call_llm_with_retry(messages, function_schema)
            
            if not llm_response or not llm_response.function_call:
                logger.warning("No function call in LLM response")
                return self._create_empty_result(turn_number, start_time, "No extraction")
            
            raw_extraction = llm_response.function_call.get("arguments", {}) if llm_response.function_call else {}
            
            # Transform extraction format to match validator expectations
            # The LLM returns nodes_added/edges_added, but validator expects nodes/edges
            transformed_extraction = {
                "nodes": raw_extraction.get("nodes_added", []),
                "edges": raw_extraction.get("edges_added", [])
            }
            
            # Use transformed extraction for validation
            raw_extraction = transformed_extraction
            
            # Stage 3: 2-stage validation (structure + schema only)
            validation_result = self.validator.validate_extraction(raw_extraction, participant_response)
            
            # Stage 4: Build final extraction result
            result = self._build_extraction_result(
                validated_extraction=validation_result,
                turn_number=turn_number,
                start_time=start_time,
                llm_response=llm_response,
                validation_errors=validation_result.get("errors", [])
            )
            
            logger.info(f"Extraction complete: {result.delta.get_summary()} "
                       f"({result.metadata.latency_ms}ms, {result.metadata.tokens_used} tokens)")
            
            return result
            
        except Exception as e:
            logger.error(f"Response processing failed: {e}", exc_info=True)
            return self._create_error_result(turn_number, start_time, str(e))
    
    async def _call_llm_with_retry(self, messages: List[dict], function_schema: dict) -> Optional[LLMResponse]:
        """Call LLM with simple retry logic."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                response = await self.llm.generate_completion_with_function_call(
                    messages=messages,
                    function_schema=function_schema
                )
                if response and response.function_call:
                    return response
                
                logger.warning(f"Attempt {attempt + 1}: No function call returned")
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All LLM attempts failed after {max_retries} tries")
                    return None
        
        return None
    
    def _build_extraction_result(self, validated_extraction: dict, turn_number: int,
                               start_time: datetime, llm_response: LLMResponse,
                               validation_errors: List[str]) -> ExtractionResult:
        """Build final extraction result from validated data."""
        
        # Build nodes
        nodes = []
        for node_data in validated_extraction.get("nodes", []):
            node = ExtractedNode(
                type=node_data.get("type", "unknown"),
                label=node_data.get("label", ""),
                quote=node_data.get("quote", "")
            )
            nodes.append(node)
        
        # Build edges
        edges = []
        for edge_data in validated_extraction.get("edges", []):
            edge = ExtractedEdge(
                type=edge_data.get("type", "unknown"),
                source=edge_data.get("source", ""),
                target=edge_data.get("target", ""),
                quote=edge_data.get("quote", ""),
                confidence=edge_data.get("confidence", 0.7)
            )
            edges.append(edge)
        
        # Build delta
        delta = GraphDelta(
            nodes_added=nodes,
            edges_added=edges,
            metadata={
                "turn_number": turn_number,
                "extraction_timestamp": datetime.now().isoformat()
            }
        )
        
        # Build metadata
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        metadata = ExtractionMetadata(
            model_used=llm_response.model_used if llm_response else "unknown",
            latency_ms=latency_ms,
            tokens_used=llm_response.tokens_used if llm_response else 0,
            validation_errors=validation_errors
        )
        
        return ExtractionResult(
            delta=delta,
            metadata=metadata,
            success=len(validation_errors) == 0,
            error_message=None if len(validation_errors) == 0 else "; ".join(validation_errors)
        )
    
    def _create_empty_result(self, turn_number: int, start_time: datetime, 
                           reason: str) -> ExtractionResult:
        """Create empty extraction result."""
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return ExtractionResult(
            delta=GraphDelta(),
            metadata=ExtractionMetadata(
                latency_ms=latency_ms,
                validation_errors=[reason]
            ),
            success=False,
            error_message=reason
        )
    
    def _create_error_result(self, turn_number: int, start_time: datetime,
                           error_message: str) -> ExtractionResult:
        """Create error extraction result."""
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return ExtractionResult(
            delta=GraphDelta(),
            metadata=ExtractionMetadata(
                latency_ms=latency_ms,
                validation_errors=["Processing error"]
            ),
            success=False,
            error_message=error_message
        )
