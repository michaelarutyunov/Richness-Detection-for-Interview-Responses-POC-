"""
Response Processor for extracting concepts from participant responses.
Streamlined single-stage extraction with essential validation only.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import ValidationError

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

            # Validate expected keys and log if format is unexpected
            if raw_extraction and "nodes_added" not in raw_extraction and "nodes" not in raw_extraction:
                logger.error(f"Unexpected LLM response format. Keys: {list(raw_extraction.keys())}")
                logger.debug(f"Full response: {raw_extraction}")

            # Transform extraction format to match validator expectations
            # The LLM returns nodes_added/edges_added, but validator expects nodes/edges
            transformed_extraction = {
                "nodes": raw_extraction.get("nodes_added", raw_extraction.get("nodes", [])),
                "edges": raw_extraction.get("edges_added", raw_extraction.get("edges", []))
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

        except asyncio.TimeoutError as e:
            logger.error(f"LLM request timeout: {e}")
            # Re-raise to allow orchestrator to retry with different provider
            raise
        except ValidationError as e:
            logger.error(f"Extraction validation failed: {e}")
            return self._create_error_result(turn_number, start_time, f"Validation error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in response processing: {e}", exc_info=True)
            # Re-raise unexpected errors for visibility
            raise
    
    async def _call_llm_with_retry(self, messages: List[dict], function_schema: dict) -> Optional[LLMResponse]:
        """Call LLM with retry logic that handles both function calling and regular completion."""
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                # Try function calling first
                response = await self.llm.generate_completion_with_function_call(
                    messages=messages,
                    function_schema=function_schema
                )

                if response and response.function_call:
                    return response

                # If no function call, try regular completion as fallback
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: No function call returned, trying regular completion...")
                
                response = await self.llm.generate_completion(messages)
                if response and response.content:
                    # Parse the content to extract structured data
                    parsed_data = self._parse_completion_content(response.content, function_schema)
                    if parsed_data:
                        # Create a mock function call response
                        response.function_call = {
                            "name": function_schema.get("name", "extract_concepts"),
                            "arguments": parsed_data
                        }
                        return response

                # Still no valid response - should retry
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: No valid response returned, retrying...")
                last_error = "No valid response from LLM"
                continue  # Explicitly continue to next attempt

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: API error: {e}")
                last_error = str(e)
                if attempt < max_retries - 1:
                    continue  # Retry on API errors

        logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
        return None
    
    def _parse_completion_content(self, content: str, function_schema: dict) -> Optional[dict]:
        """Parse regular completion content to extract structured data."""
        try:
            # Try to extract JSON-like structure from the content
            import json
            import re
            
            # Look for JSON-like structure in the content
            json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}'  # Look for nested braces
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            if json_matches:
                # Try to parse the first JSON-like structure
                for match in json_matches:
                    try:
                        parsed = json.loads(match)
                        # Check if it has the expected structure
                        if 'nodes_added' in parsed or 'edges_added' in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, try to extract nodes and edges from text
            nodes = []
            edges = []
            
            # Simple pattern matching for nodes and edges
            # Look for patterns like "Node: type=attribute, label=useful_feature"
            node_pattern = r'(?:node|concept):?\s+(\w+):?\s+(\w+)'
            edge_pattern = r'(?:edge|relationship):?\s+(\w+)\s+(\w+)\s+(\w+)'
            
            # This is a simplified parser - in production, you'd want more sophisticated parsing
            lines = content.lower().split('\n')
            for line in lines:
                if 'node' in line or 'concept' in line:
                    # Try to extract node information
                    match = re.search(node_pattern, line, re.IGNORECASE)
                    if match:
                        node_type, label = match.groups()
                        nodes.append({
                            "type": node_type,
                            "label": label.lower().replace(' ', '_'),
                            "quote": line.strip()
                        })
                
                if 'edge' in line or 'relationship' in line:
                    # Try to extract edge information
                    match = re.search(edge_pattern, line, re.IGNORECASE)
                    if match:
                        source, edge_type, target = match.groups()
                        edges.append({
                            "type": edge_type,
                            "source": source.lower().replace(' ', '_'),
                            "target": target.lower().replace(' ', '_'),
                            "quote": line.strip(),
                            "confidence": 0.7
                        })
            
            if nodes or edges:
                return {
                    "nodes_added": nodes,
                    "edges_added": edges
                }
            
            # If still no structure found, return None
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse completion content: {e}")
            return None
    
    def _build_extraction_result(self, validated_extraction: dict, turn_number: int,
                               start_time: datetime, llm_response: LLMResponse,
                               validation_errors: List[str]) -> ExtractionResult:
        """Build final extraction result from validated data."""
        
        # Build nodes with label normalization
        nodes = []
        for node_data in validated_extraction.get("nodes", []):
            # Normalize label to lowercase_with_underscores format
            raw_label = node_data.get("label", "")
            normalized_label = raw_label.lower().replace(" ", "_")
            
            node = ExtractedNode(
                type=node_data.get("type", "unknown"),
                label=normalized_label,
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
        
        # Extract token usage from LLM response
        tokens_used = 0
        if llm_response:
            # Use tokens_used field if available, otherwise calculate from usage dict
            if llm_response.tokens_used > 0:
                tokens_used = llm_response.tokens_used
            elif llm_response.usage:
                # Calculate from usage dict (prompt_tokens + completion_tokens)
                tokens_used = llm_response.usage.get('prompt_tokens', 0) + llm_response.usage.get('completion_tokens', 0)
                # If total_tokens is available, use that instead
                if 'total_tokens' in llm_response.usage:
                    tokens_used = llm_response.usage['total_tokens']
        
        metadata = ExtractionMetadata(
            model_used="unknown" if llm_response is None else llm_response.model_used,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            validation_errors=validation_errors,
            extraction_timestamp=start_time  # Pass actual start time for accuracy
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
                validation_errors=[reason],
                extraction_timestamp=start_time
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
                validation_errors=["Processing error"],
                extraction_timestamp=start_time
            ),
            success=False,
            error_message=error_message
        )
