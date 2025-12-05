"""
Graph Extraction Orchestrator - Integrates concept extraction into interview pipeline.
Processes participant responses and updates graph state with extracted concepts.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.core.models import GraphState, InterviewState, Node, Edge
from src.core.extraction_models import GraphDelta, ExtractionResult
from src.interview.extraction.response_processor import ResponseProcessor
from src.interview.extraction.concept_extractor import ConceptExtractor

logger = logging.getLogger(__name__)


class GraphExtractionOrchestrator:
    """
    Orchestrates concept extraction and graph updates during interviews.
    
    Integrates the extraction pipeline with the existing graph-driven interview system
    by processing participant responses and updating the knowledge graph.
    """
    
    def __init__(self, response_processor: ResponseProcessor, 
                 concept_extractor: Optional[ConceptExtractor] = None):
        """Initialize with extraction components."""
        self.response_processor = response_processor
        self.concept_extractor = concept_extractor
        logger.info("GraphExtractionOrchestrator initialized")
    
    async def process_participant_response(self, response_text: str, 
                                         conversation_history: List[Dict[str, str]],
                                         current_graph: GraphState,
                                         interview_state: InterviewState) -> GraphDelta:
        """
        Process participant response and extract concepts to update graph.
        
        Args:
            response_text: Participant's response text
            conversation_history: Recent conversation turns
            current_graph: Current knowledge graph state
            interview_state: Current interview state
            
        Returns:
            GraphDelta with changes applied to the graph
        """
        logger.info(f"Processing participant response for turn {interview_state.turn_number}")
        
        try:
            # Extract existing node labels for context
            existing_nodes = list(current_graph.nodes.keys())
            
            # Process response through extraction pipeline
            extraction_result = await self.response_processor.process_response(
                participant_response=response_text,
                conversation_history=conversation_history,
                existing_nodes=existing_nodes,
                turn_number=interview_state.turn_number
            )
            
            if not extraction_result.success or extraction_result.delta.is_empty():
                logger.info(f"No concepts extracted from response: {extraction_result.error_message}")
                return extraction_result.delta
            
            # Apply extracted concepts to graph
            updated_delta = self._apply_extraction_to_graph(
                delta=extraction_result.delta,
                current_graph=current_graph,
                turn_number=interview_state.turn_number
            )
            
            logger.info(f"Applied extraction: {updated_delta.get_summary()}")
            return updated_delta
            
        except Exception as e:
            logger.error(f"Failed to process participant response: {e}", exc_info=True)
            return GraphDelta()  # Return empty delta on error
    
    def _apply_extraction_to_graph(self, delta: GraphDelta, current_graph: GraphState,
                                 turn_number: int) -> GraphDelta:
        """
        Apply extracted concepts to the knowledge graph.
        
        Args:
            delta: Extracted concepts from response
            current_graph: Current graph state to update
            turn_number: Current turn number
            
        Returns:
            GraphDelta with actual changes applied (accounting for duplicates)
        """
        applied_nodes = []
        applied_edges = []
        validation_bugs = []  # Track validation bugs (missing node references)
        
        # Add nodes to graph
        for extracted_node in delta.nodes_added:
            node_id = extracted_node.label
            
            # Update existing node with new quote if node already exists
            if node_id in current_graph.nodes:
                existing_node = current_graph.nodes[node_id]
                # Append new supporting quote if it's unique and non-empty
                if extracted_node.quote and extracted_node.quote not in existing_node.source_quotes:
                    existing_node.source_quotes.append(extracted_node.quote)
                    logger.info(f"Added new quote to existing node: {node_id}")
                else:
                    logger.debug(f"Node {node_id} already exists with same quote, skipping")
                continue
            
            # Create new node
            node = Node(
                id=node_id,
                type=extracted_node.type,
                label=extracted_node.label,
                source_quotes=[extracted_node.quote],
                creation_turn=turn_number,
                visit_count=0
            )
            
            # Add to graph
            current_graph.add_node(node)
            applied_nodes.append(extracted_node)
            logger.debug(f"Added node: {node_id} ({extracted_node.type})")
        
        # Add edges to graph
        for extracted_edge in delta.edges_added:
            edge_id = f"{extracted_edge.source}-{extracted_edge.type}-{extracted_edge.target}"

            # Update existing edge with better evidence
            if edge_id in current_graph.edges:
                existing_edge = current_graph.edges[edge_id]
                updated = False

                # Update confidence if new evidence is stronger
                if extracted_edge.confidence > existing_edge.confidence:
                    old_confidence = existing_edge.confidence
                    existing_edge.confidence = extracted_edge.confidence
                    logger.info(f"Updated edge {edge_id} confidence: {old_confidence:.2f} -> {extracted_edge.confidence:.2f}")
                    updated = True

                # Append new supporting quote if it's unique and non-empty
                if extracted_edge.quote and extracted_edge.quote != existing_edge.source_quote:
                    # Check if quote is already in existing quote (to avoid duplicates)
                    if extracted_edge.quote not in existing_edge.source_quote:
                        existing_edge.source_quote = existing_edge.source_quote + " | " + extracted_edge.quote
                        logger.debug(f"Appended new quote to edge {edge_id}")
                        updated = True

                if updated:
                    logger.debug(f"Edge {edge_id} updated with new evidence")
                else:
                    logger.debug(f"Edge {edge_id} already exists with same or better evidence, skipping")
                continue
            
            # Verify nodes exist (should be guaranteed by validation)
            # If missing, create placeholder nodes to preserve edges that passed validation
            if extracted_edge.source not in current_graph.nodes:
                logger.error(f"VALIDATION BUG: Edge {edge_id} passed validation but source '{extracted_edge.source}' not in graph")
                logger.info(f"Creating placeholder node for '{extracted_edge.source}' to preserve edge")
                placeholder_node = Node(
                    id=extracted_edge.source,
                    label=extracted_edge.source,
                    type="unknown",
                    creation_turn=turn_number,
                    source_quotes=["[Node added as placeholder for edge preservation]"]
                )
                current_graph.add_node(placeholder_node)
                # Track validation bug for metrics
                validation_bugs.append({
                    "edge": edge_id,
                    "missing_node": extracted_edge.source,
                    "node_type": "source",
                    "reason": "Source node not in graph despite passing validation"
                })

            if extracted_edge.target not in current_graph.nodes:
                logger.error(f"VALIDATION BUG: Edge {edge_id} passed validation but target '{extracted_edge.target}' not in graph")
                logger.info(f"Creating placeholder node for '{extracted_edge.target}' to preserve edge")
                placeholder_node = Node(
                    id=extracted_edge.target,
                    label=extracted_edge.target,
                    type="unknown",
                    creation_turn=turn_number,
                    source_quotes=["[Node added as placeholder for edge preservation]"]
                )
                current_graph.add_node(placeholder_node)
                # Track validation bug for metrics
                validation_bugs.append({
                    "edge": edge_id,
                    "missing_node": extracted_edge.target,
                    "node_type": "target",
                    "reason": "Target node not in graph despite passing validation"
                })
            
            # Create new edge
            edge = Edge(
                id=edge_id,
                type=extracted_edge.type,
                source=extracted_edge.source,
                target=extracted_edge.target,
                source_quote=extracted_edge.quote,
                creation_turn=turn_number,
                confidence=extracted_edge.confidence
            )
            
            # Add to graph
            current_graph.add_edge(edge)
            applied_edges.append(extracted_edge)
            logger.debug(f"Added edge: {edge_id} ({extracted_edge.type}, confidence: {extracted_edge.confidence})")
        
        # Return delta with only actually applied changes
        metadata = {
            "applied_nodes": len(applied_nodes),
            "applied_edges": len(applied_edges),
            "skipped_nodes": len(delta.nodes_added) - len(applied_nodes),
            "skipped_edges": len(delta.edges_added) - len(applied_edges),
            "turn_number": turn_number
        }

        # Add validation bugs to metadata if any occurred
        if validation_bugs:
            metadata["validation_bugs"] = validation_bugs
            logger.warning(f"Found {len(validation_bugs)} validation bugs in this extraction")

        return GraphDelta(
            nodes_added=applied_nodes,
            edges_added=applied_edges,
            metadata=metadata
        )
    
    async def extract_initial_concepts(self, concept_description: str) -> GraphDelta:
        """
        Extract initial seed concepts from product/concept description.
        
        Args:
            concept_description: Initial product or concept description
            
        Returns:
            GraphDelta with seed concepts for graph initialization
        """
        if not self.concept_extractor:
            logger.warning("Concept extractor not available")
            return GraphDelta()
        
        try:
            # Extract seed concepts
            seed_nodes = await self.concept_extractor.extract_seed_concepts(concept_description)
            
            if not seed_nodes:
                logger.info("No seed concepts extracted")
                return GraphDelta()
            
            # Convert to GraphDelta format
            delta = GraphDelta(
                nodes_added=seed_nodes,
                edges_added=[],
                metadata={
                    "source": "initial_concept_extraction",
                    "concept_description": concept_description[:100]
                }
            )
            
            logger.info(f"Extracted {len(seed_nodes)} initial seed concepts")
            return delta
            
        except Exception as e:
            logger.error(f"Initial concept extraction failed: {e}")
            return GraphDelta()
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about extraction performance."""
        return {
            "extraction_enabled": True,
            "schema_path": self.response_processor.prompt_builder.schema_path,
            "validation_stages": 2,
            "confidence_threshold": 0.6,
            "max_retries": 2
        }