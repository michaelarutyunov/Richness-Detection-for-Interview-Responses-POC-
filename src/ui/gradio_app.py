"""
Gradio interface for the AI Interview System v2.

Provides a web-based chat interface for conducting interviews with graph-driven question generation.
Compatible with HuggingFace Spaces deployment.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

# Import v2 components
from src.core.models import GraphState, InterviewState, Node, Edge
from src.interview.core import ConfigurableGraphDrivenOrchestrator
from src.interview.tactics import SchemaDrivenTacticLoader, ConfigurableQuestionGenerator

from src.llm.factory import LLMClientFactory, create_default_clients
from src.llm.dual_llm_manager import DualLLMManager
from src.config.llm_config_loader import LLMConfigLoader

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewUI:
    """Gradio-based user interface for AI interviews with LLM integration."""

    def __init__(self):
        """Initialize the interview UI."""
        self.current_orchestrator = None
        self.current_session_id = None
        self.llm_client = None
        self.tactics = []
        self.current_graph_state = GraphState()  # Persistent graph state
        self.extraction_orchestrator = None  # Concept extraction orchestrator
        self.dual_llm_manager = None  # Dual LLM manager for proper configuration
        logger.info("InterviewUI v2 initialized")

    async def setup_llm_client(self):
        """Set up LLM client and extraction components using proper YAML configuration.

        Raises:
            RuntimeError: If no API keys are configured (no LLM clients available)
        """
        try:
            # Use DualLLMManager with proper YAML configuration
            logger.info("Setting up LLM clients using DualLLMManager with YAML configuration")
            
            # Create config loader and DualLLMManager
            llm_config_loader = LLMConfigLoader("configs/llm_config.yaml")
            self.dual_llm_manager = DualLLMManager(llm_config_loader)
            
            # Initialize the manager
            success = await self.dual_llm_manager.initialize()
            
            if not success:
                logger.error("Failed to initialize DualLLMManager")
                raise RuntimeError("Dual LLM initialization failed")
            
            # Get the graph extraction client (for concept extraction)
            graph_client = self.dual_llm_manager._graph_extraction_client
            
            if not graph_client:
                logger.error("No graph extraction client available")
                raise RuntimeError("No graph extraction client available")
            
            # Use the graph extraction client for the system
            self.llm_client = graph_client
            
            logger.info(f"Using graph extraction client: {type(graph_client).__name__}")

            # Initialize extraction components with properly configured client
            self._setup_extraction_components(graph_client)
            return True

        except RuntimeError:
            # Re-raise API key errors
            raise
        except Exception as e:
            logger.error(f"Could not create LLM client with DualLLMManager: {e}")
            
            # Fallback to old method if DualLLMManager fails
            logger.warning("Falling back to create_default_clients() method")
            try:
                clients = create_default_clients()
                if clients:
                    provider, client = next(iter(clients.items()))
                    self.llm_client = client
                    logger.info(f"Using fallback LLM client: {provider}")
                    self._setup_extraction_components(client)
                    return True
                else:
                    raise RuntimeError("No LLM clients available from fallback method")
            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {fallback_error}")
                raise RuntimeError(f"Application initialization failed: {e}")
    
    def _setup_extraction_components(self, llm_client):
        """Set up concept extraction components."""
        try:
            from src.interview.extraction import ExtractionPromptBuilder, ExtractionValidator, ResponseProcessor, ConceptExtractor, GraphExtractionOrchestrator
            
            # Create extraction components
            schema_path = "schemas/means_end_chain_v0.2.yaml"
            prompt_builder = ExtractionPromptBuilder(schema_path)
            validator = ExtractionValidator(schema_path)
            response_processor = ResponseProcessor(llm_client, prompt_builder, validator)
            concept_extractor = ConceptExtractor(llm_client, prompt_builder, validator)
            
            # Create extraction orchestrator
            self.extraction_orchestrator = GraphExtractionOrchestrator(
                response_processor=response_processor,
                concept_extractor=concept_extractor
            )
            
            logger.info("Concept extraction components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize extraction components: {e}")
            self.extraction_orchestrator = None
    
    def _create_extraction_orchestrator_for_interview(self) -> Optional['GraphExtractionOrchestrator']:
        """
        Factory method to create a new extraction orchestrator instance for each interview.
        
        This prevents shared state issues by ensuring each interview gets its own
        extraction orchestrator instance, while still using the same configuration
        and components as the main UI instance.
        
        Returns:
            New GraphExtractionOrchestrator instance or None if not configured
        """
        if not self.extraction_orchestrator:
            return None
        
        try:
            from src.interview.extraction import GraphExtractionOrchestrator
            
            # Create a new instance with the same configuration as the main instance
            # This ensures isolation between interviews while reusing the same setup
            return GraphExtractionOrchestrator(
                response_processor=self.extraction_orchestrator.response_processor,
                concept_extractor=self.extraction_orchestrator.concept_extractor
            )
        except Exception as e:
            logger.warning(f"Could not create extraction orchestrator for interview: {e}")
            return None

    async def start_interview_with_concept(
        self, concept_description: str
    ) -> tuple[list, dict, str, list, list, str, dict]:
        """Start new interview with concept description."""
        if not concept_description or not concept_description.strip():
            error_msg = [
                {"role": "assistant", "content": "Please provide a concept description first."}
            ]
            return (
                error_msg,
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0},
                "Error: No concept",
                [],
                [],
                "No changes yet",
                {"total_tokens": 0, "llm_provider": "none", "questions_generated": 0},
            )

        try:
            # Set up LLM client
            has_llm = await self.setup_llm_client()
            
            # Load tactics
            tactic_loader = SchemaDrivenTacticLoader()
            self.tactics = tactic_loader.load_tactics()
            
            # Create orchestrator with extraction support
            extraction_orchestrator = self._create_extraction_orchestrator_for_interview()
            
            # Create configuration loader for configurable orchestrator
            from src.config.interview_config_loader import InterviewConfigLoader
            config_loader = InterviewConfigLoader()
            
            self.current_orchestrator = ConfigurableGraphDrivenOrchestrator(
                extraction_orchestrator=extraction_orchestrator,
                config_loader=config_loader
            )
            
            # Create session
            self.current_session_id = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create initial interview state and graph
            interview_state = InterviewState(session_id=self.current_session_id)
            self.current_graph_state = GraphState()  # Initialize persistent graph state
            
            # Use hardcoded warm-up question with concept included
            first_question = f"Please read the concept and tell me your first impression: {concept_description}"
            logger.info(f"Using hardcoded warm-up question: {first_question}")
            
            # Extract initial concepts for graph building (but don't use them for first question)
            if extraction_orchestrator and self.llm_client:
                try:
                    # Extract seed concepts from concept description
                    logger.info("Extracting initial concepts from concept description")
                    initial_delta = await extraction_orchestrator.extract_initial_concepts(concept_description)
                    
                    if not initial_delta.is_empty():
                        # Apply initial concepts to graph (function modifies graph in-place)
                        extraction_orchestrator._apply_extraction_to_graph(
                            delta=initial_delta,
                            current_graph=self.current_graph_state,
                            turn_number=0
                        )
                        logger.info(f"Applied {len(initial_delta.nodes_added)} initial seed concepts to graph")
                    
                except Exception as e:
                    logger.error(f"Failed to extract initial concepts: {e}")
                    # Continue without initial concept extraction - graph will be built during interview
                    logger.info("Continuing without initial concept extraction - graph will be built during interview")
            
            # Set initial turn for first question
            interview_state.turn_number = 0
            self.interview_turn_tracker = 0

            # Build initial history
            history = [{"role": "assistant", "content": first_question}]
            
            # Initial stats
            stats = {
                "nodes": self.current_graph_state.get_node_count(),
                "edges": self.current_graph_state.get_edge_count(),
                "coverage": "0%",
                "turns": 0,
                "has_llm": has_llm,
                "provider": self.llm_client.provider if self.llm_client else "template"
            }
            
            # Token usage placeholder
            token_usage = {
                "total_tokens": 0,
                "llm_provider": self.llm_client.provider if self.llm_client else "template",
                "questions_generated": 1
            }

            logger.info(f"Interview started: {self.current_session_id}")
            return history, stats, self.current_session_id, [], [], "Interview started", token_usage
            
        except Exception as e:
            logger.error(f"Failed to start interview: {e}")
            error_msg = [
                {
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error starting the interview: {str(e)}\n\nPlease check that your API keys are set correctly in .env",
                }
            ]
            return (
                error_msg,
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0},
                "Error",
                [],
                [],
                "Error occurred",
                {"total_tokens": 0, "llm_provider": "none", "questions_generated": 0},
            )

    async def process_response(
        self, user_response: str, history: list
    ) -> tuple[list, str, dict, str, list, list, str, dict]:
        """Process participant response and generate next question with concept extraction."""
        if not self.current_orchestrator:
            return (
                history,
                "",
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0},
                "No active session",
                [],
                [],
                "No active session",
                {"total_tokens": 0, "llm_provider": "none", "questions_generated": 0},
            )

        if not user_response or not user_response.strip():
            return (
                history,
                "",
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0},
                self.current_session_id or "Not started",
                [],
                [],
                "No response provided",
                {"total_tokens": 0, "llm_provider": "none", "questions_generated": 0},
            )

        try:
            # Create proper graph state (persist across turns)
            if not hasattr(self, 'current_graph_state'):
                self.current_graph_state = GraphState()
                logger.info("Initialized new graph state for session")
            
            # Create interview state with proper turn tracking
            interview_state = InterviewState(session_id=self.current_session_id)
            # Set turn number based on actual interview progression, not chat history calculation
            if hasattr(self, 'interview_turn_tracker'):
                interview_state.turn_number = self.interview_turn_tracker
            else:
                # For existing sessions, estimate from history but track properly going forward
                interview_state.turn_number = len([h for h in history if h["role"] == "assistant"])
                self.interview_turn_tracker = interview_state.turn_number
            
            # Add previous questions to history
            for msg in history:
                interview_state.add_question(msg["content"])
                if msg["role"] == "assistant":
                    interview_state.record_tactic_usage("emotional_contrast")  # Simplified

            # Get recent conversation history for extraction context
            recent_history = history[-3:] if len(history) > 3 else history
            
            # Increment turn number for proper tracking (before generating next question)
            interview_state.increment_turn()
            self.interview_turn_tracker = interview_state.turn_number
            
            # Process response with concept extraction and generate next question
            next_question = await self.current_orchestrator.process_response(
                response_text=user_response,
                conversation_history=recent_history,
                graph_state=self.current_graph_state,
                interview_state=interview_state
            )

            if not next_question:
                next_question = "Can you tell me more about that?"

            # Update history
            history.append({"role": "user", "content": user_response})
            history.append({"role": "assistant", "content": next_question})

            # Get current stats from actual graph
            stats = {
                "nodes": self.current_graph_state.get_node_count(),
                "edges": self.current_graph_state.get_edge_count(),
                "coverage": f"{min(100, self.current_graph_state.get_node_count() * 10)}%",
                "turns": interview_state.turn_number,
                "has_llm": self.llm_client is not None,
                "provider": self.llm_client.provider if self.llm_client else "template"
            }

            # Token usage from actual tracking (no longer estimated)
            token_usage = {
                "total_tokens": interview_state.tokens_used,
                "prompt_tokens": interview_state.prompt_tokens,
                "completion_tokens": interview_state.completion_tokens,
                "llm_provider": self.llm_client.provider if self.llm_client else "template",
                "questions_generated": interview_state.turn_number + 1
            }

            # Create summary with extraction info
            summary = f"🆕 **This turn:** Generated 1 new question"

            # Get real visualization data
            new_nodes_data, new_edges_data = self._get_visualization_data(self.current_graph_state)

            return (
                history,
                "",
                stats,
                self.current_session_id or "Active",
                new_nodes_data,
                new_edges_data,
                summary,
                token_usage,
            )
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            history.append({"role": "user", "content": user_response})
            history.append({"role": "assistant", "content": error_msg})
            
            return (
                history,
                "",
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0},
                self.current_session_id or "Error",
                [],
                [],
                "Error occurred",
                {"total_tokens": 0, "llm_provider": "none", "questions_generated": 0},
            )

    def _get_visualization_data(self, graph_state):
        """Get real visualization data from graph state."""
        if not graph_state:
            return [], []
        
        # Build nodes data
        nodes_data = []
        for node_id, node in graph_state.nodes.items():
            nodes_data.append([
                node.type,
                node.label,
                node.source_quotes[0] if node.source_quotes else "",
                str(node.creation_turn),
                str(node.visit_count)
            ])
        
        # Build edges data
        edges_data = []
        for edge_id, edge in graph_state.edges.items():
            edges_data.append([
                edge.type,
                f"{edge.source} → {edge.target}",
                edge.source_quote,
                f"{edge.confidence:.2f}"
            ])
        
        return nodes_data, edges_data
    
    def refresh_visualization(self):
        """Refresh graph visualization and tables."""
        # For v2, we'll create a simple placeholder
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_annotation(
            text="Graph visualization not implemented in v2 demo", 
            showarrow=False, 
            font={"size": 16}
        )
        
        # Simple placeholder data
        nodes_data = [
            ["concept", "Example Concept", "Sample quote", "1", "1"]
        ]
        edges_data = [
            ["relates_to", "concept1 → concept2", "Sample quote", "1"]
        ]

        return fig, nodes_data, edges_data

    async def export_graphml_file(self):
        """Export GraphML file for download."""
        try:
            if not self.current_session_id:
                logger.warning("Export attempted with no active session")
                return None

            logger.info(f"Exporting GraphML for session {self.current_session_id}")

            import tempfile
            
            # Create simple GraphML content for demo
            graphml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph id="{self.current_session_id}" edgedefault="directed">
    <node id="example" />
    <edge source="example" target="demo" />
  </graph>
</graphml>"""

            # Write to temp file
            temp_file = tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".graphml",
                delete=False,
                prefix=f"interview_{self.current_session_id}_",
            )
            temp_file.write(graphml_content.encode())
            temp_file.close()

            logger.info(f"GraphML export successful: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"GraphML export failed: {e}")
            return None

    async def export_json_file(self):
        """Export JSON file for download."""
        try:
            if not self.current_session_id:
                logger.warning("JSON export attempted with no active session")
                return None

            logger.info(f"Exporting JSON for session {self.current_session_id}")

            import json
            import tempfile

            # Create simple JSON data for demo
            json_data = {
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat(),
                "graph": {
                    "nodes": [{"id": "example", "label": "Example Node", "type": "concept"}],
                    "edges": [{"source": "example", "target": "demo", "type": "relates_to"}]
                }
            }

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                prefix=f"interview_{self.current_session_id}_",
            ) as temp_file:
                json.dump(json_data, temp_file, indent=2)

            logger.info(f"JSON export successful: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return None

    async def export_transcript_file(self):
        """Export transcript file for download."""
        try:
            if not self.current_session_id:
                logger.warning("Transcript export attempted with no active session")
                return None

            logger.info(f"Exporting transcript for session {self.current_session_id}")

            import tempfile

            # Create simple transcript for demo
            transcript_text = f"""Interview Session: {self.current_session_id}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

[AI] Tell me about your experience with this concept.
[User] This is a sample response.
[AI] Can you tell me more about that?
"""

            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                prefix=f"transcript_{self.current_session_id}_",
            )
            temp_file.write(transcript_text)
            temp_file.close()

            logger.info(f"Transcript export successful: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"Transcript export failed: {e}")
            return None

    async def export_extended_report_file(self):
        """Export extended report file for download."""
        try:
            if not self.current_session_id:
                logger.warning("Extended report export attempted with no active session")
                return None

            logger.info(f"Exporting extended report for session {self.current_session_id}")

            import tempfile

            # Create simple report for demo
            report_markdown = f"""# Interview Report

**Session ID:** {self.current_session_id}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

## Summary
- Total turns: Sample
- Graph nodes: Sample
- Graph edges: Sample
- LLM provider: {self.llm_client.provider if self.llm_client else 'template'}

## Interview Transcript
See transcript file for detailed conversation.

## Graph Analysis
See JSON/GraphML files for detailed graph data.
"""

            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".md",
                delete=False,
                prefix=f"extended_report_{self.current_session_id}_",
            )
            temp_file.write(report_markdown)
            temp_file.close()

            logger.info(f"Extended report export successful: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"Extended report export failed: {e}")
            return None

    def _build_header(self):
        """Build header section."""
        gr.Markdown(
            """
            # 🎙️ AI Interview Assistant v2
            **Graph-driven adaptive interviewing with LLM-powered question generation**

            This system uses AI to conduct natural, conversational interviews
            while building a knowledge graph of your mental model.
            """
        )

    def _build_concept_input_section(self) -> tuple:
        """Build concept input section. Returns (concept_input, start_btn)."""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1: Describe the Concept")
                concept_input = gr.Textbox(
                    label="Concept Description",
                    placeholder="E.g., 'A sustainable coffee maker that uses biodegradable pods and has a built-in grinder'",
                    lines=3,
                    value="A premium coffee subscription service that delivers freshly roasted beans from local roasters every month.",
                )
                start_btn = gr.Button("Start Interview", variant="primary", size="lg")

        return concept_input, start_btn

    def _build_interview_tab(self) -> tuple:
        """Build interview chat tab."""
        with gr.TabItem("💬 Interview"):
            with gr.Row():
                # Left column: Chat interface
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Interview Conversation",
                        height=500,
                    )

                    with gr.Row():
                        user_input = gr.Textbox(
                            label="Your Response",
                            placeholder="Type your answer here and press Submit...",
                            lines=3,
                            max_lines=5,
                            show_label=False,
                        )

                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear & Restart", size="lg")

                # Right column: Interview metadata & stats
                with gr.Column(scale=1):
                    gr.Markdown("### Interview Progress")

                    session_id_display = gr.Textbox(
                        label="Session ID",
                        value="Not started",
                        interactive=False,
                        max_lines=1,
                    )

                    gr.Markdown("### Knowledge Graph Stats")
                    graph_stats = gr.JSON(
                        label="Current Graph",
                        value={
                            "nodes": 0,
                            "edges": 0,
                            "coverage": "0%",
                            "turns": 0,
                            "has_llm": False,
                            "provider": "none"
                        },
                    )

                    # LLM Usage Display
                    gr.Markdown("### LLM Usage")
                    token_usage_display = gr.JSON(
                        label="LLM Token Consumption",
                        value={
                            "total_tokens": 0,
                            "llm_provider": "none",
                            "questions_generated": 0
                        }
                    )

                    # Dynamic delta display
                    with gr.Accordion("Latest Graph Changes", open=True):
                        delta_summary = gr.Markdown("No changes yet")

                        gr.Markdown("### Newly Added Nodes")
                        new_nodes_display = gr.Dataframe(
                            headers=["Type", "Label", "Source Quote", "Turn"],
                            value=[],
                            label="Nodes Added This Turn",
                            interactive=False,
                            wrap=True,
                            column_widths=["15%", "25%", "45%", "15%"],
                        )

                        gr.Markdown("### Newly Added Relationships")
                        new_edges_display = gr.Dataframe(
                            headers=["Type", "Relationship", "Source Quote", "Turn"],
                            value=[],
                            label="Edges Added This Turn",
                            interactive=False,
                            wrap=True,
                            column_widths=["20%", "25%", "40%", "15%"],
                        )

            # Instructions
            with gr.Accordion("ℹ️ How to use", open=False):
                gr.Markdown(
                    """
                    **Instructions:**
                    1. Describe the product/concept you want to explore in the text box above
                    2. Click **Start Interview** to begin
                    3. Read the AI interviewer's question
                    4. Type your response and click **Submit**
                    5. The AI will analyze your response and ask a follow-up question
                    6. Continue until the interview completes (typically 10-15 exchanges)

                    **Tips:**
                    - Be as detailed or brief as you like
                    - There are no right or wrong answers
                    - The AI adapts its questions based on your responses
                    - Watch the Knowledge Graph stats update in real-time!

                    **LLM Integration:**
                    - Set ANTHROPIC_API_KEY or OPENAI_API_KEY for natural questions
                    - Without API keys, uses template-based questions
                    """
                )

        return (chatbot, user_input, submit_btn, clear_btn,
                session_id_display, graph_stats, delta_summary,
                new_nodes_display, new_edges_display, token_usage_display)

    def _build_visualization_tab(self) -> tuple:
        """Build graph visualization tab."""
        with gr.TabItem("📊 Graph Visualization"):
            gr.Markdown("### Interactive Knowledge Graph")

            # Interpretation guide
            with gr.Accordion("ℹ️ How to Interpret the Graph", open=False):
                gr.Markdown(
                    """
                    **Node Colors** (by type):
                    - 🔵 Blue = Attributes (product features mentioned)
                    - 🟢 Green = Functional Consequences (what the product does/enables)
                    - 🟣 Purple = Psychosocial Consequences (how it makes you feel)
                    - 🔴 Red = Terminal Values (end goals, life values)
                    - 🟠 Orange = Instrumental Values

                    **Node Size:**
                    - Larger nodes = more visits during interview

                    **LLM Integration:**
                    - Questions are generated by AI (Claude/GPT/Kimi)
                    - More natural and context-aware than templates
                    - Adapts to your responses and graph structure
                    """
                )

            graph_plot = gr.Plot(label="Graph Structure")

            with gr.Row():
                refresh_viz_btn = gr.Button("🔄 Refresh Visualization", size="sm")

            gr.Markdown("### Graph Data Tables")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Nodes**")
                    nodes_table = gr.Dataframe(
                        headers=["ID", "Type", "Label", "Quotes", "Visit Count", "Turn"],
                        interactive=False,
                    )

                with gr.Column():
                    gr.Markdown("**Edges**")
                    edges_table = gr.Dataframe(
                        headers=["ID", "Type", "Source", "Target", "Quote", "Turn"],
                        interactive=False,
                    )

        return graph_plot, refresh_viz_btn, nodes_table, edges_table

    def _build_export_tab(self) -> tuple:
        """Build export tab."""
        with gr.TabItem("💾 Export"):
            gr.Markdown(
                """
                ### Export Interview Results
                Download graph data and conversation transcript in various formats.
                """
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Graph Formats**")
                    export_graphml_btn = gr.Button(
                        "📥 Download GraphML",
                        variant="secondary",
                        size="lg",
                    )
                    graphml_file = gr.File(
                        label="GraphML File (for Gephi, yEd, Cytoscape)",
                        visible=True,
                    )

                    export_json_btn = gr.Button(
                        "📥 Download JSON",
                        variant="secondary",
                        size="lg",
                    )
                    json_file = gr.File(
                        label="JSON File (raw graph data)",
                        visible=True,
                    )

                with gr.Column():
                    gr.Markdown("**Conversation**")
                    export_transcript_btn = gr.Button(
                        "📥 Download Transcript",
                        variant="secondary",
                        size="lg",
                    )
                    transcript_file = gr.File(
                        label="Transcript File (text)",
                        visible=True,
                    )

                    export_extended_report_btn = gr.Button(
                        "📥 Download Extended Report (Markdown)",
                        variant="primary",
                        size="lg",
                    )
                    extended_report_file = gr.File(
                        label="Extended Report (Markdown with turn-by-turn breakdown)",
                        visible=True,
                    )

        export_buttons = {
            'export_graphml_btn': export_graphml_btn,
            'export_json_btn': export_json_btn,
            'export_transcript_btn': export_transcript_btn,
            'export_extended_report_btn': export_extended_report_btn,
        }

        export_files = {
            'graphml_file': graphml_file,
            'json_file': json_file,
            'transcript_file': transcript_file,
            'extended_report_file': extended_report_file,
        }

        return export_buttons, export_files

    def _wire_event_handlers(
        self,
        start_btn, submit_btn, clear_btn, refresh_viz_btn,
        export_buttons, export_files,
        concept_input, user_input, chatbot,
        session_id_display, graph_stats,
        delta_summary, new_nodes_display, new_edges_display,
        token_usage_display,
        graph_plot, nodes_table, edges_table
    ):
        """Wire all event handlers to UI components."""

        # Start interview
        start_btn.click(
            fn=self.start_interview_with_concept,
            inputs=[concept_input],
            outputs=[
                chatbot,
                graph_stats,
                session_id_display,
                new_nodes_display,
                new_edges_display,
                delta_summary,
                token_usage_display,
            ],
        )

        # Submit response
        submit_btn.click(
            fn=self.process_response,
            inputs=[user_input, chatbot],
            outputs=[
                chatbot,
                user_input,
                graph_stats,
                session_id_display,
                new_nodes_display,
                new_edges_display,
                delta_summary,
                token_usage_display,
            ],
        )

        # User input submit (Enter key)
        user_input.submit(
            fn=self.process_response,
            inputs=[user_input, chatbot],
            outputs=[
                chatbot,
                user_input,
                graph_stats,
                session_id_display,
                new_nodes_display,
                new_edges_display,
                delta_summary,
                token_usage_display,
            ],
        )

        # Clear button
        clear_btn.click(
            fn=lambda: (
                [],
                "",
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0, "has_llm": False, "provider": "none"},
                "Not started",
            ),
            outputs=[chatbot, user_input, graph_stats, session_id_display],
        )

        # Refresh visualization
        refresh_viz_btn.click(
            fn=self.refresh_visualization,
            outputs=[graph_plot, nodes_table, edges_table],
        )

        # Export handlers
        export_buttons['export_graphml_btn'].click(
            fn=self.export_graphml_file,
            outputs=[export_files['graphml_file']],
        )

        export_buttons['export_json_btn'].click(
            fn=self.export_json_file,
            outputs=[export_files['json_file']],
        )

        export_buttons['export_transcript_btn'].click(
            fn=self.export_transcript_file,
            outputs=[export_files['transcript_file']],
        )

        export_buttons['export_extended_report_btn'].click(
            fn=self.export_extended_report_file,
            outputs=[export_files['extended_report_file']],
        )

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface with event handlers wired inside Blocks context."""
        with gr.Blocks(title="AI Interview Assistant v2") as app:
            # Build UI sections using helper methods
            self._build_header()
            concept_input, start_btn = self._build_concept_input_section()

            with gr.Tabs():
                # Interview tab
                (chatbot, user_input, submit_btn, clear_btn,
                 session_id_display, graph_stats, delta_summary,
                 new_nodes_display, new_edges_display, token_usage_display) = self._build_interview_tab()

                # Visualization tab
                graph_plot, refresh_viz_btn, nodes_table, edges_table = self._build_visualization_tab()

                # Export tab
                export_buttons, export_files = self._build_export_tab()

            # Wire event handlers INSIDE context (THIS FIXES THE BUG)
            self._wire_event_handlers(
                start_btn, submit_btn, clear_btn, refresh_viz_btn,
                export_buttons, export_files,
                concept_input, user_input, chatbot,
                session_id_display, graph_stats,
                delta_summary, new_nodes_display, new_edges_display,
                token_usage_display,
                graph_plot, nodes_table, edges_table
            )

        logger.info("Gradio interface v2 built and event handlers wired")
        return app


def launch_app(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
    """
    Launch the Gradio interview application.

    Args:
        share: Whether to create a public share link
        server_name: Server hostname (0.0.0.0 for HF Spaces)
        server_port: Port to run on (7860 for HF Spaces)
    """
    ui = InterviewUI()
    app = ui.build_interface()

    logger.info(f"Launching app v2 on {server_name}:{server_port}")

    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    launch_app()