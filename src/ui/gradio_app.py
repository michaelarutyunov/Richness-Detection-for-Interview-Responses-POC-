"""
Gradio interface for the AI Interview System (NEW architecture).

Provides a web-based chat interface for conducting interviews with graph-driven question generation.
Uses the new InterviewController architecture with simplified state management.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import gradio as gr
from dotenv import load_dotenv

# Import NEW system components
from controller import InterviewController
from utils.llm_manager import LLMManager
from utils.concept_parser import ConceptParser, ParsedConcept
from utils.logger import InterviewLogger
from core.graph import Graph
from core.state import GraphState, CoverageState

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewUI:
    """Gradio-based user interface for AI interviews using NEW architecture."""

    def __init__(self):
        """Initialize the interview UI."""
        self.current_controller: Optional[InterviewController] = None
        self.current_session_id: Optional[str] = None
        self.llm_manager: Optional[LLMManager] = None
        self.concept_parser: Optional[ConceptParser] = None
        
        # Config paths (using src/config structure)
        self.config_dir = Path(__file__).parent.parent / "config"
        self.schema_path = self.config_dir / "schemas" / "means_end_chain.yaml"
        self.logic_path = self.config_dir / "interview_logic.yaml"
        self.llm_config_path = self.config_dir / "llm_config.yaml"
        self.concepts_dir = self.config_dir / "concepts"
        
        logger.info("InterviewUI initialized with NEW architecture")

    def setup_llm_manager(self) -> bool:
        """Set up LLM manager using new configuration system."""
        try:
            logger.info("Setting up LLM manager with new configuration")
            
            # Check if config files exist
            if not self.llm_config_path.exists():
                raise FileNotFoundError(f"LLM config not found: {self.llm_config_path}")
            
            if not self.logic_path.exists():
                raise FileNotFoundError(f"Interview logic not found: {self.logic_path}")
                
            if not self.schema_path.exists():
                raise FileNotFoundError(f"Schema not found: {self.schema_path}")
            
            # Initialize LLM manager
            self.llm_manager = LLMManager.from_config_file(str(self.llm_config_path))
            
            # Check available providers
            available_providers = self.llm_manager.list_available_providers()
            if not available_providers:
                logger.warning("No LLM providers available - check API keys in .env")
                return False
            
            logger.info(f"Available LLM providers: {available_providers}")
            
            # Initialize concept parser
            self.concept_parser = ConceptParser(self.llm_manager)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup LLM manager: {e}")
            return False

    def get_available_concepts(self) -> List[str]:
        """Get list of available concept files."""
        try:
            if not self.concepts_dir.exists():
                return []
            
            concept_files = []
            for file_path in self.concepts_dir.glob("*.md"):
                concept_files.append(file_path.name)
            
            return sorted(concept_files)
        except Exception as e:
            logger.error(f"Error listing concepts: {e}")
            return []

    def parse_concept_text(self, concept_text: str, concept_name: str = "User Concept") -> ParsedConcept:
        """Parse concept text using LLM or fallback to simple parsing."""
        try:
            if self.concept_parser:
                # Try LLM-based parsing
                return self.concept_parser.parse_text(concept_text, concept_name)
            else:
                # Fallback to simple concept
                from utils.concept_parser import ParsedConcept, ConceptElements
                elements = ConceptElements(
                    insight="",
                    promise=concept_text[:200] if len(concept_text) > 200 else concept_text,
                    rtb=""
                )
                return ParsedConcept(
                    name=concept_name,
                    description=concept_text,
                    elements=elements
                )
        except Exception as e:
            logger.error(f"Concept parsing failed: {e}")
            # Ultimate fallback
            from utils.concept_parser import ParsedConcept, ConceptElements
            return ParsedConcept(
                name=concept_name,
                description=concept_text,
                elements=ConceptElements()
            )

    async def start_interview_with_concept(
        self, concept_text: str, selected_concept_file: Optional[str] = None
    ) -> Tuple[List[Dict], Dict, str, str, List[List], List[List], Dict]:
        """Start new interview with concept description or file."""
        if not concept_text.strip() and not selected_concept_file:
            error_msg = [{"role": "assistant", "content": "Please provide a concept description or select a concept file."}]
            return (
                error_msg,
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0, "status": "error"},
                "Error: No concept",
                "No concept provided",
                [],
                [],
                {"error": "No concept provided"}
            )

        try:
            # Setup LLM manager if not already done
            has_llm = self.setup_llm_manager()
            
            # Determine concept source
            if selected_concept_file and selected_concept_file != "None":
                # Load from file
                concept_path = self.concepts_dir / selected_concept_file
                if not concept_path.exists():
                    raise FileNotFoundError(f"Concept file not found: {concept_path}")
                
                controller = InterviewController.from_concept_file(
                    concept_path=str(concept_path),
                    schema_path=str(self.schema_path),
                    logic_path=str(self.logic_path),
                    llm_config_path=str(self.llm_config_path),
                    max_turns=20
                )
                concept_name = controller.config.concept_name
                concept_description = controller.config.concept_text
            else:
                # Parse provided text
                concept = self.parse_concept_text(concept_text, "User Concept")
                element_config = concept.get_element_config()
                
                controller = InterviewController.initialize(
                    concept_text=concept.description,
                    schema_path=str(self.schema_path),
                    logic_path=str(self.logic_path),
                    llm_config_path=str(self.llm_config_path),
                    element_config=element_config,
                    concept_name=concept.name,
                    max_turns=20
                )
                concept_name = concept.name
                concept_description = concept.description
            
            self.current_controller = controller
            self.current_session_id = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate opening question
            opening_question = controller.generate_opening()
            
            # Build initial history
            history = [{"role": "assistant", "content": opening_question}]
            
            # Initial stats
            stats = {
                "nodes": len(controller.graph.nodes),
                "edges": len(controller.graph.edges),
                "coverage": f"{len([e for e in controller.coverage_state.reference_elements if controller.coverage_state.element_node_mappings.get(e)])}/{len(controller.coverage_state.reference_elements)}",
                "turns": 0,
                "status": "active",
                "has_llm": has_llm
            }
            
            # Get initial graph data for tables
            nodes_data, edges_data = self._get_graph_table_data(controller.graph)
            
            logger.info(f"Interview started: {self.current_session_id} for concept: {concept_name}")
            
            return (
                history,
                stats,
                self.current_session_id,
                concept_description,
                nodes_data,
                edges_data,
                {"status": "started", "concept": concept_name}
            )
            
        except Exception as e:
            logger.error(f"Failed to start interview: {e}")
            error_msg = [{"role": "assistant", "content": f"Sorry, I encountered an error starting the interview: {str(e)}\n\nPlease check that your configuration files are in place and API keys are set correctly in .env"}]
            return (
                error_msg,
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0, "status": "error"},
                "Error",
                "Error occurred",
                [],
                [],
                {"error": str(e)}
            )

    async def process_response(
        self, user_response: str, history: List[Dict]
    ) -> Tuple[List[Dict], str, Dict, str, List[List], List[List], Dict]:
        """Process participant response and generate next question."""
        if not self.current_controller:
            return (
                history,
                "",
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0, "status": "error"},
                "No active session",
                [],
                [],
                {"error": "No active session"}
            )

        if not user_response.strip():
            return (
                history,
                "",
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0, "status": "error"},
                self.current_session_id or "Not started",
                [],
                [],
                {"error": "No response provided"}
            )

        try:
            # Process response and get next question
            next_question = self.current_controller.process_response(user_response)
            
            # Update history
            history.append({"role": "user", "content": user_response})
            history.append({"role": "assistant", "content": next_question})
            
            # Update stats
            stats = {
                "nodes": len(self.current_controller.graph.nodes),
                "edges": len(self.current_controller.graph.edges),
                "coverage": f"{len([e for e in self.current_controller.coverage_state.reference_elements if self.current_controller.coverage_state.element_node_mappings.get(e)])}/{len(self.current_controller.coverage_state.reference_elements)}",
                "turns": self.current_controller.state.turn_count,
                "status": "complete" if self.current_controller.state.is_complete else "active",
                "completion_reason": self.current_controller.state.completion_reason
            }
            
            # Get updated graph data for tables
            nodes_data, edges_data = self._get_graph_table_data(self.current_controller.graph)
            
            logger.info(f"Processed turn {self.current_controller.state.turn_count}")
            
            return (
                history,
                "",
                stats,
                self.current_session_id or "Active",
                nodes_data,
                edges_data,
                {"status": "processed", "turn": self.current_controller.state.turn_count}
            )
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            history.append({"role": "user", "content": user_response})
            history.append({"role": "assistant", "content": error_msg})
            
            return (
                history,
                "",
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0, "status": "error"},
                self.current_session_id or "Error",
                [],
                [],
                {"error": str(e)}
            )

    def _get_graph_table_data(self, graph: Graph) -> Tuple[List[List], List[List]]:
        """Get graph data for display tables."""
        if not graph:
            return [], []
        
        # Nodes data: [ID, Type, Label, Timestamp, Ambiguous]
        nodes_data = []
        for node in graph.nodes.values():
            nodes_data.append([
                node.id[:8] + "..." if len(node.id) > 8 else node.id,
                node.node_type or "unknown",
                node.label,
                node.timestamp.strftime("%H:%M:%S"),
                "Yes" if node.is_ambiguous else "No"
            ])
        
        # Sort by timestamp (newest first)
        nodes_data.sort(key=lambda x: x[3], reverse=True)
        
        # Edges data: [ID, Type, Source, Target, Relation]
        edges_data = []
        for edge in graph.edges.values():
            source_node = graph.get_node(edge.source_id)
            target_node = graph.get_node(edge.target_id)
            source_label = source_node.label if source_node else edge.source_id[:8]
            target_label = target_node.label if target_node else edge.target_id[:8]
            
            edges_data.append([
                edge.id[:8] + "..." if len(edge.id) > 8 else edge.id,
                edge.relation_type,
                source_label,
                target_label,
                f"{source_label} → {target_label}"
            ])
        
        return nodes_data, edges_data

    async def export_json_file(self) -> Optional[str]:
        """Export complete session data as JSON."""
        try:
            if not self.current_controller or not self.current_session_id:
                logger.warning("JSON export attempted with no active session")
                return None

            logger.info(f"Exporting JSON for session {self.current_session_id}")
            
            import json
            import tempfile
            
            # Get complete session data with proper serialization
            try:
                session_data = self.current_controller.export_session()
                
                # Ensure all data is JSON serializable by converting Pydantic models
                export_data = {
                    "session_id": self.current_session_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "system_version": "NEW_architecture_v1",
                    "metadata": {
                        "concept_name": self.current_controller.config.concept_name,
                        "turns_completed": self.current_controller.state.turn_count,
                        "is_complete": self.current_controller.state.is_complete,
                        "completion_reason": self.current_controller.state.completion_reason,
                        "graph_nodes": len(self.current_controller.graph.nodes),
                        "graph_edges": len(self.current_controller.graph.edges),
                        "coverage_satisfied": self.current_controller.coverage_state.is_satisfied(),
                        "coverage_gaps": len(self.current_controller.coverage_state.get_gaps())
                    },
                    "interview_summary": self.current_controller.get_interview_summary()
                }
                
                # Test JSON serialization
                json.dumps(export_data, default=str)  # Use default=str for any remaining issues
                
            except Exception as serialize_error:
                logger.warning(f"Full session export failed, falling back to summary: {serialize_error}")
                # Fallback to just summary data
                export_data = {
                    "session_id": self.current_session_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "system_version": "NEW_architecture_v1",
                    "note": "Full export failed, providing summary only",
                    "interview_summary": self.current_controller.get_interview_summary(),
                    "transcript": self.current_controller.get_transcript()
                }
            
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                prefix=f"interview_{self.current_session_id}_",
            ) as temp_file:
                json.dump(export_data, temp_file, indent=2, default=str)
            
            logger.info(f"JSON export successful: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return None

    async def export_transcript_file(self) -> Optional[str]:
        """Export interview transcript as text file."""
        try:
            if not self.current_controller or not self.current_session_id:
                logger.warning("Transcript export attempted with no active session")
                return None

            logger.info(f"Exporting transcript for session {self.current_session_id}")
            
            import tempfile
            
            # Get transcript from controller
            transcript = self.current_controller.get_transcript()
            
            # Add header with session info
            full_transcript = f"""Interview Session: {self.current_session_id}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Concept: {self.current_controller.config.concept_name}
Turns: {self.current_controller.state.turn_count}
Status: {'Complete' if self.current_controller.state.is_complete else 'Incomplete'}

{transcript}

--- Session Summary ---
Total Nodes: {len(self.current_controller.graph.nodes)}
Total Edges: {len(self.current_controller.graph.edges)}
Coverage Satisfied: {self.current_controller.coverage_state.is_satisfied()}
Coverage Gaps: {len(self.current_controller.coverage_state.get_gaps())}
"""
            
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                prefix=f"transcript_{self.current_session_id}_",
            )
            temp_file.write(full_transcript)
            temp_file.close()
            
            logger.info(f"Transcript export successful: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"Transcript export failed: {e}")
            return None

    def _build_header(self):
        """Build header section."""
        gr.Markdown(
            """
            # 🎙️ Automonous Graph-driven AI Interviewer
            """
        )

    def _build_concept_input_section(self) -> Tuple:
        """Build concept input section."""
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Step 1: Describe the Concept")
                concept_input = gr.Textbox(
                    label="Concept Description",
                    placeholder="E.g., 'A sustainable coffee maker that uses biodegradable pods and has a built-in grinder'",
                    lines=2,
                    value="A premium coffee subscription service that delivers freshly roasted beans from local roasters every month.",
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Or Load from File")
                available_concepts = self.get_available_concepts()
                concept_file_dropdown = gr.Dropdown(
                    choices=["None"] + available_concepts,
                    label="Select Concept File",
                    value="None",
                    info="Choose a concept file or use text input above"
                )
        
        with gr.Row():
            start_btn = gr.Button("Start Interview", variant="primary", size="lg")
            status_text = gr.Textbox(
                label="Status",
                value="Ready to start",
                interactive=False,
                max_lines=1
            )

        return concept_input, concept_file_dropdown, start_btn, status_text

    def _build_interview_tab(self) -> Tuple:
        """Build interview chat tab."""
        with gr.TabItem("💬 Interview"):
            with gr.Row():
                # Left column: Chat interface
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Interview Conversation",
                        height=400,
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
                            "coverage": "0/0",
                            "turns": 0,
                            "status": "ready",
                            "completion_reason": None
                        },
                    )

                    # Concept display
                    concept_display = gr.Textbox(
                        label="Current Concept",
                        value="No concept loaded",
                        interactive=False,
                        lines=2
                    )

            # Instructions
            with gr.Accordion("ℹ️ How to use", open=False):
                gr.Markdown(
                    """
                    **Instructions:**
                    1. Describe a product/concept in the text box above, or select a concept file
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

                    **Setup Requirements:**
                    - Configuration files must be in src/config/
                    - Set API keys in .env file (ANTHROPIC_API_KEY, etc.)
                    - Without API keys, system will show clear error messages
                    """
                )

        return (chatbot, user_input, submit_btn, clear_btn,
                session_id_display, graph_stats, concept_display)

    def _build_graph_tab(self) -> Tuple:
        """Build graph visualization tab with node/edge tables."""
        with gr.TabItem("📊 Knowledge Graph"):
            gr.Markdown("### Current Knowledge Graph")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Nodes** (Concepts mentioned)")
                    nodes_table = gr.Dataframe(
                        headers=["ID", "Type", "Label", "Created", "Ambiguous"],
                        label="Graph Nodes",
                        interactive=False,
                        wrap=True,
                        column_widths=["15%", "15%", "40%", "15%", "15%"],
                    )
                
                with gr.Column():
                    gr.Markdown("**Edges** (Relationships between concepts)")
                    edges_table = gr.Dataframe(
                        headers=["ID", "Type", "Source", "Target", "Relationship"],
                        label="Graph Edges",
                        interactive=False,
                        wrap=True,
                        column_widths=["15%", "20%", "25%", "25%", "15%"],
                    )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Graph Data", size="sm")
                graph_summary = gr.Textbox(
                    label="Graph Summary",
                    value="No graph data yet",
                    interactive=False,
                    lines=3
                )

        return nodes_table, edges_table, refresh_btn, graph_summary

    def _build_export_tab(self) -> Tuple:
        """Build export tab."""
        with gr.TabItem("💾 Export"):
            gr.Markdown(
                """
                ### Export Interview Results
                Download session data and conversation transcript.
                """
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Session Data**")
                    export_json_btn = gr.Button(
                        "📥 Download JSON (Complete Session)",
                        variant="primary",
                        size="lg",
                    )
                    json_file = gr.File(
                        label="JSON File (complete session data)",
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

            with gr.Accordion("ℹ️ Export Information", open=False):
                gr.Markdown(
                    """
                    **JSON Export:** Complete session data including:
                    - Interview configuration and summary
                    - Full knowledge graph (nodes & edges)
                    - Complete conversation history
                    - Coverage tracking data
                    - Schema validation results

                    **Transcript Export:** Clean text format with:
                    - Session metadata (ID, date, concept)
                    - Full conversation history
                    - Session summary statistics
                    """
                )

        return export_json_btn, json_file, export_transcript_btn, transcript_file

    def _wire_event_handlers(
        self,
        start_btn, submit_btn, clear_btn, refresh_btn,
        export_json_btn, export_transcript_btn,
        concept_input, concept_file_dropdown, user_input, chatbot,
        session_id_display, graph_stats, concept_display, status_text,
        nodes_table, edges_table, graph_summary,
        json_file, transcript_file
    ):
        """Wire all event handlers to UI components."""

        # Start interview
        start_btn.click(
            fn=self.start_interview_with_concept,
            inputs=[concept_input, concept_file_dropdown],
            outputs=[
                chatbot,
                graph_stats,
                session_id_display,
                concept_display,
                nodes_table,
                edges_table,
                status_text
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
                nodes_table,
                edges_table,
                status_text
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
                nodes_table,
                edges_table,
                status_text
            ],
        )

        # Clear button
        clear_btn.click(
            fn=lambda: (
                [],  # chatbot
                "",  # user_input
                {"nodes": 0, "edges": 0, "coverage": "0/0", "turns": 0, "status": "ready", "completion_reason": None},  # graph_stats
                "Not started",  # session_id_display
                "No concept loaded",  # concept_display
                [],  # nodes_table
                [],  # edges_table
                "Ready to start"  # status_text
            ),
            outputs=[chatbot, user_input, graph_stats, session_id_display, concept_display, nodes_table, edges_table, status_text],
        )

        # Refresh graph data
        refresh_btn.click(
            fn=lambda: self._refresh_graph_data() if self.current_controller else ([], [], "No active session"),
            outputs=[nodes_table, edges_table, graph_summary]
        )

        # Export handlers
        export_json_btn.click(
            fn=self.export_json_file,
            outputs=[json_file],
        )

        export_transcript_btn.click(
            fn=self.export_transcript_file,
            outputs=[transcript_file],
        )

    def _refresh_graph_data(self) -> Tuple[List[List], List[List], str]:
        """Refresh graph data for tables."""
        if not self.current_controller:
            return [], [], "No active session"
        
        nodes_data, edges_data = self._get_graph_table_data(self.current_controller.graph)
        summary = self.current_controller.graph.summary()
        
        return nodes_data, edges_data, summary

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface with event handlers wired inside Blocks context."""
        with gr.Blocks(title="AI Interview Assistant (NEW Architecture)") as app:
            # Build UI sections using helper methods
            self._build_header()
            concept_input, concept_file_dropdown, start_btn, status_text = self._build_concept_input_section()

            with gr.Tabs():
                # Interview tab
                (chatbot, user_input, submit_btn, clear_btn,
                 session_id_display, graph_stats, concept_display) = self._build_interview_tab()

                # Graph tab
                nodes_table, edges_table, refresh_btn, graph_summary = self._build_graph_tab()

                # Export tab
                export_json_btn, json_file, export_transcript_btn, transcript_file = self._build_export_tab()

            # Wire event handlers INSIDE context
            self._wire_event_handlers(
                start_btn, submit_btn, clear_btn, refresh_btn,
                export_json_btn, export_transcript_btn,
                concept_input, concept_file_dropdown, user_input, chatbot,
                session_id_display, graph_stats, concept_display, status_text,
                nodes_table, edges_table, graph_summary,
                json_file, transcript_file
            )

        logger.info("Gradio interface (NEW architecture) built and event handlers wired")
        return app


def launch_app(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
    """
    Launch the Gradio interview application.

    Args:
        share: Whether to create a public share link
        server_name: Server hostname (0.0.0.0 for HF Spaces)
        server_port: Server port
    """
    ui = InterviewUI()
    app = ui.build_interface()
    
    logger.info(f"Launching Gradio app on {server_name}:{server_port}")
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Interview Assistant (NEW Architecture)")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    launch_app(share=args.share, server_name=args.host, server_port=args.port)