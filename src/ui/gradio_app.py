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
import networkx as nx
import plotly.graph_objects as go

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

    def get_available_schemas(self) -> List[str]:
        """Get list of available schema files."""
        try:
            schemas_dir = self.config_dir / "schemas"
            if not schemas_dir.exists():
                return []

            schema_files = []
            for file_path in schemas_dir.glob("*.yaml"):
                schema_files.append(file_path.stem)  # Get filename without extension

            return sorted(schema_files)
        except Exception as e:
            logger.error(f"Error listing schemas: {e}")
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
        self, concept_text: str, selected_concept_file: Optional[str] = None, selected_schema: Optional[str] = None
    ) -> Tuple[List[Dict], Dict, str, str, List[List], List[List], go.Figure, Dict]:
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
                self._create_plotly_graph(Graph()),
                {"error": "No concept provided"}
            )

        try:
            # Setup LLM manager if not already done
            has_llm = self.setup_llm_manager()

            # Determine schema path from selected schema
            if selected_schema:
                schema_path = self.config_dir / "schemas" / f"{selected_schema}.yaml"
            else:
                schema_path = self.schema_path  # Fallback to default

            # Validate schema exists
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")

            logger.info(f"Using schema: {schema_path.name}")

            # Determine concept source
            if selected_concept_file and selected_concept_file != "None":
                # Load from file
                concept_path = self.concepts_dir / selected_concept_file
                if not concept_path.exists():
                    raise FileNotFoundError(f"Concept file not found: {concept_path}")

                controller = InterviewController.from_concept_file(
                    concept_path=str(concept_path),
                    schema_path=str(schema_path),
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
                    schema_path=str(schema_path),
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

            # Create initial graph visualization
            graph_plot = self._create_plotly_graph(controller.graph)

            logger.info(f"Interview started: {self.current_session_id} for concept: {concept_name}")

            return (
                history,
                stats,
                self.current_session_id,
                concept_description,
                nodes_data,
                edges_data,
                graph_plot,
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
                self._create_plotly_graph(Graph()),
                {"error": str(e)}
            )

    async def process_response(
        self, user_response: str, history: List[Dict]
    ) -> Tuple[List[Dict], str, Dict, str, List[List], List[List], go.Figure, Dict]:
        """Process participant response and generate next question."""
        if not self.current_controller:
            return (
                history,
                "",
                {"nodes": 0, "edges": 0, "coverage": "0%", "turns": 0, "status": "error"},
                "No active session",
                [],
                [],
                self._create_plotly_graph(Graph()),
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
                self._create_plotly_graph(Graph()),
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

            # Create updated graph visualization
            graph_plot = self._create_plotly_graph(self.current_controller.graph)

            logger.info(f"Processed turn {self.current_controller.state.turn_count}")

            return (
                history,
                "",
                stats,
                self.current_session_id or "Active",
                nodes_data,
                edges_data,
                graph_plot,
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
                self._create_plotly_graph(Graph()),
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

    def _create_plotly_graph(self, graph: Graph) -> go.Figure:
        """Create an interactive Plotly visualization of the knowledge graph."""
        if not graph or len(graph.nodes) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No graph data yet. Start an interview to see the knowledge graph grow!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor="white",
                height=600
            )
            return fig

        # Convert Graph to NetworkX
        G = nx.DiGraph()

        # Add nodes with attributes
        for node in graph.nodes.values():
            G.add_node(
                node.id,
                label=node.label,
                node_type=node.node_type or "untyped",
                is_ambiguous=node.is_ambiguous,
                timestamp=node.timestamp
            )

        # Add edges
        for edge in graph.edges.values():
            G.add_edge(
                edge.source_id,
                edge.target_id,
                relation_type=edge.relation_type
            )

        # Calculate layout using spring layout
        try:
            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        except:
            # Fallback to simple circular layout if spring fails
            pos = nx.circular_layout(G)

        # Get node types for color mapping
        node_types = set(data.get('node_type', 'untyped') for _, data in G.nodes(data=True))
        color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        type_to_color = {t: color_palette[i % len(color_palette)] for i, t in enumerate(sorted(node_types))}

        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            source_id, target_id, edge_data = edge
            x0, y0 = pos[source_id]
            x1, y1 = pos[target_id]

            # Create edge line
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1.5, color='#888'),
                hoverinfo='text',
                text=f"{edge_data.get('relation_type', 'related')}",
                showlegend=False
            )
            edge_traces.append(edge_trace)

            # Add arrow annotation
            # Calculate arrow position (90% along the edge)
            arrow_x = x0 + 0.9 * (x1 - x0)
            arrow_y = y0 + 0.9 * (y1 - y0)

        # Create node trace
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_sizes = []
        node_border_colors = []
        node_border_widths = []

        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

            node_data = G.nodes[node_id]
            node_type = node_data.get('node_type', 'untyped')
            node_colors.append(type_to_color[node_type])

            # Node size based on degree (connections)
            degree = G.degree(node_id)
            node_sizes.append(20 + degree * 5)

            # Hover text
            label = node_data.get('label', node_id)
            connections = G.degree(node_id)
            hover_text = f"<b>{label}</b><br>"
            hover_text += f"Type: {node_type}<br>"
            hover_text += f"Connections: {connections}"
            if node_data.get('is_ambiguous'):
                hover_text += "<br><i>⚠️ Ambiguous</i>"
            node_text.append(hover_text)

            # Border for ambiguous nodes
            if node_data.get('is_ambiguous'):
                node_border_colors.append('red')
                node_border_widths.append(3)
            else:
                node_border_colors.append('#888')
                node_border_widths.append(1)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[G.nodes[n].get('label', n)[:15] for n in G.nodes()],
            textposition="top center",
            textfont=dict(size=9),
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(
                    color=node_border_colors,
                    width=node_border_widths
                )
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Knowledge Graph ({len(G.nodes)} nodes, {len(G.edges)} edges)",
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600,
            dragmode='pan'
        )

        # Add legend manually for node types
        legend_text = "<b>Node Types:</b><br>"
        for node_type in sorted(type_to_color.keys()):
            color = type_to_color[node_type]
            legend_text += f"<span style='color:{color}'>⬤</span> {node_type}<br>"

        fig.add_annotation(
            text=legend_text,
            xref="paper", yref="paper",
            x=1.02, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#888",
            borderwidth=1,
            font=dict(size=10)
        )

        return fig

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

    async def export_visualization_html(self) -> Optional[str]:
        """Export graph visualization as interactive HTML file."""
        try:
            if not self.current_controller or not self.current_session_id:
                logger.warning("Visualization export attempted with no active session")
                return None

            logger.info(f"Exporting visualization for session {self.current_session_id}")

            import tempfile

            # Create the plotly figure
            fig = self._create_plotly_graph(self.current_controller.graph)

            # Write to temporary HTML file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix="_graph_viz.html",
                delete=False,
                prefix=f"viz_{self.current_session_id}_",
            ) as temp_file:
                fig.write_html(temp_file.name)

            logger.info(f"Visualization export successful: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"Visualization export failed: {e}")
            return None

    async def export_graph_data_json(self) -> Optional[str]:
        """Export graph data (nodes and edges) as JSON file."""
        try:
            if not self.current_controller or not self.current_session_id:
                logger.warning("Graph data export attempted with no active session")
                return None

            logger.info(f"Exporting graph data for session {self.current_session_id}")

            import json
            import tempfile

            # Get the raw graph data
            graph = self.current_controller.graph

            # Prepare nodes data with full details
            nodes_export = []
            for node in graph.nodes.values():
                nodes_export.append({
                    "id": node.id,
                    "label": node.label,
                    "node_type": node.node_type,
                    "timestamp": node.timestamp.isoformat(),
                    "is_ambiguous": node.is_ambiguous,
                    "metadata": node.metadata
                })

            # Prepare edges data with full details
            edges_export = []
            for edge in graph.edges.values():
                source_node = graph.get_node(edge.source_id)
                target_node = graph.get_node(edge.target_id)

                edges_export.append({
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "source_label": source_node.label if source_node else edge.source_id[:8],
                    "target_label": target_node.label if target_node else edge.target_id[:8],
                    "relation_type": edge.relation_type,
                    "metadata": edge.metadata
                })

            # Create export data structure
            export_data = {
                "session_id": self.current_session_id,
                "export_timestamp": datetime.now().isoformat(),
                "graph_metadata": {
                    "total_nodes": len(graph.nodes),
                    "total_edges": len(graph.edges),
                    "concept_name": self.current_controller.config.concept_name
                },
                "nodes": nodes_export,
                "edges": edges_export
            }

            # Write to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix="_graph_data.json",
                delete=False,
                prefix=f"graph_{self.current_session_id}_",
            ) as temp_file:
                json.dump(export_data, temp_file, indent=2)

            logger.info(f"Graph data export successful: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"Graph data export failed: {e}")
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
            with gr.Column(scale=1):
                gr.Markdown("### Step 2: Select Methodology Schema")
                available_schemas = self.get_available_schemas()
                schema_dropdown = gr.Dropdown(
                    choices=available_schemas,
                    label="Interview Methodology",
                    value="means_end_chain" if "means_end_chain" in available_schemas else (available_schemas[0] if available_schemas else None),
                    info="Choose which methodology framework to use for the interview"
                )

        with gr.Row():
            start_btn = gr.Button("Start Interview", variant="primary", size="lg")
            status_text = gr.Textbox(
                label="Status",
                value="Ready to start",
                interactive=False,
                max_lines=1
            )

        return concept_input, concept_file_dropdown, schema_dropdown, start_btn, status_text

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
                download_graph_btn = gr.Button("📥 Download Graph Data (JSON)", size="sm")
                graph_data_file = gr.File(
                    label="Graph Data File",
                    visible=True,
                )
                graph_summary = gr.Textbox(
                    label="Graph Summary",
                    value="No graph data yet",
                    interactive=False,
                    lines=3
                )

        return nodes_table, edges_table, download_graph_btn, graph_data_file, graph_summary

    def _build_visualization_tab(self) -> Tuple:
        """Build interactive graph visualization tab."""
        with gr.TabItem("🔍 Graph Visualization"):
            gr.Markdown("### Interactive Knowledge Graph")
            gr.Markdown("Explore the interview knowledge graph with interactive visualization. Pan, zoom, and hover over nodes to see details.")

            # Main visualization
            graph_plot = gr.Plot(
                label="Knowledge Graph Visualization",
                value=self._create_plotly_graph(Graph())  # Start with empty graph
            )

            with gr.Row():
                refresh_viz_btn = gr.Button("🔄 Refresh Visualization", size="sm")
                download_viz_btn = gr.Button("📥 Download as HTML", size="sm")
                viz_file = gr.File(
                    label="Visualization File",
                    visible=True
                )

            # Info panel
            with gr.Accordion("ℹ️ Visualization Guide", open=False):
                gr.Markdown(
                    """
                    **How to use the visualization:**
                    - **Pan**: Click and drag to move around
                    - **Zoom**: Scroll to zoom in/out
                    - **Hover**: Move cursor over nodes to see details
                    - **Node Size**: Larger nodes have more connections
                    - **Node Color**: Different colors represent different node types
                    - **Red Border**: Indicates ambiguous nodes that need clarification

                    **Understanding the graph:**
                    - Nodes represent concepts mentioned in the interview
                    - Edges (lines) show relationships between concepts
                    - The layout uses force-directed positioning for natural clustering
                    - Hover over edges to see the relationship type
                    """
                )

        return graph_plot, refresh_viz_btn, download_viz_btn, viz_file

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
        start_btn, submit_btn, clear_btn, download_graph_btn,
        export_json_btn, export_transcript_btn,
        concept_input, concept_file_dropdown, schema_dropdown, user_input, chatbot,
        session_id_display, graph_stats, concept_display, status_text,
        nodes_table, edges_table, graph_data_file, graph_summary,
        graph_plot, refresh_viz_btn, download_viz_btn, viz_file,
        json_file, transcript_file
    ):
        """Wire all event handlers to UI components."""

        # Start interview
        start_btn.click(
            fn=self.start_interview_with_concept,
            inputs=[concept_input, concept_file_dropdown, schema_dropdown],
            outputs=[
                chatbot,
                graph_stats,
                session_id_display,
                concept_display,
                nodes_table,
                edges_table,
                graph_plot,
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
                graph_plot,
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
                graph_plot,
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
                self._create_plotly_graph(Graph()),  # graph_plot
                "Ready to start"  # status_text
            ),
            outputs=[chatbot, user_input, graph_stats, session_id_display, concept_display, nodes_table, edges_table, graph_plot, status_text],
        )

        # Download graph data
        download_graph_btn.click(
            fn=self.export_graph_data_json,
            outputs=[graph_data_file]
        )

        # Refresh visualization
        refresh_viz_btn.click(
            fn=lambda: self._create_plotly_graph(self.current_controller.graph if self.current_controller else Graph()),
            outputs=[graph_plot]
        )

        # Download visualization as HTML
        download_viz_btn.click(
            fn=self.export_visualization_html,
            outputs=[viz_file]
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

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface with event handlers wired inside Blocks context."""
        with gr.Blocks(title="AI Interview Assistant (NEW Architecture)") as app:
            # Build UI sections using helper methods
            self._build_header()
            concept_input, concept_file_dropdown, schema_dropdown, start_btn, status_text = self._build_concept_input_section()

            with gr.Tabs():
                # Interview tab
                (chatbot, user_input, submit_btn, clear_btn,
                 session_id_display, graph_stats, concept_display) = self._build_interview_tab()

                # Graph tab
                nodes_table, edges_table, download_graph_btn, graph_data_file, graph_summary = self._build_graph_tab()

                # Visualization tab
                graph_plot, refresh_viz_btn, download_viz_btn, viz_file = self._build_visualization_tab()

                # Export tab
                export_json_btn, json_file, export_transcript_btn, transcript_file = self._build_export_tab()

            # Wire event handlers INSIDE context
            self._wire_event_handlers(
                start_btn, submit_btn, clear_btn, download_graph_btn,
                export_json_btn, export_transcript_btn,
                concept_input, concept_file_dropdown, schema_dropdown, user_input, chatbot,
                session_id_display, graph_stats, concept_display, status_text,
                nodes_table, edges_table, graph_data_file, graph_summary,
                graph_plot, refresh_viz_btn, download_viz_btn, viz_file,
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