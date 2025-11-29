"""
Gradio interface for the AI Interview System.

Provides a web-based chat interface for conducting interviews.
Compatible with HuggingFace Spaces deployment.
"""

import logging
import os
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

from src.core.schema_manager import SchemaManager
from src.interview.concept_extractor import ConceptExtractor
from src.interview.interview_manager import InterviewManager
from src.interview.prompt_builder import PromptBuilder
from src.interview.validator import Validator
from src.llm.anthropic_client import AnthropicClient
from src.llm.kimi_client import KimiClient

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewSession:
    """Manages a single interview session with state."""

    def __init__(self, schema_path: str, concept_description: str):
        """Initialize interview session."""
        self.concept_description = concept_description
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load schema
        self.schema = SchemaManager(schema_path)
        self.schema.load_schema()
        self.schema.validate_schema()

        # Create LLM clients
        self.extraction_client = KimiClient(
            api_key=os.getenv("KIMI_API_KEY", ""),
            model="moonshot-v1-32k",
            temperature=0.3,
            max_tokens=2000,
            timeout=30,
        )

        self.question_client = AnthropicClient(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=300,
            timeout=20,
        )

        # Create concept extractor
        self.concept_extractor = ConceptExtractor(
            llm_client=self.extraction_client,
            prompt_builder=PromptBuilder(),
            validator=Validator(self.schema),
        )

        # Interview manager will be initialized after seed extraction
        self.manager: InterviewManager | None = None
        self.seeds_extracted = False

        logger.info(
            f"Session {self.session_id} initialized for concept: {concept_description[:50]}..."
        )

    async def initialize(self):
        """Extract seed nodes and initialize interview manager."""
        if self.seeds_extracted:
            return

        logger.info("Extracting seed nodes from concept...")

        # Extract seed nodes
        delta = await self.concept_extractor.extract_seed_nodes(self.concept_description)

        # Create interview manager
        self.manager = InterviewManager(
            schema_manager=self.schema,
            extraction_client=self.extraction_client,
            question_client=self.question_client,
            min_richness=8.0,  # Lower threshold for testing
            max_turns=15,
        )

        # Apply seed nodes to graph
        if delta.nodes_added:
            nodes_added, _ = self.manager.graph.apply_delta(delta, turn_number=0)
            logger.info(f"Added {nodes_added} seed nodes to graph")
        else:
            logger.warning("No seed nodes extracted from concept")

        self.seeds_extracted = True

    async def start(self) -> str:
        """Start the interview."""
        await self.initialize()
        question = await self.manager.start_interview()
        return question

    async def process_response(self, user_response: str) -> str:
        """Process user response and get next question."""
        if not self.manager:
            return "Error: Interview not initialized"

        next_question = await self.manager.process_response(user_response)
        return next_question

    def get_stats(self) -> dict:
        """Get current interview statistics."""
        if not self.manager:
            return {"nodes": 0, "edges": 0, "coverage": "0%", "richness": 0.0, "turns": 0}

        summary = self.manager.get_summary()
        return {
            "nodes": summary["nodes"],
            "edges": summary["edges"],
            "coverage": f"{summary['coverage'] * 100:.1f}%",
            "richness": round(summary["richness"], 2),
            "turns": summary["turns"],
        }

    def is_complete(self) -> bool:
        """Check if interview should end."""
        if not self.manager:
            return False
        return not self.manager.should_continue()

    def export_graph(self, path: str):
        """Export interview graph."""
        if self.manager:
            self.manager.export_graph(path)


class InterviewUI:
    """Gradio-based user interface for AI interviews."""

    def __init__(self):
        """Initialize the interview UI."""
        self.current_session: InterviewSession | None = None
        logger.info("InterviewUI initialized")

    async def start_interview_with_concept(
        self, concept_description: str
    ) -> tuple[list, dict, str]:
        """Start new interview with concept description."""
        if not concept_description or not concept_description.strip():
            error_msg = [
                {"role": "assistant", "content": "Please provide a concept description first."}
            ]
            return (
                error_msg,
                {"nodes": 0, "edges": 0, "coverage": "0%", "richness": 0.0, "turns": 0},
                "Error: No concept",
            )

        # Create new session
        self.current_session = InterviewSession(
            schema_path="schemas/means_end_chain_v0.1.yaml",
            concept_description=concept_description,
        )

        # Start interview
        try:
            first_question = await self.current_session.start()

            # Build initial history
            history = [{"role": "assistant", "content": first_question}]
            stats = self.current_session.get_stats()
            session_id = self.current_session.session_id

            logger.info(f"Interview started: {session_id}")
            return history, stats, session_id
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
                {"nodes": 0, "edges": 0, "coverage": "0%", "richness": 0.0, "turns": 0},
                "Error",
            )

    async def process_response(
        self, user_response: str, history: list
    ) -> tuple[list, str, dict, str]:
        """Process participant response and generate next question."""
        if not self.current_session:
            return (
                history,
                "",
                {"nodes": 0, "edges": 0, "coverage": "0%", "richness": 0.0, "turns": 0},
                "No active session",
            )

        if not user_response or not user_response.strip():
            return history, "", self.current_session.get_stats(), self.current_session.session_id

        try:
            # Process response
            next_question = await self.current_session.process_response(user_response)

            # Update history
            history.append({"role": "user", "content": user_response})
            history.append({"role": "assistant", "content": next_question})

            # Get updated stats
            stats = self.current_session.get_stats()

            # Check if complete
            if self.current_session.is_complete():
                completion_msg = "\n\n✅ **Interview Complete!** Thank you for your time."
                history[-1]["content"] += completion_msg

            return history, "", stats, self.current_session.session_id
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            history.append({"role": "user", "content": user_response})
            history.append({"role": "assistant", "content": error_msg})
            return history, "", self.current_session.get_stats(), self.current_session.session_id

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface."""
        with gr.Blocks(title="AI Interview Assistant") as app:
            gr.Markdown(
                """
                # 🎙️ AI Interview Assistant
                **Graph-driven adaptive interviewing for concept testing**

                This system uses AI to conduct natural, conversational interviews
                while building a knowledge graph of your mental model.
                """
            )

            # Concept input section
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
                        label="Session ID", value="Not started", interactive=False, max_lines=1
                    )

                    gr.Markdown("### Knowledge Graph Stats")
                    graph_stats = gr.JSON(
                        label="Current Graph",
                        value={
                            "nodes": 0,
                            "edges": 0,
                            "coverage": "0%",
                            "richness": 0.0,
                            "turns": 0,
                        },
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

                    **Requirements:**
                    - KIMI_API_KEY and ANTHROPIC_API_KEY must be set in .env file
                    """
                )

            # Event handlers
            start_btn.click(
                fn=self.start_interview_with_concept,
                inputs=[concept_input],
                outputs=[chatbot, graph_stats, session_id_display],
            )

            submit_btn.click(
                fn=self.process_response,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input, graph_stats, session_id_display],
            )

            user_input.submit(
                fn=self.process_response,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input, graph_stats, session_id_display],
            )

            clear_btn.click(
                fn=lambda: (
                    [],
                    "",
                    {"nodes": 0, "edges": 0, "coverage": "0%", "richness": 0.0, "turns": 0},
                    "Not started",
                ),
                outputs=[chatbot, user_input, graph_stats, session_id_display],
            )

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

    logger.info(f"Launching app on {server_name}:{server_port}")

    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    launch_app()
