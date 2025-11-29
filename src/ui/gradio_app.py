"""
Gradio interface for the AI Interview System.

Provides a web-based chat interface for conducting interviews.
Compatible with HuggingFace Spaces deployment.
"""

import logging

import gradio as gr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewUI:
    """Gradio-based user interface for AI interviews."""

    def __init__(self):
        """Initialize the interview UI."""
        self.conversation_history: list[tuple[str, str]] = []
        self.turn_number: int = 0
        self.session_id: str | None = None

        # Will be wired up to Interview Controller in Phase 4
        self.controller = None

        logger.info("InterviewUI initialized")

    def start_interview(self) -> str:
        """
        Initialize a new interview session.

        Returns:
            First question to display to participant
        """
        self.turn_number = 0
        self.conversation_history = []

        # TODO: Initialize Interview Controller in Phase 4
        # self.controller = InterviewController(schema_path=...)

        first_question = (
            "Welcome! Thank you for participating in this interview. "
            "I'm here to understand your thoughts and experiences with this product. "
            "There are no right or wrong answers - I'm just curious about your perspective.\n\n"
            "Let's start with: What's your initial reaction to this product?"
        )

        logger.info("Interview started")
        return first_question

    def process_response(self, user_response: str, history: list) -> tuple[list, str]:
        """
        Process participant response and generate next question.

        Args:
            user_response: What the participant said
            history: Gradio chat history

        Returns:
            Tuple of (updated_history, cleared_input)
        """
        if not user_response or not user_response.strip():
            # Empty response - don't process
            return history, ""

        self.turn_number += 1
        logger.info(f"Processing turn {self.turn_number}")

        # TODO: Phase 4 - Replace with actual Interview Controller
        # graph_delta = self.controller.process_response(user_response)
        # next_question = self.controller.generate_next_question()

        # PLACEHOLDER: Echo + generic follow-up
        if self.turn_number < 5:
            next_question = self._generate_placeholder_question(user_response)
        else:
            next_question = (
                "Thank you for sharing all of that! "
                "Is there anything else you'd like to add that we haven't covered?"
            )

        # Update conversation history
        history.append({"role": "user", "content": user_response})
        history.append({"role": "assistant", "content": next_question})

        logger.info(f"Turn {self.turn_number} completed")

        return history, ""  # Return updated history and clear input

    def _generate_placeholder_question(self, response: str) -> str:
        """
        Generate a simple placeholder question (Phase 0 only).

        Args:
            response: Participant's response

        Returns:
            Next question
        """
        templates = [
            "That's interesting. Can you tell me more about that?",
            "What specifically stands out to you?",
            "How does that make you feel?",
            "What else comes to mind when you think about that?",
            "Could you elaborate on that a bit?",
        ]

        # Rotate through templates based on turn number
        return templates[(self.turn_number - 1) % len(templates)]

    def build_interface(self) -> gr.Blocks:
        """
        Build the Gradio interface.

        Returns:
            Gradio Blocks app
        """
        # Note: In Gradio 6.x with Python 3.13, gr.Blocks() has limited parameters
        # Theme and CSS can be applied via .launch() or other methods
        with gr.Blocks(title="AI Interview Assistant") as app:
            gr.Markdown(
                """
                # 🎙️ AI Interview Assistant
                **Graph-driven adaptive interviewing for concept testing**

                This system uses AI to conduct natural, conversational interviews
                while building a knowledge graph of your mental model.
                """
            )

            with gr.Row():
                # Left column: Chat interface
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Interview Conversation",
                        height=500,
                        autoscroll=True,
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
                        clear_btn = gr.Button("Clear", size="lg")

                # Right column: Interview metadata & stats
                with gr.Column(scale=1):
                    gr.Markdown("### Interview Progress")

                    turn_display = gr.Number(label="Current Turn", value=0, interactive=False)

                    gr.Markdown("### Graph Statistics")
                    graph_stats = gr.JSON(
                        label="Knowledge Graph",
                        value={
                            "nodes": 0,
                            "edges": 0,
                            "coverage": "0%",
                        },
                    )

                    gr.Markdown("### Session Info")
                    _ = gr.Textbox(
                        label="Session ID", value="Not started", interactive=False, max_lines=1
                    )

            # Instructions
            with gr.Accordion("ℹ️ How to use", open=False):
                gr.Markdown(
                    """
                    **Instructions:**
                    1. Read the question presented by the AI interviewer
                    2. Type your response in the text box
                    3. Click **Submit** to send your answer
                    4. The AI will analyze your response and ask a follow-up question
                    5. Continue the conversation naturally - there are no wrong answers!

                    **Tips:**
                    - Be as detailed or brief as you like
                    - Feel free to ask for clarification
                    - If you're unsure, say so - that's valuable feedback too
                    """
                )

            # Event handlers
            def update_turn_number(history):
                """Update turn counter based on history length."""
                return len(history) // 2  # Divide by 2 since each turn has user + assistant

            submit_btn.click(
                fn=self.process_response,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input],
            ).then(fn=update_turn_number, inputs=[chatbot], outputs=[turn_display])

            # Allow Enter key to submit (only works when input is focused)
            user_input.submit(
                fn=self.process_response,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input],
            ).then(fn=update_turn_number, inputs=[chatbot], outputs=[turn_display])

            clear_btn.click(
                fn=lambda: ([], "", 0, {"nodes": 0, "edges": 0, "coverage": "0%"}),
                outputs=[chatbot, user_input, turn_display, graph_stats],
            )

            # Initialize with first question on load
            app.load(fn=self.start_interview, outputs=None)

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
