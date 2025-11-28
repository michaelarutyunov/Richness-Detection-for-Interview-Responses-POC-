"""
AI Interview System - HuggingFace Space Entry Point

This is the main entry point for the Gradio application.
Compatible with HuggingFace Spaces deployment.
"""

import os
import logging
from src.ui.gradio_app import launch_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verify environment variables are set (for HF Spaces)
def check_environment():
    """Check that required environment variables are configured."""
    required_vars = ["KIMI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.warning(
            f"Missing environment variables: {', '.join(missing_vars)}\n"
            f"Interview will run in demo mode only.\n"
            f"For full functionality, configure secrets in HuggingFace Space settings."
        )
    else:
        logger.info("All required environment variables are set ✓")

if __name__ == "__main__":
    check_environment()

    logger.info("Starting AI Interview System...")

    # Launch Gradio app
    # Uses HuggingFace Space defaults: 0.0.0.0:7860
    launch_app(
        share=False,  # HF Spaces handles public access
        server_name="0.0.0.0",
        server_port=7860
    )
