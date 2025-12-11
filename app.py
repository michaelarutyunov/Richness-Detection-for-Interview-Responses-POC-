#!/usr/bin/env python3
"""
Launcher for the NEW Gradio UI.
Simple entry point for the new interview system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ui.gradio_app import launch_app


if __name__ == "__main__":
    print("🚀 Launching NEW Gradio Interview UI...")
    print("Using NEW architecture with InterviewController")
    print("Config directory: src/config/")
    print("")
    
    launch_app(share=False, server_name="0.0.0.0", server_port=7860)