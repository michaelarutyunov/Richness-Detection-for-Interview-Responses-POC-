# NEW Gradio UI - AI Interview Assistant

This is the NEW Gradio interface for the AI Interview System, built using the simplified InterviewController architecture.

## Key Features

- **NEW Architecture**: Uses the simplified `InterviewController` instead of the complex legacy orchestrator
- **Unified Configuration**: Single config system with `llm_config.yaml` and `interview_config.yaml`
- **Real-time Graph Visualization**: Node and edge tables showing the knowledge graph as it builds
- **Multiple Input Methods**: Text concepts or load from concept files
- **Export Functionality**: JSON (complete session data) and transcript (text format)
- **Clear Error Messages**: Helpful feedback when setup is incomplete

## Architecture Changes

### Legacy vs NEW System

| Aspect | Legacy System | NEW System |
|--------|---------------|------------|
| Orchestrator | `ConfigurableGraphDrivenOrchestrator` | `InterviewController` |
| State Management | Manual graph state tracking | Integrated `GraphState.compute()` |
| LLM Management | Complex dual LLM setup | Simplified `LLMManager` |
| Configuration | Multiple config files | Unified config system |
| Concept Parsing | Manual/heuristic | LLM-powered parsing |
| Coverage Tracking | Manual implementation | Built-in `CoverageState` |

## Setup

1. **Configuration Files**: Ensure these files exist in `src/config/`:
   - `llm_config.yaml` - LLM provider configuration
   - `interview_config.yaml` - Interview flow settings
   - `interview_logic.yaml` - Strategy selection logic
   - `schemas/means_end_chain.yaml` - Methodology schema
   - `concepts/*.md` - Concept files (optional)

2. **API Keys**: Set environment variables in `.env`:
   ```bash
   ANTHROPIC_API_KEY=your_key_here
   # or
   OPENAI_API_KEY=your_key_here
   # or
   KIMI_API_KEY=your_key_here
   ```

3. **Launch**: Run the Gradio app:
   ```bash
   python launch_new_gradio.py
   # or
   python -m ui.gradio_app
   ```

## Usage

1. **Start Interview**:
   - Enter a concept description in the text box, OR
   - Select a concept file from the dropdown
   - Click "Start Interview"

2. **Conduct Interview**:
   - Read the AI's opening question
   - Type your response and click "Submit"
   - Continue the conversation (typically 10-15 turns)

3. **Monitor Progress**:
   - Watch graph stats update in real-time
   - View nodes and edges in the Knowledge Graph tab
   - Track coverage progress

4. **Export Results**:
   - Download complete session data as JSON
   - Download conversation transcript as text

## UI Components

### Main Tabs

- **💬 Interview**: Chat interface with conversation history
- **📊 Knowledge Graph**: Node/edge tables showing extracted concepts
- **💾 Export**: Download session data and transcripts

### Key Features

- **Concept Input**: Text area or file selection
- **Real-time Stats**: Nodes, edges, coverage, turns
- **Graph Tables**: Live updating node/edge data
- **Export Options**: JSON and transcript formats
- **Error Handling**: Clear messages for missing setup

## Testing

Run the comprehensive test suite:
```bash
python test_complete_ui.py
```

This tests:
- Interview initialization
- Multi-turn processing
- Graph data updates
- Export functionality
- Error handling
- Concept file loading

## Error Messages

The UI provides clear error messages for common issues:

- **No API Keys**: "No LLM providers available - check API keys in .env"
- **Missing Configs**: "Configuration file not found: [path]"
- **No Concept**: "Please provide a concept description or select a concept file"
- **Export Without Session**: Graceful degradation with None return

## Development

The main UI class is `InterviewUI` in `gradio_app.py`. Key methods:

- `setup_llm_manager()`: Initialize LLM configuration
- `start_interview_with_concept()`: Begin new interview
- `process_response()`: Handle user input and generate next question
- `export_json_file()`: Export complete session data
- `export_transcript_file()`: Export conversation transcript

## Comparison with Legacy System

### Improvements
- **Simpler Architecture**: Single controller vs complex orchestrator
- **Better State Management**: Integrated state computation
- **Cleaner Configuration**: Unified config files
- **Improved Error Handling**: Clear, actionable error messages
- **Better Concept Parsing**: LLM-powered vs heuristic

### Maintained Features
- Graph-driven question generation
- Real-time knowledge graph building
- Coverage tracking
- Export functionality
- Gradio web interface

## Future Enhancements

Potential improvements for the NEW system:
- Graph visualization with charts/plots
- Advanced analytics dashboard
- Multi-language support
- Custom schema support
- Interview templates
- Advanced export formats