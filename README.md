---
title: AI Interview Assistant
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
python_version: 3.11
---

# 🎙️ AI-Based Graph Interviewer - Minimal POC

**Graph-driven adaptive interviewing system for FMCG concept testing**

This system uses AI to conduct natural, conversational interviews while constructing a knowledge graph that represents the participant's mental model. Unlike traditional scripted interviews, the AI adapts its questions based on the emerging graph structure, ensuring comprehensive coverage while maintaining conversational flow.

## ✨ Features

- **Schema-driven knowledge graph construction** - Flexible YAML-based mental model definitions (now v0.2 with enhanced extraction prompts)
- **Advanced extraction prompts system** - New LLM prompts with confidence scoring and quote validation
- **Dual LLM architecture** - Kimi K2 for fast extraction + Claude Sonnet for natural questions
- **Real-time adaptive questioning** - Dynamically selects next question based on graph state
- **Gradio 6 interface** - Clean, professional web UI
- **YAML-based configuration** - Easy experimentation with different interview strategies and extraction prompts
- **Minimal POC scope** - Phases 0-4 implemented for rapid validation

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **uv** (recommended) or pip
- API keys for:
  - **Kimi** (Moonshot AI) - [Get key](https://platform.moonshot.cn/)
  - **Anthropic Claude** - [Get key](https://console.anthropic.com/)

### Installation with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/YOUR_USERNAME/Richness-Detection-for-Interview-Responses-POC-.git
cd Richness-Detection-for-Interview-Responses-POC-

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Or use uv sync for pyproject.toml
uv sync
```

### Installation with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Create `.env` file**:
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys** to `.env`:
   ```bash
   KIMI_API_KEY=your_moonshot_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here

   DEFAULT_SCHEMA=schemas/means_end_chain_v0.2.yaml
   DEFAULT_MODEL_CONFIG=configs/model_config.yaml
   LOG_LEVEL=INFO
   ```

### Run Locally

```bash
python app.py
```

Open your browser to `http://localhost:7860`

## 📁 Project Structure

```
/
├── app.py                          # Entry point (HF Space compatible)
├── requirements.txt                # Dependencies (for HF Spaces)
├── pyproject.toml                  # uv/pip configuration
├── .env.example                    # Environment template
├── .gitignore                      # Git exclusions
│
├── src/
│   ├── core/
│   │   ├── schema_manager.py       # [Phase 1] Schema loading & validation
│   │   ├── interview_graph.py      # [Phase 1] NetworkX graph operations
│   │   └── data_models.py          # ✅ Pydantic models
│   ├── interview/
│   │   ├── controller.py           # [Phase 4] Main orchestrator
│   │   ├── manager.py              # [Phase 3] Opportunity ranking
│   │   ├── response_processor.py   # [Phase 2] LLM extraction
│   │   └── question_generator.py   # [Phase 3] Question creation
│   ├── llm/
│   │   ├── client_factory.py       # [Phase 2] LLM client abstraction
│   │   ├── kimi_client.py          # [Phase 2] Moonshot/Kimi integration
│   │   └── anthropic_client.py     # [Phase 2] Anthropic integration
│   └── ui/
│       └── gradio_app.py           # ✅ Gradio interface
│
├── schemas/
│   ├── means_end_chain_v0.2.yaml   # ✅ Means-End Chain schema (v0.2)
│   └── usage_script_v0.1.yaml      # ✅ Usage Script schema
│
├── configs/
│   ├── default_interview.yaml      # ✅ Interview configuration
│   └── model_config.yaml           # ✅ LLM configuration
│
├── prompts/
│   ├── extraction_prompts.yaml     # ✅ Enhanced v0.2 extraction prompts with confidence scoring
│   └── question_templates.yaml     # ✅ Question templates
│
├── tests/
│   ├── fixtures/
│   │   └── sample_responses.json   # ✅ Test data
│   ├── test_schema_manager.py      # [Phase 1]
│   └── test_interview_graph.py     # [Phase 1]
│
└── data/                           # (Not committed - local only)
    └── interviews/                 # Interview logs
```

✅ = Completed (Phase 0)
[Phase X] = To be implemented

## 🧪 Schema Types

### 1. Means-End Chain (`means_end_chain_v0.2.yaml`)

Models consumer thinking as a vertical ladder of abstraction:

**Attributes** → **Functional Consequences** → **Psychosocial Consequences** → **Values**

**Best for:** Understanding WHY consumers value features, motivational drivers, positioning/messaging strategy.

**Example:**
- Attribute: "affordable_price"
- → Functional: "regular_purchase"
- → Psychosocial: "financial_peace_of_mind"
- → Value: "family_wellbeing"

### 2. Usage Script (`usage_script_v0.1.yaml`)

Models consumer experience as a temporal script:

**Triggers** → **Actions** → **Outcomes** → **Emotions** (in **Settings**)

**Best for:** Understanding WHEN, WHERE, and HOW the product fits into lives, contextual marketing.

**Example:**
- Trigger: "running_late"
- → Action: "grabbed_from_fridge"
- → Outcome: "hunger_satisfied"
- → Emotion: "relief"

## 🔄 Schema v0.2 Migration & Extraction Prompts

### What's New in v0.2

The system has migrated from **schema v0.1 to v0.2**, representing a major enhancement in how interview responses are processed:

**Key Improvements:**
- **Enhanced extraction prompts** - New LLM prompts in `prompts/extraction_prompts.yaml` provide more accurate concept and relationship extraction
- **Structured extraction system** - Confidence-based extraction with direct quote support
- **Schema-driven extraction** - Extraction behavior is now fully controlled by schema definitions
- **Better node merging** - Intelligent duplicate detection and concept consolidation

### New Extraction Prompts System

Located in `prompts/extraction_prompts.yaml`, the new system provides:

- **System prompts** - Define extraction behavior and rules for LLMs
- **User prompt templates** - Context-aware prompts that include schema definitions
- **Function calling schema** - Structured output format for reliable extraction
- **Confidence scoring** - Edge confidence levels (0.6-1.0) for quality control

### Migration Benefits

- **More accurate extractions** - Better alignment with schema definitions
- **Reduced hallucination** - Strict schema adherence prevents invalid extractions  
- **Improved consistency** - Standardized extraction across different LLM providers
- **Better quote support** - Every extraction includes supporting evidence

## 🎯 Implementation Phases

### ✅ **Phase 0: Foundation** (COMPLETED)
- Project structure
- Enhanced schema YAMLs
- Pydantic data models
- LLM prompt templates
- Gradio interface skeleton
- Configuration files

### 🚧 **Phase 1: Core Infrastructure** (Week 1-2)
- Schema Manager (load & validate YAML)
- Interview Graph (NetworkX wrapper)
- Unit tests

### 🔮 **Phase 2: Extraction Pipeline** (Week 3-4)
- Response Processor (LLM extraction)
- LLM client factory (Kimi + Claude)
- Integration tests

### 🔮 **Phase 3: Interview Logic** (Week 5)
- Interview Manager (opportunity ranking)
- Question Generator (template + LLM)

### 🔮 **Phase 4: Integration & UI** (Week 6)
- Interview Controller (turn loop)
- Wire up Gradio interface
- End-to-end testing

## 🚢 Deployment to HuggingFace Spaces

### Pre-Deployment Checklist

- ✅ `app.py` runs locally without errors
- ✅ `requirements.txt` has all dependencies with versions
- ✅ README has HF frontmatter (already included above)
- ✅ No `.env` file committed (only `.env.example`)
- ✅ Code reads from `os.environ` (not .env file)
- ✅ Gradio launches with `server_name="0.0.0.0"`, `server_port=7860`

### Deployment Steps

1. **Create HuggingFace Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select Gradio SDK, Python 3.11

2. **Clone Space locally**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
   cd SPACE_NAME
   ```

3. **Copy files from this repo**
   ```bash
   # Copy essential files (from your POC directory)
   cp -r app.py requirements.txt README.md src/ schemas/ configs/ prompts/ ../SPACE_NAME/
   ```

4. **Configure Secrets in HF Space UI**
   - Go to Space Settings → Repository secrets
   - Add:
     - `KIMI_API_KEY` = your_moonshot_api_key
     - `ANTHROPIC_API_KEY` = your_anthropic_api_key

5. **Commit and push**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push
   ```

6. **Verify deployment**
   - HuggingFace will automatically build and deploy
   - Check Space logs for errors
   - Test the interface

### Files to Copy vs. Exclude

**✅ COPY to HF Space:**
- `app.py`
- `requirements.txt`
- `README.md`
- `src/`
- `schemas/`
- `configs/`
- `prompts/`

**❌ DO NOT COPY:**
- `.env` (use HF Secrets instead)
- `.env.example` (optional documentation)
- `data/` (local interview logs)
- `notebooks/` (local experimentation)
- `tests/` (optional)
- `context/` (design docs - optional)
- `.git/` (HF Space has its own git)

### Logs on HuggingFace Spaces

When deployed to HuggingFace Spaces, logs are automatically written to `/tmp/logs/` due to the read-only filesystem:

- **Local development**: Logs saved to `src/logs/`
- **HuggingFace Spaces**: Logs saved to `/tmp/logs/`

Note: Logs on HF Spaces are ephemeral and cleared on Space restart.

**Environment Variables**:
- `SPACE_ID`: Auto-set by HuggingFace Spaces (do not set manually)

## 🧑‍💻 Development

### Running Tests

```bash
# With uv
uv run pytest

# With pip
pytest
```

### Code Formatting

```bash
# Format with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/
```

### Using uv for Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Add a new dependency
uv add package-name

# Update dependencies
uv pip compile pyproject.toml -o requirements.txt
```

## 📊 Model Configuration

### Dual LLM Architecture

**Graph Processing (Fast Extraction)**:
- Provider: Moonshot AI (Kimi)
- Model: `moonshot-v1-32k`
- Temperature: 0.3 (consistent extraction)
- Fallback: Claude Haiku, GPT-4o-mini

**Question Generation (Natural Language)**:
- Provider: Anthropic
- Model: `claude-3-5-sonnet-20241022`
- Temperature: 0.7 (varied questions)
- Fallback: Claude Haiku

Configure in `configs/model_config.yaml`.

## 🐛 Troubleshooting

### "Missing environment variables" warning
- Create `.env` file with your API keys (local dev)
- Or configure secrets in HF Space settings (deployment)

### Gradio won't launch
- Check port 7860 isn't already in use
- Verify all dependencies installed: `uv pip list` or `pip list`

### LLM API errors
- Verify API keys are valid
- Check API credit balance
- Review rate limits in `configs/model_config.yaml`

### Import errors
- Ensure you're in the virtual environment
- Run `uv sync` or `pip install -r requirements.txt` again

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Design inspired by **Means-End Chain Theory** (Reynolds & Gutman, 1988)
- Built with **Gradio 6**, **NetworkX**, and **Pydantic**
- LLM integration via **Anthropic Claude** and **Moonshot Kimi**

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Status**: Phase 0 Complete ✅ | Ready for Phase 1 Implementation 🚀
