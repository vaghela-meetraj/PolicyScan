# PolicyScan

An Agentic Retrieval-Augmented Generation (RAG) assistant for querying and analyzing insurance policy documents.

Powered by LangChain, Groq (`openai/gpt-oss-120b`), Jina AI Embeddings, FAISS, and Streamlit.

---

## Features

- **Modular Architecture:** Organized modules under `insurance_assistant/` handling document loading, text splitting, embeddings, vector indexing, agent execution, tracing, and logging.
- **Persistent Vector Caching:** FAISS index is generated and saved locally to disk (`data/faiss_index/`), avoiding re-embedding on subsequent runs.
- **Tool-Assisted Retrieval:** LangChain agent equipped with a `search_insurance_policy` tool to query the FAISS index before answering policy-related questions.
- **Context-Constrained Prompting:** System prompt instructs the model to answer based on retrieved document context and state when requested information is absent.
- **LangSmith Tracing:** Integrated tracing configuration for tracking end-to-end request latency, tool execution, and token usage via LangSmith.
- **Day-Wise Structured Logging:** Centralized logging module configured to write runtime events simultaneously to the console and to date-organized directories (`logs/YYYY-MM-DD/run_*.log`).
- **Streamlit Web UI:** Chat interface with SVG icons, user/assistant message alignment, and sample query buttons.
- **CLI Runner:** Script interface for running evaluation questions from the terminal via `main.py`.

---

## Project Structure

```text
PolicyScan/
├── app.py                     # Streamlit Web UI application
├── main.py                    # CLI entry point for batch testing
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git exclusion rules
├── assets/                    # UI vector icons (assistant & user SVGs)
│   ├── assistant.svg
│   └── user.svg
├── data/                      # Policy documents & local vector stores
│   ├── apex_health_insurance.txt
│   └── policy.txt
├── docs/                      # Technical documentation & engineering guides
│   ├── 01 Logger.md           # In-depth guide to production logging architecture
│   └── 02 Tracing.md          # Guide to LLM tracing, observability, and LangSmith
└── insurance_assistant/       # Core modular RAG package
    ├── __init__.py            # Package initialization
    ├── config.py              # Centralized settings & prompt configuration
    ├── document_loader.py     # Document ingestion (TextLoader)
    ├── splitter.py            # Recursive character chunking
    ├── embeddings.py          # Jina AI embedding model wrapper
    ├── vector_store.py        # FAISS build, persistence, and loading
    ├── tools.py               # LangChain search tool factory
    ├── llm.py                 # ChatGroq LLM model initialization
    ├── agent.py               # LangChain ReAct agent creator
    ├── logger.py              # Day-wise logging setup (File & Stream handlers)
    ├── tracing.py             # LangSmith tracing status validation
    └── pipeline.py            # End-to-end RAG orchestrator
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/vaghela-meetraj/PolicyScan.git
cd PolicyScan
```

### 2. Set Up Virtual Environment
```powershell
# Using uv (fast)
uv venv virtenv
.\virtenv\Scripts\activate

# Or using standard python
python -m venv virtenv
.\virtenv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root based on `.env.example`:
```env
# LLM & Embedding API Keys
GROQ_API_KEY=your_groq_api_key_here
JINA_API_KEY=your_jina_api_key_here

# Optional: LangSmith Tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT="PolicyScan"
```

---

## Usage

### Run the Streamlit Web UI
Launch the interactive chat interface in your browser:
```powershell
streamlit run app.py
```
Access the interface at `http://localhost:8501`.

### Run the CLI Batch Runner
Run demo queries in the terminal:
```powershell
python main.py
```

---

## Documentation

- **Logging Guide:** For architectural details on handlers, formatters, log levels, and production practices, refer to [01 Logger.md](docs/01%20Logger.md).
- **Tracing Guide:** For details on end-to-end request journeys, spans, token consumption, and LangSmith integration, refer to [02 Tracing.md](docs/02%20Tracing.md).

---

## Tech Stack

- **Framework:** [LangChain](https://www.langchain.com/)
- **LLM Engine:** [Groq](https://groq.com/) (`openai/gpt-oss-120b`)
- **Embeddings:** [Jina AI](https://jina.ai/) (`jina-embeddings-v2-base-en`)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
- **Observability:** [LangSmith](https://smith.langchain.com/)
- **Web UI:** [Streamlit](https://streamlit.io/)
- **Environment Management:** [uv](https://github.com/astral-sh/uv) / `venv`
