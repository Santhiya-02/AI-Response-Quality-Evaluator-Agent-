# AI Response Quality Evaluator Dashboard

A production-ready web application for evaluating the quality of AI-generated responses using semantic retrieval, reference knowledge bases, and LLM-based judge agents.

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to **`http://localhost:8000`** to view the interactive dashboard.

---

## Command-Line Verification Suite

To run the pipeline checks, LLM judges, and consistency validation suite directly in the console, execute:
```bash
python m2_pipeline_test.py
```

---

## Project Structure

```
├── main.py                   # FastAPI REST entry point & Static File Server
├── requirements.txt          # Python packages list
├── .env                      # API keys & configuration
├── m2_pipeline_test.py       # Console-based verification pipeline
├── src/                      # Evaluation business logic
│   ├── input_module.py       # Input validation & document parsers
│   ├── knowledge_base.py     # ChromaDB vector index & embeddings
│   ├── benchmark_loader.py   # TriviaQA / SQuAD loaders
│   ├── scoring.py            # Baseline similarity scoring
│   ├── judges.py             # LLM Relevance, Accuracy, & Hallucination judges
│   └── validation.py         # Statistical consistency validation engine
└── frontend/                 # React SPA (Vite + TypeScript)
    ├── src/
    │   ├── App.tsx           # Dashboard tabs & visual interfaces
    │   └── index.css         # Glassmorphic dark styling sheet
    └── dist/                 # Production-compiled client static files
```
