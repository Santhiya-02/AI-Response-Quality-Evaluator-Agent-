# AI Response Quality Evaluator Agent — Milestone 1

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## Project Structure

```
├── app.py                    # Streamlit UI (main entry point)
├── requirements.txt
├── .env                      # API keys (optional for M1)
├── architecture.md           # Full architecture documentation
├── chroma_db/                # Persistent vector database
└── src/
    ├── input_module.py       # Evaluation Input Module
    ├── knowledge_base.py     # Chunking + Embedding + ChromaDB
    ├── benchmark_loader.py   # TriviaQA / SQuAD dataset loader
    └── scoring.py            # Baseline scoring metrics
```

## Milestone 1 Demo Flow

1. **Load Knowledge Base** — Use the sidebar to load TriviaQA or SQuAD, or upload your own PDF/TXT/MD files
2. **Evaluate a Response** — Enter a question + AI response (+ optional reference answer) and click Evaluate
3. **Retrieval Demo** — Go to the Retrieval Demo tab and search the knowledge base with any query
4. **Architecture** — View the full system architecture, agent responsibilities, and scoring design

## Milestone 2 Preview
The next milestone implements three LLM-based judge agents:
- **Relevance Judge** — scores how relevant the response is to the question
- **Accuracy Judge** — scores factual accuracy against the reference answer
- **Hallucination Detector** — detects claims not grounded in retrieved context
