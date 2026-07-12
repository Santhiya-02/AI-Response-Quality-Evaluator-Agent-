# AI Response Quality Evaluator Agent — Milestone 1

## Project Overview
A multi-agent system for evaluating the quality of AI-generated responses using semantic retrieval, reference knowledge bases, and LLM-based judge agents.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND (app.py)                    │
│   Tab 1: Evaluate Response  |  Tab 2: Retrieval Demo  |  Tab 3: Arch │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    EVALUATION INPUT MODULE   │
              │       (src/input_module.py)  │
              │  • Validates question + resp │
              │  • Parses PDF/TXT/MD uploads │
              │  • Builds EvaluationInput    │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│  BENCHMARK      │ │  DOCUMENT       │ │  SCORING        │
│  LOADER         │ │  CHUNKER        │ │  MODULE         │
│  (benchmark_    │ │  (knowledge_    │ │  (scoring.py)   │
│   loader.py)    │ │   base.py)      │ │                 │
│  TriviaQA/SQuAD │ │  400-char chunks│ │  Cosine sim     │
└────────┬────────┘ │  80-char overlap│ │  Token F1       │
         │          └────────┬────────┘ └────────▲────────┘
         │                   │                   │
         └──────────┬────────┘                   │
                    │                            │
         ┌──────────▼──────────┐                 │
         │   EMBEDDING MODEL   │                 │
         │  all-MiniLM-L6-v2   │                 │
         │  384-dim vectors    │                 │
         └──────────┬──────────┘                 │
                    │                            │
         ┌──────────▼──────────┐                 │
         │   VECTOR DATABASE   │                 │
         │   ChromaDB          │                 │
         │   Cosine similarity │                 │
         │   Persistent index  │                 │
         └──────────┬──────────┘                 │
                    │                            │
         ┌──────────▼──────────┐                 │
         │  RETRIEVAL ENGINE   ├─────────────────┘
         │  Top-K semantic     │  retrieved chunks
         │  nearest neighbors  │  + similarity scores
         └─────────────────────┘
```

---

## Orchestration Flow

1. **User submits** question + AI response (+ optional reference answer + optional documents)
2. **Input Module** validates inputs and parses any uploaded files
3. **Chunking Engine** splits text into 400-char overlapping chunks
4. **Embedding Model** encodes chunks into 384-dim vectors
5. **ChromaDB** upserts vectors with metadata (source, chunk_index)
6. **Retrieval Engine** embeds the query and fetches top-K nearest chunks
7. **Scoring Module** computes baseline metrics using retrieved chunks and optional reference
8. **UI** displays scores, retrieved chunks, and similarity bars

---

## Agent Responsibility Document

| Agent | Milestone | File | Responsibility |
|---|---|---|---|
| Evaluation Input Module | 1 | `src/input_module.py` | Accept & validate question, AI response, reference answer, document uploads |
| Document Parser | 1 | `src/input_module.py` | Extract text from PDF, TXT, MD files |
| Chunking Engine | 1 | `src/knowledge_base.py` | Split documents into 400-char overlapping chunks |
| Embedding Model | 1 | `src/knowledge_base.py` | Generate 384-dim semantic embeddings (all-MiniLM-L6-v2) |
| Vector Database | 1 | `src/knowledge_base.py` | Persist and index vectors with cosine similarity (ChromaDB) |
| Retrieval Engine | 1 | `src/knowledge_base.py` | Retrieve top-K relevant chunks for a query |
| Benchmark Loader | 1 | `src/benchmark_loader.py` | Load TriviaQA / SQuAD datasets to seed knowledge base |
| Scoring Module (baseline) | 1 | `src/scoring.py` | Compute semantic similarity, token F1, retrieval relevance |
| Relevance Judge Agent | 2 | `src/agents/relevance.py` | LLM-based scoring of response relevance to question |
| Accuracy Judge Agent | 2 | `src/agents/accuracy.py` | LLM-based factual accuracy vs reference answer |
| Hallucination Detector | 2 | `src/agents/hallucination.py` | Detect claims not grounded in retrieved context |

---

## Scoring Design

| Metric | Method | Requires | Range |
|---|---|---|---|
| Question-Response Relevance | Cosine similarity (embeddings) | Always | 0–1 |
| Retrieval Relevance | Avg cosine similarity of top-K chunks | KB loaded | 0–1 |
| Response Grounding | Cosine similarity (response vs top chunk) | KB loaded | 0–1 |
| Semantic Similarity | Cosine similarity (response vs reference) | Reference answer | 0–1 |
| Token F1 | Token overlap F1 (SQuAD-style) | Reference answer | 0–1 |

Score interpretation: ≥0.7 = Good (green), 0.4–0.7 = Moderate (yellow), <0.4 = Poor (red)

---

## Database Schema

### ChromaDB Collection: `reference_knowledge_base`

| Field | Type | Description |
|---|---|---|
| `id` | string | MD5 hash of source + chunk content |
| `document` | string | Raw chunk text (≤400 chars) |
| `embedding` | float[384] | Sentence embedding vector |
| `metadata.source` | string | Dataset name or filename |
| `metadata.chunk_index` | int | Position of chunk within source document |

### Evaluation Record (in-session)

| Field | Type | Description |
|---|---|---|
| `question` | string | User-submitted question |
| `ai_response` | string | AI-generated response to evaluate |
| `reference_answer` | string? | Ground-truth answer (optional) |
| `retrieved_chunks` | list[dict] | Top-K chunks with similarity scores |
| `scores` | dict | All computed metric scores |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.35 |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Vector Database | ChromaDB (persistent, local) |
| Document Parsing | PyPDF2 |
| Benchmark Datasets | HuggingFace Datasets (TriviaQA, SQuAD) |
| Language | Python 3.10+ |

---

## Milestone 1 Deliverables Checklist

- [x] Documented understanding of LLM evaluation, hallucinations, RAG, RAGAS, TruLens (see Architecture tab)
- [x] Architecture diagram, orchestration flow, agent responsibility document, scoring design, database schema
- [x] Working Evaluation Input Module (question, AI response, reference answer, document uploads)
- [x] Reference Knowledge Base with chunking, embeddings, vector indexing, semantic retrieval
- [x] Demonstration that a query retrieves the most relevant reference chunks (Retrieval Demo tab)
