"""
AI Response Quality Evaluator Agent — Milestone 1 Demo
Streamlit application demonstrating:
  1. Evaluation Input Module
  2. Reference Knowledge Base (chunking, embedding, vector indexing)
  3. Semantic Retrieval demonstration
  4. Baseline scoring pipeline
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
from src.input_module import build_evaluation_input
from src.knowledge_base import add_documents, retrieve, get_collection_stats, reset_collection
from src.scoring import score_response

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Response Quality Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        border: 1px solid #2d6a9f;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card .label { color: #7eb8f7; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-card .value { color: #ffffff; font-size: 2rem; font-weight: 700; margin-top: 4px; }
    .metric-card .sub   { color: #a0c4e8; font-size: 0.75rem; margin-top: 2px; }
    .score-bar-bg { background: #1a2a3a; border-radius: 8px; height: 10px; margin-top: 6px; }
    .score-bar    { border-radius: 8px; height: 10px; }
    .chunk-card {
        background: #0f1e2e;
        border-left: 4px solid #2d6a9f;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #cde0f5;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green  { background: #1a4731; color: #4ade80; }
    .badge-yellow { background: #3d3000; color: #facc15; }
    .badge-red    { background: #3d1010; color: #f87171; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #7eb8f7;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 6px;
        margin-bottom: 14px;
    }
    div[data-testid="stSidebar"] { background: #0a1520; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def score_badge(score):
    if score is None:
        return '<span class="badge badge-yellow">N/A</span>'
    if score >= 0.7:
        return f'<span class="badge badge-green">✓ {score:.2f}</span>'
    if score >= 0.4:
        return f'<span class="badge badge-yellow">~ {score:.2f}</span>'
    return f'<span class="badge badge-red">✗ {score:.2f}</span>'


def score_color(score):
    if score is None:
        return "#4a5568"
    if score >= 0.7:
        return "#4ade80"
    if score >= 0.4:
        return "#facc15"
    return "#f87171"


def render_metric_card(label, value, sub="", score=None):
    bar_html = ""
    if score is not None:
        color = score_color(score)
        width = int(score * 100)
        bar_html = f"""
        <div class="score-bar-bg">
          <div class="score-bar" style="width:{width}%;background:{color};"></div>
        </div>"""
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="sub">{sub}</div>
      {bar_html}
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("🧠 AI Evaluator")
    st.caption("Milestone 1 — Foundation Demo")
    st.divider()

    st.markdown("### 📚 Knowledge Base")
    stats = get_collection_stats()
    st.metric("Indexed Chunks", stats["total_chunks"])

    st.markdown("**Load Benchmark Dataset**")
    dataset_choice = st.selectbox(
        "Select dataset",
        ["— select —", "TriviaQA (200 QA pairs)", "SQuAD (200 passages)"],
        label_visibility="collapsed"
    )
    if st.button("⬇ Load Dataset", use_container_width=True):
        if dataset_choice == "— select —":
            st.warning("Please select a dataset first.")
        else:
            with st.spinner(f"Loading {dataset_choice}..."):
                from src.benchmark_loader import BENCHMARK_LOADERS
                texts = BENCHMARK_LOADERS[dataset_choice]()
                n = add_documents(texts, source_name=dataset_choice)
                st.success(f"Added {n} chunks from {dataset_choice}")
                st.rerun()

    if st.button("🗑 Reset Knowledge Base", use_container_width=True, type="secondary"):
        reset_collection()
        st.success("Knowledge base cleared.")
        st.rerun()

    st.divider()
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files and st.button("📥 Index Documents", use_container_width=True):
        with st.spinner("Indexing documents..."):
            file_data = [(f.name, f.read()) for f in uploaded_files]
            from src.input_module import parse_uploaded_file
            texts = [parse_uploaded_file(n, b) for n, b in file_data]
            n = add_documents(texts, source_name="user_docs")
            st.success(f"Indexed {n} chunks from {len(uploaded_files)} file(s).")
            st.rerun()

    st.divider()
    st.caption("🔬 Embedding: all-MiniLM-L6-v2\n🗄 Vector DB: ChromaDB\n📐 Chunk size: 400 chars")


# ── Main Content ──────────────────────────────────────────────────────────────
st.title("🧠 AI Response Quality Evaluator Agent")
st.caption("Milestone 1 — Evaluation Input Module · Reference Knowledge Base · Semantic Retrieval")

tab1, tab2, tab3 = st.tabs(["📝 Evaluate Response", "🔍 Retrieval Demo", "🏗 Architecture"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Evaluate Response
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.markdown('<div class="section-header">📥 Evaluation Input</div>', unsafe_allow_html=True)

        question = st.text_area(
            "Question *",
            placeholder="e.g. What is the capital of France?",
            height=80
        )
        ai_response = st.text_area(
            "AI Response *",
            placeholder="Paste the AI-generated response to evaluate...",
            height=120
        )
        reference_answer = st.text_area(
            "Reference Answer (optional)",
            placeholder="Ground-truth answer for accuracy scoring...",
            height=80
        )

        top_k = st.slider("Top-K retrieval chunks", 1, 10, 5)

        evaluate_btn = st.button("🚀 Evaluate", type="primary", use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">📊 Evaluation Results</div>', unsafe_allow_html=True)

        if evaluate_btn:
            if not question.strip() or not ai_response.strip():
                st.error("Question and AI Response are required fields.")
            else:
                with st.spinner("Running evaluation pipeline..."):
                    eval_input = build_evaluation_input(
                        question=question,
                        ai_response=ai_response,
                        reference_answer=reference_answer
                    )
                    valid, msg = eval_input.is_valid()
                    if not valid:
                        st.error(msg)
                    else:
                        retrieved = retrieve(eval_input.question, top_k=top_k) if stats["total_chunks"] > 0 else []
                        scores = score_response(
                            question=eval_input.question,
                            ai_response=eval_input.ai_response,
                            reference_answer=eval_input.reference_answer,
                            retrieved_chunks=retrieved
                        )

                        # Score cards
                        c1, c2 = st.columns(2)
                        with c1:
                            v = scores["question_response_relevance"]
                            render_metric_card("Question–Response Relevance",
                                               f"{v:.2f}" if v else "—", "semantic similarity", v)
                        with c2:
                            v = scores["retrieval_relevance"]
                            render_metric_card("Retrieval Relevance",
                                               f"{v:.2f}" if v else "—",
                                               "avg chunk similarity" if v else "no KB loaded", v)

                        c3, c4 = st.columns(2)
                        with c3:
                            v = scores["response_grounding"]
                            render_metric_card("Response Grounding",
                                               f"{v:.2f}" if v else "—",
                                               "vs top chunk" if v else "no KB loaded", v)
                        with c4:
                            v = scores["semantic_similarity"]
                            render_metric_card("Semantic Similarity",
                                               f"{v:.2f}" if v else "—",
                                               "vs reference" if v else "no reference", v)

                        if scores["token_f1"] is not None:
                            st.markdown(f"**Token F1 (vs reference):** {score_badge(scores['token_f1'])}", unsafe_allow_html=True)

                        # Retrieved chunks
                        if retrieved:
                            st.markdown("---")
                            st.markdown(f"**Top {len(retrieved)} Retrieved Chunks**")
                            for i, chunk in enumerate(retrieved):
                                st.markdown(f"""
                                <div class="chunk-card">
                                  <b>#{i+1}</b> &nbsp;
                                  {score_badge(chunk['similarity_score'])} &nbsp;
                                  <span style="color:#7eb8f7;font-size:0.78rem;">source: {chunk['source']}</span>
                                  <br><br>{chunk['chunk'][:300]}{'...' if len(chunk['chunk']) > 300 else ''}
                                </div>""", unsafe_allow_html=True)
                        elif stats["total_chunks"] == 0:
                            st.info("💡 Load a benchmark dataset or upload documents in the sidebar to enable retrieval scoring.")
        else:
            st.info("Fill in the form and click **Evaluate** to run the pipeline.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Retrieval Demo
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔍 Semantic Retrieval Demonstration</div>', unsafe_allow_html=True)
    st.caption("Demonstrates that a query can retrieve the most relevant reference chunks from the vector database.")

    demo_query = st.text_input(
        "Enter a query to search the knowledge base",
        placeholder="e.g. Who invented the telephone?"
    )
    demo_k = st.slider("Number of results", 1, 10, 5, key="demo_k")

    if st.button("🔎 Search Knowledge Base", type="primary"):
        if not demo_query.strip():
            st.warning("Please enter a query.")
        elif stats["total_chunks"] == 0:
            st.warning("Knowledge base is empty. Load a dataset or upload documents first.")
        else:
            with st.spinner("Searching..."):
                results = retrieve(demo_query, top_k=demo_k)

            st.success(f"Retrieved {len(results)} chunks for: *\"{demo_query}\"*")

            # Similarity table
            df = pd.DataFrame([{
                "Rank": i + 1,
                "Similarity": f"{r['similarity_score']:.4f}",
                "Source": r["source"],
                "Chunk Preview": r["chunk"][:120] + "..."
            } for i, r in enumerate(results)])
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("**Full Chunk Details**")
            for i, r in enumerate(results):
                with st.expander(f"Chunk #{i+1} — Similarity: {r['similarity_score']:.4f} | Source: {r['source']}"):
                    st.write(r["chunk"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Architecture
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🏗 System Architecture — Milestone 1</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Orchestration Flow")
        st.code("""
┌─────────────────────────────────────────┐
│           EVALUATION INPUT MODULE        │
│  • Question + AI Response (required)     │
│  • Reference Answer (optional)           │
│  • Document Upload PDF/TXT/MD (optional) │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  DOCUMENT PARSER   │
         │  PyPDF2 / plain txt│
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────────────────┐
         │    CHUNKING ENGINE             │
         │  400-char chunks, 80 overlap   │
         └─────────┬──────────────────────┘
                   │
         ┌─────────▼──────────────────────┐
         │    EMBEDDING MODEL             │
         │  all-MiniLM-L6-v2 (local)      │
         │  384-dim dense vectors         │
         └─────────┬──────────────────────┘
                   │
         ┌─────────▼──────────────────────┐
         │    VECTOR DATABASE             │
         │  ChromaDB (persistent)         │
         │  Cosine similarity index       │
         └─────────┬──────────────────────┘
                   │
         ┌─────────▼──────────────────────┐
         │    SEMANTIC RETRIEVAL          │
         │  Top-K nearest neighbors       │
         │  Returns chunks + scores       │
         └─────────┬──────────────────────┘
                   │
         ┌─────────▼──────────────────────┐
         │    SCORING MODULE (M1 Base)    │
         │  • Question-Response Relevance │
         │  • Retrieval Relevance         │
         │  • Response Grounding          │
         │  • Semantic Similarity (ref)   │
         │  • Token F1 (ref)              │
         └────────────────────────────────┘
        """, language="text")

    with col_b:
        st.markdown("#### Agent Responsibility Document")
        agents = {
            "Agent": [
                "Evaluation Input Module",
                "Document Parser",
                "Chunking Engine",
                "Embedding Model",
                "Vector DB (ChromaDB)",
                "Retrieval Engine",
                "Scoring Module (M1)",
                "Relevance Judge (M2)",
                "Accuracy Judge (M2)",
                "Hallucination Detector (M2)"
            ],
            "Milestone": ["1","1","1","1","1","1","1","2","2","2"],
            "Responsibility": [
                "Accept & validate all evaluation inputs",
                "Extract text from PDF/TXT/MD uploads",
                "Split documents into overlapping chunks",
                "Generate 384-dim semantic embeddings",
                "Persist & index vectors with cosine similarity",
                "Retrieve top-K relevant chunks for a query",
                "Compute baseline similarity & F1 scores",
                "Score response relevance to question (LLM)",
                "Score factual accuracy vs reference (LLM)",
                "Detect hallucinations vs retrieved context (LLM)"
            ]
        }
        st.dataframe(pd.DataFrame(agents), use_container_width=True, hide_index=True)

        st.markdown("#### Database Schema")
        st.code("""
ChromaDB Collection: reference_knowledge_base
┌──────────────┬──────────────────────────────┐
│ Field        │ Description                  │
├──────────────┼──────────────────────────────┤
│ id           │ MD5 hash (source + chunk)    │
│ document     │ Raw chunk text               │
│ embedding    │ 384-dim float vector         │
│ metadata     │ {source, chunk_index}        │
└──────────────┴──────────────────────────────┘

Evaluation Record (in-session):
┌──────────────────┬──────────────────────────┐
│ question         │ User input string        │
│ ai_response      │ LLM output to evaluate   │
│ reference_answer │ Ground truth (optional)  │
│ retrieved_chunks │ List[{chunk, score, src}]│
│ scores           │ Dict of metric scores    │
└──────────────────┴──────────────────────────┘
        """, language="text")

        st.markdown("#### Scoring Design")
        scoring_df = pd.DataFrame({
            "Metric": ["Question-Response Relevance", "Retrieval Relevance", "Response Grounding",
                       "Semantic Similarity", "Token F1"],
            "Method": ["Cosine sim (embed)", "Avg cosine sim (chunks)", "Cosine sim (response vs top chunk)",
                       "Cosine sim (response vs reference)", "Token overlap F1"],
            "Requires": ["Always", "KB loaded", "KB loaded", "Reference answer", "Reference answer"],
            "Range": ["0–1", "0–1", "0–1", "0–1", "0–1"]
        })
        st.dataframe(scoring_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Technology Stack")
    tech_cols = st.columns(4)
    techs = [
        ("🖥 Frontend", "Streamlit"),
        ("🧮 Embeddings", "sentence-transformers\nall-MiniLM-L6-v2"),
        ("🗄 Vector DB", "ChromaDB\n(persistent, local)"),
        ("📦 Datasets", "HuggingFace\nTriviaQA · SQuAD"),
    ]
    for col, (title, desc) in zip(tech_cols, techs):
        with col:
            st.markdown(f"**{title}**\n\n{desc}")
