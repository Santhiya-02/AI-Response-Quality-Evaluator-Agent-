"""
Scoring Module — baseline lexical and semantic similarity scores.
"""

import re
import os
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_embedder = None
_groq_client = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using sentence embeddings."""
    embedder = _get_embedder()
    embs = embedder.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embs[0], embs[1]).item()
    return round(score, 4)


def token_overlap_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score (similar to SQuAD metric)."""
    def tokenize(s):
        return set(re.findall(r'\b\w+\b', s.lower()))

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def groq_judge(prompt: str) -> str:
    """Call Groq LLM with a prompt and return the response text."""
    client = _get_groq_client()
    if not client:
        return "Groq API key not configured."
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()


def score_response(question: str, ai_response: str, reference_answer: str = None,
                   retrieved_chunks: list[dict] = None,
                   run_judges: bool = False) -> dict:
    """
    Compute baseline scores (M1) and optionally run LLM judge agents (M2).
    Set run_judges=True to invoke Relevance, Accuracy, and Hallucination judges.
    """
    scores = {}

    if retrieved_chunks:
        chunk_sims = [c["similarity_score"] for c in retrieved_chunks]
        scores["retrieval_relevance"] = round(sum(chunk_sims) / len(chunk_sims), 4)
        scores["response_grounding"] = semantic_similarity(ai_response, retrieved_chunks[0]["chunk"])
    else:
        scores["retrieval_relevance"] = None
        scores["response_grounding"] = None

    if reference_answer:
        scores["semantic_similarity"] = semantic_similarity(ai_response, reference_answer)
        scores["token_f1"] = token_overlap_f1(ai_response, reference_answer)
    else:
        scores["semantic_similarity"] = None
        scores["token_f1"] = None

    scores["question_response_relevance"] = semantic_similarity(question, ai_response)

    if run_judges:
        from src.judges import run_all_judges
        scores["judges"] = run_all_judges(question, ai_response, reference_answer, retrieved_chunks)

    return scores
