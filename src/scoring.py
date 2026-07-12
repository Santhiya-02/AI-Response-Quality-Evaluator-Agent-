"""
Scoring Module (Milestone 1 Foundation)
Provides basic lexical and semantic similarity scores.
Full judge agents (Relevance, Accuracy, Hallucination) are implemented in Milestone 2.
"""

import re
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


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


def score_response(question: str, ai_response: str, reference_answer: str = None,
                   retrieved_chunks: list[dict] = None) -> dict:
    """
    Compute Milestone 1 baseline scores.
    Returns a score dict used as input to Milestone 2 judge agents.
    """
    scores = {}

    # Retrieval relevance: avg similarity of top retrieved chunks to the question
    if retrieved_chunks:
        chunk_sims = [c["similarity_score"] for c in retrieved_chunks]
        scores["retrieval_relevance"] = round(sum(chunk_sims) / len(chunk_sims), 4)
        # Response-to-context grounding
        top_chunk = retrieved_chunks[0]["chunk"] if retrieved_chunks else ""
        scores["response_grounding"] = semantic_similarity(ai_response, top_chunk)
    else:
        scores["retrieval_relevance"] = None
        scores["response_grounding"] = None

    # Reference-based scores (when reference answer is provided)
    if reference_answer:
        scores["semantic_similarity"] = semantic_similarity(ai_response, reference_answer)
        scores["token_f1"] = token_overlap_f1(ai_response, reference_answer)
    else:
        scores["semantic_similarity"] = None
        scores["token_f1"] = None

    # Question-response relevance
    scores["question_response_relevance"] = semantic_similarity(question, ai_response)

    return scores
