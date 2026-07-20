"""
Scoring Module

Provides:
- Semantic similarity
- Token-overlap F1
- Retrieval diagnostics
- LLM judge execution
- Weighted final score and verdict
"""

import re
import string
from collections import Counter
from typing import Optional

from sentence_transformers import util

from src.knowledge_base import _get_embedder


JUDGE_WEIGHTS = {
    "relevance": 0.25,
    "accuracy": 0.35,
    "hallucination": 0.25,
    "completeness": 0.15
}


def semantic_similarity(
    text_a: str,
    text_b: str
) -> float:
    """
    Calculate cosine similarity between two texts.

    The returned value normally ranges from -1 to 1.
    Higher values indicate greater semantic similarity.
    """
    text_a = (text_a or "").strip()
    text_b = (text_b or "").strip()

    if not text_a or not text_b:
        return 0.0

    embedder = _get_embedder()

    embeddings = embedder.encode(
        [text_a, text_b],
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    score = util.cos_sim(
        embeddings[0],
        embeddings[1]
    ).item()

    score = max(-1.0, min(1.0, float(score)))

    return round(score, 4)


def _normalize_answer(text: str) -> list[str]:
    """Apply SQuAD-style answer normalization."""
    text = (text or "").lower()

    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )

    text = " ".join(text.split())

    return text.split()


def token_overlap_f1(
    prediction: str,
    reference: str
) -> float:
    """Calculate SQuAD-style token-level F1."""
    prediction_tokens = _normalize_answer(prediction)
    reference_tokens = _normalize_answer(reference)

    if not prediction_tokens or not reference_tokens:
        return 0.0

    common_tokens = (
        Counter(prediction_tokens)
        & Counter(reference_tokens)
    )

    common_count = sum(common_tokens.values())

    if common_count == 0:
        return 0.0

    precision = (
        common_count / len(prediction_tokens)
    )

    recall = (
        common_count / len(reference_tokens)
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
    )

    return round(f1, 4)


def _valid_retrieved_chunks(
    retrieved_chunks: Optional[list[dict]]
) -> list[dict]:
    """Return retrieved chunks containing usable text."""
    if not retrieved_chunks:
        return []

    return [
        chunk
        for chunk in retrieved_chunks
        if isinstance(chunk, dict)
        and str(chunk.get("chunk", "")).strip()
    ]


def _retrieval_scores(
    retrieved_chunks: list[dict]
) -> list[float]:
    """Extract valid retrieval-similarity scores."""
    scores: list[float] = []

    for chunk in retrieved_chunks:
        value = chunk.get("similarity_score")

        if value is None or isinstance(value, bool):
            continue

        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue

    return scores


def aggregate_judge_scores(
    judge_results: dict
) -> dict:
    """
    Calculate a weighted overall judge score.

    Missing judge scores are excluded, and the remaining
    weights are normalized. Coverage shows how much of the
    intended evaluation was successfully completed.
    """
    weighted_total = 0.0
    available_weight = 0.0
    component_scores: dict[str, Optional[float]] = {}

    for judge_name, weight in JUDGE_WEIGHTS.items():
        judge_result = judge_results.get(
            judge_name,
            {}
        )

        score = judge_result.get("score")

        if score is None or isinstance(score, bool):
            component_scores[judge_name] = None
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            component_scores[judge_name] = None
            continue

        if not 0 <= score <= 10:
            component_scores[judge_name] = None
            continue

        component_scores[judge_name] = score
        weighted_total += score * weight
        available_weight += weight

    coverage = round(available_weight, 2)

    if available_weight == 0:
        return {
            "overall_score": None,
            "verdict": "Cannot Evaluate",
            "evaluation_coverage": 0.0,
            "limited_evidence": True,
            "component_scores": component_scores
        }

    normalized_score = (
        weighted_total / available_weight
    )

    percentage_score = round(
        normalized_score * 10,
        2
    )

    if percentage_score >= 85:
        verdict = "Excellent"
    elif percentage_score >= 70:
        verdict = "Good"
    elif percentage_score >= 50:
        verdict = "Needs Improvement"
    else:
        verdict = "Poor"

    limited_evidence = coverage < 0.75

    if limited_evidence:
        verdict = f"{verdict} — Limited Evidence"

    return {
        "overall_score": percentage_score,
        "verdict": verdict,
        "evaluation_coverage": coverage,
        "limited_evidence": limited_evidence,
        "component_scores": component_scores
    }


def score_response(
    question: str,
    ai_response: str,
    reference_answer: Optional[str] = None,
    retrieved_chunks: Optional[list[dict]] = None,
    run_judges: bool = False
) -> dict:
    """
    Calculate baseline metrics and optionally run LLM judges.
    """
    question = (question or "").strip()
    ai_response = (ai_response or "").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not ai_response:
        raise ValueError("AI response cannot be empty.")

    reference_answer = (
        reference_answer.strip()
        if reference_answer
        and reference_answer.strip()
        else None
    )

    chunks = _valid_retrieved_chunks(
        retrieved_chunks
    )

    scores: dict = {}

    retrieval_scores = _retrieval_scores(chunks)

    if retrieval_scores:
        scores["retrieval_relevance"] = {
            "average": round(
                sum(retrieval_scores)
                / len(retrieval_scores),
                4
            ),
            "maximum": round(
                max(retrieval_scores),
                4
            )
        }
    else:
        scores["retrieval_relevance"] = None

    if chunks:
        combined_context = "\n".join(
            str(chunk.get("chunk", ""))[:1000]
            for chunk in chunks[:5]
        )

        scores["response_grounding"] = (
            semantic_similarity(
                ai_response,
                combined_context
            )
        )
    else:
        scores["response_grounding"] = None

    if reference_answer:
        scores["semantic_similarity"] = (
            semantic_similarity(
                ai_response,
                reference_answer
            )
        )

        scores["token_f1"] = token_overlap_f1(
            ai_response,
            reference_answer
        )
    else:
        scores["semantic_similarity"] = None
        scores["token_f1"] = None

    scores["question_response_relevance"] = (
        semantic_similarity(
            question,
            ai_response
        )
    )

    if run_judges:
        from src.judges import run_all_judges

        judge_results = run_all_judges(
            question=question,
            ai_response=ai_response,
            reference_answer=reference_answer,
            retrieved_chunks=chunks
        )

        scores["judges"] = judge_results
        scores["final_evaluation"] = (
            aggregate_judge_scores(
                judge_results
            )
        )

    return scores