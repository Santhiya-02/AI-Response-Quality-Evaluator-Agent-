"""
Judge Validation — tests scoring consistency across benchmark QA pairs.
"""

import statistics
from src.judges import run_all_judges
from src.knowledge_base import retrieve


# Curated benchmark pairs covering varied difficulty and topics
BENCHMARK_PAIRS = [
    {
        "question": "What is the capital of France?",
        "ai_response": "The capital of France is Paris.",
        "reference_answer": "Paris",
    },
    {
        "question": "What is the capital of France?",
        "ai_response": "The capital of France is Berlin, a major European city.",
        "reference_answer": "Paris",
    },
    {
        "question": "What does RAG stand for in AI?",
        "ai_response": "RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval systems with language models.",
        "reference_answer": "Retrieval-Augmented Generation",
    },
    {
        "question": "What does RAG stand for in AI?",
        "ai_response": "RAG is a type of neural network architecture used for image classification tasks.",
        "reference_answer": "Retrieval-Augmented Generation",
    },
    {
        "question": "Who wrote the play Hamlet?",
        "ai_response": "Hamlet was written by William Shakespeare.",
        "reference_answer": "William Shakespeare",
    },
    {
        "question": "Who wrote the play Hamlet?",
        "ai_response": "Hamlet is a famous novel about a Danish prince, written in the 19th century.",
        "reference_answer": "William Shakespeare",
    },
    {
        "question": "What is photosynthesis?",
        "ai_response": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose.",
        "reference_answer": "Photosynthesis is the process plants use to convert light energy into chemical energy stored as glucose.",
    },
    {
        "question": "What is photosynthesis?",
        "ai_response": "Photosynthesis is how animals digest food using stomach acids to break down nutrients.",
        "reference_answer": "Photosynthesis is the process plants use to convert light energy into chemical energy stored as glucose.",
    },
]


def run_validation(use_kb: bool = True, progress_callback=None) -> dict:
    """
    Run all judges on benchmark pairs and compute consistency metrics.
    Returns summary statistics and per-pair results.
    """
    results = []

    for i, pair in enumerate(BENCHMARK_PAIRS):
        if progress_callback:
            progress_callback(i, len(BENCHMARK_PAIRS), pair["question"])

        chunks = retrieve(pair["question"], top_k=3) if use_kb else []

        judgments = run_all_judges(
            question=pair["question"],
            ai_response=pair["ai_response"],
            reference_answer=pair["reference_answer"],
            retrieved_chunks=chunks,
        )

        results.append({
            "question": pair["question"],
            "ai_response": pair["ai_response"],
            "reference_answer": pair["reference_answer"],
            "relevance_score": judgments["relevance"]["score"],
            "relevance_verdict": judgments["relevance"]["verdict"],
            "relevance_reasoning": judgments["relevance"]["reasoning"],
            "accuracy_score": judgments["accuracy"]["score"],
            "accuracy_verdict": judgments["accuracy"]["verdict"],
            "accuracy_reasoning": judgments["accuracy"]["reasoning"],
            "accuracy_evidence": judgments["accuracy"]["evidence"],
            "hallucination_detected": judgments["hallucination"]["hallucination_detected"],
            "hallucination_score": judgments["hallucination"]["score"],
            "flagged_statements": judgments["hallucination"]["flagged_statements"],
            "hallucination_reasoning": judgments["hallucination"]["reasoning"],
        })

    return _compute_summary(results)


def _compute_summary(results: list[dict]) -> dict:
    """Compute consistency and quality statistics from validation results."""
    rel_scores = [r["relevance_score"] for r in results if r["relevance_score"] is not None]
    acc_scores = [r["accuracy_score"] for r in results if r["accuracy_score"] is not None]
    hal_scores = [r["hallucination_score"] for r in results if r["hallucination_score"] is not None]

    # Consistency: correct pairs (index 0,2,4,6) should score higher than incorrect (1,3,5,7)
    correct_pairs = results[0::2]   # even indices = correct answers
    wrong_pairs   = results[1::2]   # odd  indices = wrong answers

    def avg(lst): return round(statistics.mean(lst), 2) if lst else None
    def stdev(lst): return round(statistics.stdev(lst), 2) if len(lst) > 1 else 0.0

    correct_acc = [r["accuracy_score"] for r in correct_pairs if r["accuracy_score"] is not None]
    wrong_acc   = [r["accuracy_score"] for r in wrong_pairs   if r["accuracy_score"] is not None]

    consistency_check = (avg(correct_acc) or 0) > (avg(wrong_acc) or 0)

    return {
        "total_pairs": len(results),
        "summary": {
            "relevance": {"mean": avg(rel_scores), "stdev": stdev(rel_scores)},
            "accuracy":  {"mean": avg(acc_scores), "stdev": stdev(acc_scores)},
            "hallucination_grounding": {"mean": avg(hal_scores), "stdev": stdev(hal_scores)},
        },
        "consistency_check": {
            "correct_answer_avg_accuracy": avg(correct_acc),
            "wrong_answer_avg_accuracy": avg(wrong_acc),
            "judges_consistent": consistency_check,
        },
        "per_pair": results,
    }
