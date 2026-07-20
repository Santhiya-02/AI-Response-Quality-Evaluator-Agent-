"""
Judge Validation

Tests:
- Correct-versus-incorrect discrimination
- Classification accuracy
- Repeated-run consistency
- Judge failure rate
"""

import statistics
from typing import Optional

from src.judges import run_all_judges
from src.knowledge_base import retrieve


def _mean(values: list[float]) -> Optional[float]:
    """Return a rounded mean for non-empty values."""
    return (
        round(statistics.mean(values), 2)
        if values
        else None
    )


def _stdev(values: list[float]) -> Optional[float]:
    """Return a rounded sample standard deviation."""
    if not values:
        return None

    if len(values) == 1:
        return 0.0

    return round(statistics.stdev(values), 2)


def _numeric_scores(
    runs: list[dict],
    judge_name: str
) -> list[float]:
    """Extract valid numeric judge scores."""
    scores: list[float] = []

    for run in runs:
        score = (
            run.get(judge_name, {})
            .get("score")
        )

        if score is None or isinstance(score, bool):
            continue

        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue

    return scores


def _majority_boolean(
    values: list[Optional[bool]]
) -> Optional[bool]:
    """Return the majority Boolean result."""
    valid_values = [
        value
        for value in values
        if isinstance(value, bool)
    ]

    if not valid_values:
        return None

    true_count = sum(valid_values)
    false_count = len(valid_values) - true_count

    return true_count > false_count


def _controlled_context(pair: dict) -> list[dict]:
    """
    Create controlled evidence for judge validation.

    This isolates judge quality from retrieval quality.
    """
    reference = str(
        pair.get("reference_answer", "")
    ).strip()

    if not reference:
        return []

    return [{
        "chunk": reference,
        "source": "validation_reference",
        "source_type": "controlled_validation",
        "chunk_index": 0,
        "similarity_score": 1.0
    }]
BENCHMARK_PAIRS = [
    {
        "id": "capital_correct",
        "question": "What is the capital of France?",
        "ai_response": "The capital of France is Paris.",
        "reference_answer": (
            "Paris is the capital of France."
        ),
        "expected_accurate": True,
        "expected_relevant": True,
        "expected_hallucination": False,
    },
    {
        "id": "capital_incorrect",
        "question": "What is the capital of France?",
        "ai_response": (
            "The capital of France is Berlin, "
            "a major European city."
        ),
        "reference_answer": (
            "Paris is the capital of France."
        ),
        "expected_accurate": False,
        "expected_relevant": True,
        "expected_hallucination": True,
    },
    {
        "id": "capital_irrelevant",
        "question": "What is the capital of France?",
        "ai_response": (
            "Photosynthesis allows plants to use "
            "sunlight to produce energy."
        ),
        "reference_answer": (
            "Paris is the capital of France."
        ),
        "expected_accurate": False,
        "expected_relevant": False,
        "expected_hallucination": True,
    },
    {
        "id": "rag_correct",
        "question": "What does RAG stand for in AI?",
        "ai_response": (
            "RAG stands for Retrieval-Augmented "
            "Generation, a technique that combines "
            "retrieval systems with language models."
        ),
        "reference_answer": (
            "RAG stands for Retrieval-Augmented "
            "Generation. It combines information "
            "retrieval with language-model generation."
        ),
        "expected_accurate": True,
        "expected_relevant": True,
        "expected_hallucination": False,
    },
    {
        "id": "rag_incorrect",
        "question": "What does RAG stand for in AI?",
        "ai_response": (
            "RAG is a neural-network architecture "
            "used for image classification."
        ),
        "reference_answer": (
            "RAG stands for Retrieval-Augmented "
            "Generation. It combines information "
            "retrieval with language-model generation."
        ),
        "expected_accurate": False,
        "expected_relevant": True,
        "expected_hallucination": True,
    },
    {
        "id": "hamlet_correct",
        "question": "Who wrote the play Hamlet?",
        "ai_response": (
            "Hamlet was written by William Shakespeare."
        ),
        "reference_answer": (
            "Hamlet is a play written by "
            "William Shakespeare."
        ),
        "expected_accurate": True,
        "expected_relevant": True,
        "expected_hallucination": False,
    },
    {
        "id": "hamlet_incorrect",
        "question": "Who wrote the play Hamlet?",
        "ai_response": (
            "Hamlet is a novel written in the "
            "19th century by Charles Dickens."
        ),
        "reference_answer": (
            "Hamlet is a play written by "
            "William Shakespeare."
        ),
        "expected_accurate": False,
        "expected_relevant": True,
        "expected_hallucination": True,
    },
    {
        "id": "photosynthesis_correct",
        "question": "What is photosynthesis?",
        "ai_response": (
            "Photosynthesis is the process by which "
            "plants use sunlight, water and carbon "
            "dioxide to produce glucose and oxygen."
        ),
        "reference_answer": (
            "Photosynthesis is the process by which "
            "plants use sunlight, water and carbon "
            "dioxide to produce glucose and oxygen."
        ),
        "expected_accurate": True,
        "expected_relevant": True,
        "expected_hallucination": False,
    },
    {
        "id": "photosynthesis_incorrect",
        "question": "What is photosynthesis?",
        "ai_response": (
            "Photosynthesis is how animals digest "
            "food using stomach acids."
        ),
        "reference_answer": (
            "Photosynthesis is the process by which "
            "plants use sunlight, water and carbon "
            "dioxide to produce glucose and oxygen."
        ),
        "expected_accurate": False,
        "expected_relevant": True,
        "expected_hallucination": True,
    },
    {
        "id": "photosynthesis_irrelevant",
        "question": "What is photosynthesis?",
        "ai_response": (
            "William Shakespeare wrote the play Hamlet."
        ),
        "reference_answer": (
            "Photosynthesis is the process by which "
            "plants use sunlight, water and carbon "
            "dioxide to produce glucose and oxygen."
        ),
        "expected_accurate": False,
        "expected_relevant": False,
        "expected_hallucination": True,
    },
]

def run_validation(
    repetitions: int = 1,
    context_mode: str = "controlled",
    progress_callback=None,
    use_kb: Optional[bool] = None
) -> dict:
    """
    Run judges against labeled benchmark cases.

    repetitions=1 performs a quick functional validation.
    repetitions=3 or more measures repeated-run consistency.

    context_mode:
    - controlled: use the known reference as evidence
    - knowledge_base: test retrieval and judging together
    """
    if use_kb is not None:
        context_mode = "knowledge_base" if use_kb else "controlled"

    if repetitions < 1:
        raise ValueError(
            "repetitions must be at least 1."
        )

    if context_mode not in {
        "controlled",
        "knowledge_base"
    }:
        raise ValueError(
            "context_mode must be 'controlled' "
            "or 'knowledge_base'."
        )

    case_results: list[dict] = []
    total_runs = len(BENCHMARK_PAIRS) * repetitions
    completed_runs = 0

    for pair in BENCHMARK_PAIRS:
        runs: list[dict] = []

        if context_mode == "controlled":
            chunks = _controlled_context(pair)
        else:
            chunks = retrieve(
                pair["question"],
                top_k=3,
                source_type="benchmark"
            )

        for repetition in range(repetitions):
            judgments = run_all_judges(
                question=pair["question"],
                ai_response=pair["ai_response"],
                reference_answer=pair.get(
                    "reference_answer"
                ),
                retrieved_chunks=chunks
            )

            runs.append(judgments)
            completed_runs += 1

            if progress_callback:
                progress_callback(
                    completed_runs,
                    total_runs,
                    pair["question"]
                )

        relevance_scores = _numeric_scores(
            runs,
            "relevance"
        )

        accuracy_scores = _numeric_scores(
            runs,
            "accuracy"
        )

        hallucination_scores = _numeric_scores(
            runs,
            "hallucination"
        )

        completeness_scores = _numeric_scores(
            runs,
            "completeness"
        )

        hallucination_decisions = [
            run.get(
                "hallucination",
                {}
            ).get("hallucination_detected")
            for run in runs
        ]

        case_results.append({
            "id": pair.get("id", pair["question"]),
            "question": pair["question"],
            "ai_response": pair["ai_response"],
            "reference_answer": pair.get(
                "reference_answer"
            ),
            "expected_accurate": pair.get(
                "expected_accurate"
            ),
            "expected_relevant": pair.get(
                "expected_relevant"
            ),
            "expected_hallucination": pair.get(
                "expected_hallucination"
            ),
            "relevance_mean": _mean(
                relevance_scores
            ),
            "relevance_stdev": _stdev(
                relevance_scores
            ),
            "accuracy_mean": _mean(
                accuracy_scores
            ),
            "accuracy_stdev": _stdev(
                accuracy_scores
            ),
            "hallucination_grounding_mean": _mean(
                hallucination_scores
            ),
            "hallucination_grounding_stdev": _stdev(
                hallucination_scores
            ),
            "completeness_mean": _mean(
                completeness_scores
            ),
            "completeness_stdev": _stdev(
                completeness_scores
            ),
            "hallucination_detected": (
                _majority_boolean(
                    hallucination_decisions
                )
            ),
            "runs": runs
        })

    return _compute_summary(
        case_results=case_results,
        repetitions=repetitions,
        context_mode=context_mode
    )


def _classification_accuracy(
    expected: list[bool],
    predicted: list[Optional[bool]]
) -> Optional[float]:
    """Calculate classification accuracy."""
    comparisons = [
        expected_value == predicted_value
        for expected_value, predicted_value
        in zip(expected, predicted)
        if isinstance(expected_value, bool)
        and isinstance(predicted_value, bool)
    ]

    if not comparisons:
        return None

    return round(
        sum(comparisons) / len(comparisons),
        4
    )


def _compute_summary(
    case_results: list[dict],
    repetitions: int,
    context_mode: str
) -> dict:
    """Calculate validation statistics."""
    correct_cases = [
        case
        for case in case_results
        if case.get("expected_accurate") is True
        and case.get("accuracy_mean") is not None
    ]

    incorrect_cases = [
        case
        for case in case_results
        if case.get("expected_accurate") is False
        and case.get("accuracy_mean") is not None
    ]

    correct_accuracy_scores = [
        case["accuracy_mean"]
        for case in correct_cases
    ]

    incorrect_accuracy_scores = [
        case["accuracy_mean"]
        for case in incorrect_cases
    ]

    correct_average = _mean(
        correct_accuracy_scores
    )

    incorrect_average = _mean(
        incorrect_accuracy_scores
    )

    separation_margin = None

    if (
        correct_average is not None
        and incorrect_average is not None
    ):
        separation_margin = round(
            correct_average - incorrect_average,
            2
        )

    expected_accuracy = [
        case.get("expected_accurate")
        for case in case_results
    ]

    predicted_accuracy = [
        (
            case["accuracy_mean"] >= 7
            if case["accuracy_mean"] is not None
            else None
        )
        for case in case_results
    ]

    expected_relevance = [
        case.get("expected_relevant")
        for case in case_results
    ]

    predicted_relevance = [
        (
            case["relevance_mean"] >= 7
            if case["relevance_mean"] is not None
            else None
        )
        for case in case_results
    ]

    expected_hallucination = [
        case.get("expected_hallucination")
        for case in case_results
    ]

    predicted_hallucination = [
        case.get("hallucination_detected")
        for case in case_results
    ]

    consistency_values: list[float] = []

    if repetitions > 1:
        for case in case_results:
            for field in [
                "relevance_stdev",
                "accuracy_stdev",
                "hallucination_grounding_stdev",
                "completeness_stdev"
            ]:
                value = case.get(field)

                if value is not None:
                    consistency_values.append(value)

    average_run_variation = (
        _mean(consistency_values)
        if repetitions > 1
        else None
    )

    judge_failures = 0
    total_judge_calls = 0

    for case in case_results:
        for run in case["runs"]:
            for judge_name in [
                "relevance",
                "accuracy",
                "hallucination",
                "completeness"
            ]:
                total_judge_calls += 1

                if (
                    run.get(judge_name, {})
                    .get("score") is None
                ):
                    judge_failures += 1

    failure_rate = (
        round(
            judge_failures / total_judge_calls,
            4
        )
        if total_judge_calls
        else 0.0
    )

    return {
        "total_cases": len(case_results),
        "repetitions": repetitions,
        "context_mode": context_mode,
        "discrimination": {
            "correct_answer_average": correct_average,
            "incorrect_answer_average": (
                incorrect_average
            ),
            "separation_margin": separation_margin,
            "passed": (
                separation_margin is not None
                and separation_margin > 0
            )
        },
        "classification_accuracy": {
            "accuracy_judge": (
                _classification_accuracy(
                    expected_accuracy,
                    predicted_accuracy
                )
            ),
            "relevance_judge": (
                _classification_accuracy(
                    expected_relevance,
                    predicted_relevance
                )
            ),
            "hallucination_detector": (
                _classification_accuracy(
                    expected_hallucination,
                    predicted_hallucination
                )
            )
        },
        "consistency": {
            "measured": repetitions > 1,
            "average_score_stdev": (
                average_run_variation
            ),
            "stable": (
                average_run_variation is not None
                and average_run_variation <= 1.0
            )
        },
        "reliability": {
            "judge_failures": judge_failures,
            "total_judge_calls": total_judge_calls,
            "failure_rate": failure_rate
        },
        "per_case": case_results
    }