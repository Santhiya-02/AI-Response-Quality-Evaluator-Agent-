"""
Benchmark Dataset Loader

Loads reference QA records from Hugging Face datasets.
Supports TriviaQA and SQuAD.
"""

from itertools import islice
from datasets import load_dataset


def _validate_num_samples(num_samples: int) -> None:
    """Validate the requested sample count."""
    if not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer.")

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")


def load_triviaqa(num_samples: int = 200) -> list[str]:
    """Load TriviaQA question-answer records."""
    _validate_num_samples(num_samples)

    try:
        dataset = load_dataset(
            "trivia_qa",
            "rc.nocontext",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        texts: list[str] = []

        for item in islice(dataset, num_samples):
            question = str(item.get("question", "")).strip()
            answer_data = item.get("answer") or {}

            canonical_answer = str(
                answer_data.get("value", "")
            ).strip()

            aliases = [
                str(answer).strip()
                for answer in answer_data.get("aliases", [])
                if str(answer).strip()
            ]

            answer = canonical_answer

            if not answer and aliases:
                answer = aliases[0]

            if question and answer:
                texts.append(
                    f"Question: {question}\n"
                    f"Reference answer: {answer}"
                )

        return texts

    except Exception as error:
        raise RuntimeError(
            f"Failed to load TriviaQA: {error}"
        ) from error


def load_squad(num_samples: int = 200) -> list[str]:
    """Load SQuAD question-answer-context records."""
    _validate_num_samples(num_samples)

    try:
        dataset = load_dataset(
            "squad",
            split="train",
            streaming=True
        )

        texts: list[str] = []

        for item in islice(dataset, num_samples):
            question = str(item.get("question", "")).strip()
            context = str(item.get("context", "")).strip()

            answer_data = item.get("answers") or {}
            answers = answer_data.get("text", [])

            answer = (
                str(answers[0]).strip()
                if answers
                else ""
            )

            if question and answer and context:
                texts.append(
                    f"Question: {question}\n"
                    f"Reference answer: {answer}\n"
                    f"Supporting context: {context}"
                )

        return texts

    except Exception as error:
        raise RuntimeError(
            f"Failed to load SQuAD: {error}"
        ) from error


BENCHMARK_LOADERS = {
    "TriviaQA (200 QA pairs)": lambda: load_triviaqa(200),
    "SQuAD (200 QA pairs)": lambda: load_squad(200),
}