"""
Benchmark Dataset Loader
Loads reference QA pairs from HuggingFace datasets to seed the knowledge base.
Supports: TriviaQA, SQuAD
"""

from datasets import load_dataset


def load_triviaqa(num_samples: int = 200) -> list[str]:
    """Load TriviaQA samples as text passages for the knowledge base."""
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", trust_remote_code=True)
    texts = []
    for item in ds.select(range(min(num_samples, len(ds)))):
        q = item.get("question", "")
        answers = item.get("answer", {}).get("aliases", [])
        ans_str = " | ".join(answers[:3]) if answers else ""
        if q and ans_str:
            texts.append(f"Question: {q}\nAnswer: {ans_str}")
    return texts


def load_squad(num_samples: int = 200) -> list[str]:
    """Load SQuAD context passages for the knowledge base."""
    ds = load_dataset("squad", split="train", trust_remote_code=True)
    seen, texts = set(), []
    for item in ds:
        ctx = item.get("context", "").strip()
        if ctx and ctx not in seen:
            seen.add(ctx)
            texts.append(ctx)
        if len(texts) >= num_samples:
            break
    return texts


BENCHMARK_LOADERS = {
    "TriviaQA (200 QA pairs)": lambda: load_triviaqa(200),
    "SQuAD (200 passages)": lambda: load_squad(200),
}
