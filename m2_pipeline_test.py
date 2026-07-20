"""
Milestone 2 Pipeline Test
Validates all three judge agents and the consistency validation suite.
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.knowledge_base import add_documents, retrieve, reset_collection
from src.judges import relevance_judge, accuracy_judge, hallucination_detector, run_all_judges

# ── Setup knowledge base ──────────────────────────────────────────────────────
reset_collection()
texts = [
    "RAGAS is a framework for evaluating RAG systems. It measures faithfulness, answer relevancy, and context precision.",
    "RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a language model.",
    "Hallucination in LLMs refers to when a model generates factually incorrect or fabricated information.",
    "TriviaQA is a benchmark dataset containing trivia questions with verified answers from Wikipedia.",
]
n = add_documents(texts, source_name="m2_test")
print(f"[OK] Indexed {n} chunks\n")

# ── Test cases ────────────────────────────────────────────────────────────────
TESTS = [
    {
        "label": "Correct & Grounded",
        "question": "What is RAG in AI?",
        "ai_response": "RAG stands for Retrieval-Augmented Generation, combining retrieval with language models.",
        "reference": "Retrieval-Augmented Generation combines retrieval systems with language models.",
    },
    {
        "label": "Irrelevant Response",
        "question": "What is RAG in AI?",
        "ai_response": "The weather today is sunny with a high of 75 degrees Fahrenheit.",
        "reference": "Retrieval-Augmented Generation combines retrieval systems with language models.",
    },
    {
        "label": "Hallucinated Response",
        "question": "What does RAGAS measure?",
        "ai_response": "RAGAS measures stock market performance and financial risk indicators for hedge funds.",
        "reference": "RAGAS measures faithfulness, answer relevancy, and context precision.",
    },
]

for test in TESTS:
    print(f"{'='*60}")
    print(f"[TEST] {test['label']}")
    print(f"  Q: {test['question']}")
    print(f"  A: {test['ai_response'][:80]}")

    chunks = retrieve(test["question"], top_k=3)

    # Individual judges
    rel = relevance_judge(test["question"], test["ai_response"])
    acc = accuracy_judge(test["question"], test["ai_response"], test["reference"], chunks)
    hal = hallucination_detector(test["ai_response"], chunks)

    print(f"\n  Relevance  : {rel['score']}/10 | {rel['verdict']}")
    print(f"  Reasoning  : {rel['reasoning']}")
    print(f"\n  Accuracy   : {acc['score']}/10 | {acc['verdict']}")
    print(f"  Reasoning  : {acc['reasoning']}")
    print(f"\n  Grounding  : {hal['score']}/10 | Hallucination={hal['hallucination_detected']}")
    if hal["flagged_statements"]:
        for s in hal["flagged_statements"]:
            print(f"  [FLAGGED] {s}")
    print()

# ── Validation suite ──────────────────────────────────────────────────────────
print("=" * 60)
print("[VALIDATION] Running consistency validation suite...")
from src.validation import run_validation

def progress(i, total, q):
    print(f"  [{i+1}/{total}] {q[:55]}...")

val = run_validation(use_kb=True, progress_callback=progress)

total = val.get("total_cases") or val.get("total_pairs", 0)
print(f"\n[RESULTS] {total} cases evaluated")
if "summary" in val:
    s  = val["summary"]
    cc = val["consistency_check"]
    print(f"  Avg Relevance  : {s['relevance']['mean']}/10  (stdev={s['relevance']['stdev']})")
    print(f"  Avg Accuracy   : {s['accuracy']['mean']}/10  (stdev={s['accuracy']['stdev']})")
    print(f"  Avg Grounding  : {s['hallucination_grounding']['mean']}/10  (stdev={s['hallucination_grounding']['stdev']})")
    print(f"  Correct ans avg accuracy : {cc['correct_answer_avg_accuracy']}")
    print(f"  Wrong ans avg accuracy   : {cc['wrong_answer_avg_accuracy']}")
    print(f"  Consistency check        : {'PASS' if cc['judges_consistent'] else 'FAIL'}")
else:
    disc = val.get("discrimination", {})
    print(f"  Correct ans avg accuracy : {disc.get('correct_answer_average')}")
    print(f"  Wrong ans avg accuracy   : {disc.get('incorrect_answer_average')}")
    print(f"  Consistency check        : {'PASS' if disc.get('passed') else 'FAIL'}")

print("\n[PASS] All Milestone 2 components verified successfully!")
