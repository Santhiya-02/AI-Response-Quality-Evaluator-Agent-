import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.knowledge_base import add_documents, retrieve, reset_collection
from src.scoring import score_response

reset_collection()

texts = [
    "RAGAS is a framework for evaluating RAG systems. It measures faithfulness, answer relevancy, and context precision.",
    "TruLens is an evaluation framework for LLM applications that provides feedback functions to measure quality of AI responses.",
    "Hallucination in LLMs refers to when a model generates factually incorrect or fabricated information not grounded in context.",
    "RAG combines a retrieval system with a language model to ground responses in factual documents.",
]
n = add_documents(texts, source_name="demo")
print(f"[OK] Indexed {n} chunks into ChromaDB")

results = retrieve("What is RAGAS used for?", top_k=3)
print(f"[OK] Retrieved {len(results)} chunks for query 'What is RAGAS used for?'")
for r in results:
    print(f"   [{r['similarity_score']:.4f}] {r['chunk'][:80]}...")

scores = score_response(
    question="What is RAGAS?",
    ai_response="RAGAS is an evaluation framework for RAG systems measuring faithfulness and relevancy.",
    reference_answer="RAGAS evaluates Retrieval Augmented Generation systems.",
    retrieved_chunks=results
)
print(f"\n[OK] Scoring pipeline complete:")
for k, v in scores.items():
    print(f"   {k}: {v}")

print("\n[PASS] All Milestone 1 components verified successfully!")
