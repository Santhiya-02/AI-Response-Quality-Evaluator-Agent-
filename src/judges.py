"""
LLM Judge Agents powered by Groq:
  - RelevanceJudge
  - AccuracyJudge
  - HallucinationDetector
"""

import json
import os
import re
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_groq_client = None


def _client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _call(prompt: str, max_tokens: int = 512) -> str:
    resp = _client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(text: str) -> dict:
    """Extract first JSON object from LLM output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ── Relevance Judge ───────────────────────────────────────────────────────────

def relevance_judge(question: str, ai_response: str) -> dict:
    """
    Score how relevant the AI response is to the question.
    Returns: {score: 0-10, reasoning: str, verdict: str}
    """
    prompt = f"""You are a strict relevance evaluator. Score how relevant the AI response is to the question.

Question: {question}
AI Response: {ai_response}

Scoring scale:
0-3  : Off-topic or completely irrelevant
4-6  : Partially relevant, misses key aspects
7-9  : Mostly relevant, addresses the question well
10   : Perfectly relevant and directly answers the question

Respond ONLY with valid JSON:
{{"score": <integer 0-10>, "reasoning": "<one sentence explaining the score>", "verdict": "<Highly Relevant|Partially Relevant|Not Relevant>"}}"""

    raw = _call(prompt)
    result = _parse_json(raw)
    return {
        "score": int(result.get("score", 0)),
        "reasoning": result.get("reasoning", raw),
        "verdict": result.get("verdict", "Unknown"),
    }


# ── Accuracy Judge ────────────────────────────────────────────────────────────

def accuracy_judge(question: str, ai_response: str,
                   reference_answer: str = None,
                   retrieved_chunks: list[dict] = None) -> dict:
    """
    Score factual accuracy of the AI response against reference answer and/or retrieved chunks.
    Returns: {score: 0-10, reasoning: str, verdict: str, evidence: str}
    """
    reference_section = ""
    if reference_answer:
        reference_section += f"\nReference Answer: {reference_answer}"
    if retrieved_chunks:
        top_chunks = "\n".join(f"- {c['chunk'][:300]}" for c in retrieved_chunks[:3])
        reference_section += f"\nRetrieved Source Chunks:\n{top_chunks}"

    if not reference_section:
        return {
            "score": None,
            "reasoning": "No reference answer or retrieved chunks provided.",
            "verdict": "Cannot Evaluate",
            "evidence": "",
        }

    prompt = f"""You are a strict factual accuracy evaluator. Score the factual correctness of the AI response.

Question: {question}
AI Response: {ai_response}
{reference_section}

Scoring scale:
0-3  : Factually incorrect or contradicts the reference
4-6  : Partially correct, some inaccuracies
7-9  : Mostly accurate with minor gaps
10   : Fully accurate and consistent with the reference

Respond ONLY with valid JSON:
{{"score": <integer 0-10>, "reasoning": "<one sentence>", "verdict": "<Accurate|Partially Accurate|Inaccurate>", "evidence": "<quote from reference that supports or contradicts the response>"}}"""

    raw = _call(prompt)
    result = _parse_json(raw)
    return {
        "score": int(result.get("score", 0)),
        "reasoning": result.get("reasoning", raw),
        "verdict": result.get("verdict", "Unknown"),
        "evidence": result.get("evidence", ""),
    }


# ── Hallucination Detector ────────────────────────────────────────────────────

def hallucination_detector(ai_response: str, retrieved_chunks: list[dict]) -> dict:
    """
    Identify claims in the AI response not grounded in retrieved context.
    Returns: {hallucination_detected: bool, flagged_statements: list[str], reasoning: str, score: 0-10}
    Score = grounding score (10 = fully grounded, 0 = fully hallucinated).
    """
    if not retrieved_chunks:
        return {
            "hallucination_detected": None,
            "flagged_statements": [],
            "reasoning": "No retrieved context available to cross-reference.",
            "score": None,
        }

    context = "\n".join(f"[{i+1}] {c['chunk'][:300]}" for i, c in enumerate(retrieved_chunks[:4]))

    prompt = f"""You are a hallucination detection expert. Identify any claims in the AI response that are NOT supported by the provided context.

AI Response: {ai_response}

Retrieved Context:
{context}

Instructions:
- List specific statements from the AI response that cannot be verified from the context.
- If all claims are grounded, return an empty list.
- Grounding score: 10 = fully grounded, 0 = completely hallucinated.

Respond ONLY with valid JSON:
{{"hallucination_detected": <true|false>, "flagged_statements": ["<statement1>", "<statement2>"], "reasoning": "<brief explanation>", "score": <integer 0-10>}}"""

    raw = _call(prompt, max_tokens=600)
    result = _parse_json(raw)

    flagged = result.get("flagged_statements", [])
    if isinstance(flagged, str):
        flagged = [flagged]

    return {
        "hallucination_detected": bool(result.get("hallucination_detected", False)),
        "flagged_statements": flagged,
        "reasoning": result.get("reasoning", raw),
        "score": int(result.get("score", 0)),
    }


# ── Combined Judge Runner ─────────────────────────────────────────────────────

def run_all_judges(question: str, ai_response: str,
                   reference_answer: str = None,
                   retrieved_chunks: list[dict] = None) -> dict:
    """Run all three judges and return combined results."""
    return {
        "relevance": relevance_judge(question, ai_response),
        "accuracy": accuracy_judge(question, ai_response, reference_answer, retrieved_chunks),
        "hallucination": hallucination_detector(ai_response, retrieved_chunks or []),
    }
