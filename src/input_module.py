"""
Evaluation Input Module
Validates and structures inputs: question, AI response, optional reference answer, optional documents.
"""

import io
import PyPDF2
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvaluationInput:
    question: str
    ai_response: str
    reference_answer: Optional[str] = None
    uploaded_texts: list[str] = field(default_factory=list)
    source_label: str = "user_upload"

    def is_valid(self) -> tuple[bool, str]:
        if not self.question.strip():
            return False, "Question cannot be empty."
        if not self.ai_response.strip():
            return False, "AI response cannot be empty."
        return True, "OK"

    def has_reference(self) -> bool:
        return bool(self.reference_answer and self.reference_answer.strip())

    def has_documents(self) -> bool:
        return len(self.uploaded_texts) > 0


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from PDF bytes."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def parse_uploaded_file(filename: str, file_bytes: bytes) -> str:
    """Dispatch to correct parser based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("txt", "md"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Please upload PDF, TXT, or MD files.")


def build_evaluation_input(
    question: str,
    ai_response: str,
    reference_answer: str = "",
    uploaded_files: list[tuple[str, bytes]] = None,
    source_label: str = "user_upload"
) -> EvaluationInput:
    """Build and return a validated EvaluationInput object."""
    texts = []
    if uploaded_files:
        for fname, fbytes in uploaded_files:
            text = parse_uploaded_file(fname, fbytes)
            if text.strip():
                texts.append(text)

    return EvaluationInput(
        question=question.strip(),
        ai_response=ai_response.strip(),
        reference_answer=reference_answer.strip() if reference_answer else None,
        uploaded_texts=texts,
        source_label=source_label
    )
