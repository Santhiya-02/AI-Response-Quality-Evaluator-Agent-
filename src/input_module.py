"""
Evaluation Input Module
Validates and structures inputs: question, AI response,
optional reference answer, and optional documents.
"""

import io
from dataclasses import dataclass, field
from typing import Optional

import PyPDF2


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
        return bool(
            self.reference_answer
            and self.reference_answer.strip()
        )

    def has_documents(self) -> bool:
        return any(text.strip() for text in self.uploaded_texts)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from PDF bytes."""
    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))

        if reader.is_encrypted:
            raise ValueError(
                "Password-protected PDFs are not supported."
            )

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"Could not read the uploaded PDF: {error}"
        ) from error

    if not text.strip():
        raise ValueError(
            "No readable text was found in the PDF. "
            "The PDF may contain scanned images."
        )

    return text.strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT or Markdown bytes."""
    if not file_bytes:
        raise ValueError("The uploaded text file is empty.")

    text = file_bytes.decode("utf-8", errors="ignore").strip()

    if not text:
        raise ValueError(
            "No readable text was found in the uploaded file."
        )

    return text


def parse_uploaded_file(
    filename: str,
    file_bytes: bytes
) -> str:
    """Dispatch to the correct parser based on file extension."""
    if not filename or "." not in filename:
        raise ValueError(
            "The uploaded file must have a valid extension."
        )

    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        return extract_text_from_pdf(file_bytes)

    if extension in {"txt", "md"}:
        return extract_text_from_txt(file_bytes)

    raise ValueError(
        f"Unsupported file type: .{extension}. "
        "Please upload PDF, TXT, or MD files."
    )


def build_evaluation_input(
    question: str,
    ai_response: str,
    reference_answer: str = "",
    uploaded_files: Optional[list[tuple[str, bytes]]] = None,
    source_label: str = "user_upload"
) -> EvaluationInput:
    """Build and validate an EvaluationInput object."""
    texts: list[str] = []

    if uploaded_files:
        for filename, file_bytes in uploaded_files:
            text = parse_uploaded_file(filename, file_bytes)
            texts.append(text)

    evaluation_input = EvaluationInput(
        question=(question or "").strip(),
        ai_response=(ai_response or "").strip(),
        reference_answer=(
            reference_answer.strip()
            if reference_answer and reference_answer.strip()
            else None
        ),
        uploaded_texts=texts,
        source_label=(source_label or "user_upload").strip()
    )

    valid, message = evaluation_input.is_valid()

    if not valid:
        raise ValueError(message)

    return evaluation_input