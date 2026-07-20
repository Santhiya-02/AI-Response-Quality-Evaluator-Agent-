"""
FastAPI Backend
AI Response Quality Evaluator
"""

import logging
from pathlib import Path
from typing import Literal, Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.benchmark_loader import BENCHMARK_LOADERS
from src.input_module import (
    build_evaluation_input,
    parse_uploaded_file
)
from src.judges import run_all_judges
from src.knowledge_base import (
    add_documents,
    get_collection_stats,
    reset_collection,
    retrieve
)
from src.scoring import (
    aggregate_judge_scores,
    score_response
)
from src.validation import run_validation


# ------------------------------------------------------------------
# Application configuration
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
STATIC_DIR = BASE_DIR / "static"

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_UPLOAD_FILES = 5

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md"
}


app = FastAPI(
    title="AI Response Quality Evaluator API",
    description=(
        "Evaluates AI responses using semantic metrics, "
        "retrieval and specialized LLM judges."
    ),
    version="1.0.0"
)


# Serve styles.css and app.js.
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class EvaluationRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=20_000
    )

    ai_response: str = Field(
        ...,
        min_length=1,
        max_length=50_000
    )

    reference_answer: Optional[str] = Field(
        default=None,
        max_length=30_000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )

    source_type: Optional[str] = None


class JudgeRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=20_000
    )

    ai_response: str = Field(
        ...,
        min_length=1,
        max_length=50_000
    )

    reference_answer: Optional[str] = Field(
        default=None,
        max_length=30_000
    )

    top_k: int = Field(
        default=4,
        ge=1,
        le=8
    )

    source_type: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=20_000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )

    source_name: Optional[str] = None
    source_type: Optional[str] = None

    min_similarity: float = Field(
        default=0.25,
        ge=-1.0,
        le=1.0
    )


class BenchmarkRequest(BaseModel):
    dataset: str = Field(
        ...,
        min_length=1,
        max_length=100
    )


class ValidationRequest(BaseModel):
    repetitions: int = Field(
        default=1,
        ge=1,
        le=5
    )

    context_mode: Literal[
        "controlled",
        "knowledge_base"
    ] = "controlled"


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _retrieve_if_available(
    question: str,
    top_k: int,
    source_type: Optional[str] = None
) -> list[dict]:
    """Retrieve chunks only when the KB is not empty."""
    stats = get_collection_stats()

    if stats["total_chunks"] == 0:
        return []

    return retrieve(
        query=question,
        top_k=top_k,
        source_type=source_type
    )


def _index_uploaded_files(
    uploaded_items: list[tuple[str, bytes]]
) -> dict:
    """
    Parse and index uploaded files.

    This runs in a worker thread because PDF parsing and
    embedding generation are blocking operations.
    """
    total_chunks = 0
    indexed_files: list[dict] = []

    for filename, content in uploaded_items:
        text = parse_uploaded_file(
            filename,
            content
        )

        chunks_added = add_documents(
            texts=[text],
            source_name=filename,
            source_type="user_upload"
        )

        total_chunks += chunks_added

        indexed_files.append({
            "filename": filename,
            "chunks_added": chunks_added
        })

    return {
        "chunks_added": total_chunks,
        "files_processed": len(indexed_files),
        "files": indexed_files
    }


# ------------------------------------------------------------------
# Frontend routes
# ------------------------------------------------------------------

@app.get(
    "/",
    response_class=FileResponse,
    include_in_schema=False
)
def serve_index() -> FileResponse:
    """Serve the dashboard."""
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html was not found."
        )

    return FileResponse(INDEX_FILE)


@app.get("/api/health")
def health_check() -> dict:
    """Check whether the FastAPI backend is available."""
    return {
        "status": "healthy",
        "service": "AI Response Quality Evaluator"
    }


# ------------------------------------------------------------------
# Knowledge-base routes
# ------------------------------------------------------------------

@app.get("/api/kb/stats")
def api_kb_stats() -> dict:
    """Return knowledge-base statistics."""
    try:
        return get_collection_stats()

    except Exception as error:
        logger.exception(
            "Failed to retrieve knowledge-base statistics."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve knowledge-base "
                "statistics."
            )
        ) from error


@app.post("/api/kb/reset")
def api_kb_reset() -> dict:
    """Delete and recreate the ChromaDB collection."""
    try:
        reset_collection()

        return {
            "status": "success",
            "message": (
                "Knowledge base reset successfully."
            )
        }

    except Exception as error:
        logger.exception(
            "Knowledge-base reset failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Knowledge-base reset failed."
        ) from error


@app.post("/api/kb/load-benchmark")
def api_kb_load_benchmark(
    request: BenchmarkRequest
) -> dict:
    """Load a supported benchmark dataset."""
    dataset_name = request.dataset.strip()

    # Supports the label used by the older HTML file.
    dataset_aliases = {
        "SQuAD (200 passages)": (
            "SQuAD (200 QA pairs)"
        )
    }

    dataset_name = dataset_aliases.get(
        dataset_name,
        dataset_name
    )

    if dataset_name not in BENCHMARK_LOADERS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid benchmark dataset.",
                "available_datasets": list(
                    BENCHMARK_LOADERS.keys()
                )
            }
        )

    try:
        texts = BENCHMARK_LOADERS[
            dataset_name
        ]()

        chunks_added = add_documents(
            texts=texts,
            source_name=dataset_name,
            source_type="benchmark"
        )

        return {
            "status": "success",
            "dataset": dataset_name,
            "records_loaded": len(texts),
            "chunks_added": chunks_added
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Failed to load benchmark: %s",
            dataset_name
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to load benchmark "
                f"dataset: {dataset_name}."
            )
        ) from error


@app.post("/api/kb/upload")
async def api_kb_upload(
    files: list[UploadFile] = File(...)
) -> dict:
    """Parse and index uploaded PDF, TXT or MD files."""
    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one file is required."
        )

    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"A maximum of {MAX_UPLOAD_FILES} "
                "files can be uploaded at once."
            )
        )

    uploaded_items: list[tuple[str, bytes]] = []

    try:
        for uploaded_file in files:
            filename = Path(
                uploaded_file.filename or ""
            ).name

            if not filename:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Every uploaded file must "
                        "have a filename."
                    )
                )

            extension = Path(
                filename
            ).suffix.lower()

            if extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type for "
                        f"{filename}. Upload PDF, TXT "
                        "or MD files."
                    )
                )

            # Reading one extra byte detects oversized files.
            content = await uploaded_file.read(
                MAX_FILE_SIZE + 1
            )

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} is empty."
                )

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{filename} exceeds the "
                        "10 MB size limit."
                    )
                )

            uploaded_items.append(
                (filename, content)
            )

        result = await run_in_threadpool(
            _index_uploaded_files,
            uploaded_items
        )

        return {
            "status": "success",
            **result
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception(
            "Document upload and indexing failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The uploaded documents could not "
                "be processed."
            )
        ) from error

    finally:
        for uploaded_file in files:
            await uploaded_file.close()


@app.post("/api/kb/search")
def api_kb_search(
    request: SearchRequest
) -> dict:
    """Search the knowledge base directly."""
    try:
        results = retrieve(
            query=request.query.strip(),
            top_k=request.top_k,
            source_name=request.source_name,
            source_type=request.source_type,
            min_similarity=request.min_similarity
        )

        return {
            "query": request.query.strip(),
            "count": len(results),
            "results": results
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception(
            "Knowledge-base search failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Knowledge-base search failed."
        ) from error


# ------------------------------------------------------------------
# Evaluation routes
# ------------------------------------------------------------------

@app.post("/api/evaluate")
def api_evaluate(
    request: EvaluationRequest
) -> dict:
    """Calculate baseline response-quality metrics."""
    try:
        evaluation_input = build_evaluation_input(
            question=request.question,
            ai_response=request.ai_response,
            reference_answer=(
                request.reference_answer or ""
            )
        )

        retrieved_chunks = _retrieve_if_available(
            question=evaluation_input.question,
            top_k=request.top_k,
            source_type=request.source_type
        )

        scores = score_response(
            question=evaluation_input.question,
            ai_response=evaluation_input.ai_response,
            reference_answer=(
                evaluation_input.reference_answer
            ),
            retrieved_chunks=retrieved_chunks,
            run_judges=False
        )

        return {
            "scores": scores,
            "retrieved_chunks": retrieved_chunks
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception(
            "Response evaluation failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Response evaluation failed."
        ) from error


@app.post("/api/judge")
def api_judge(
    request: JudgeRequest
) -> dict:
    """Run all LLM judge agents."""
    question = request.question.strip()
    ai_response = request.ai_response.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if not ai_response:
        raise HTTPException(
            status_code=400,
            detail="AI response cannot be empty."
        )

    try:
        retrieved_chunks = _retrieve_if_available(
            question=question,
            top_k=request.top_k,
            source_type=request.source_type
        )

        judgments = run_all_judges(
            question=question,
            ai_response=ai_response,
            reference_answer=(
                request.reference_answer.strip()
                if request.reference_answer
                else None
            ),
            retrieved_chunks=retrieved_chunks
        )

        final_evaluation = (
            aggregate_judge_scores(
                judgments
            )
        )

        return {
            "judgments": judgments,
            "final_evaluation": final_evaluation,
            "retrieved_chunks": retrieved_chunks
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception(
            "Judge execution failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Judge execution failed."
        ) from error


# ------------------------------------------------------------------
# Validation routes
# ------------------------------------------------------------------

@app.post("/api/validate")
def api_validate(
    request: ValidationRequest
) -> dict:
    """Run the configurable validation suite."""
    try:
        return run_validation(
            repetitions=request.repetitions,
            context_mode=request.context_mode
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception(
            "Validation suite failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Validation suite failed."
        ) from error


@app.get("/api/validate")
def api_validate_quick() -> dict:
    """
    Backward-compatible quick validation endpoint.

    The frontend uses POST, but this keeps older code working.
    """
    try:
        return run_validation(
            repetitions=1,
            context_mode="controlled"
        )

    except Exception as error:
        logger.exception(
            "Quick validation failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Quick validation failed."
        ) from error