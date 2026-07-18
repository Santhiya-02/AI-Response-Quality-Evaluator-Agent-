import os
import sys
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.input_module import build_evaluation_input
from src.knowledge_base import add_documents, retrieve, get_collection_stats, reset_collection
from src.scoring import score_response
from src.judges import run_all_judges
from src.validation import run_validation

app = FastAPI(title="AI Response Quality Evaluator API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    question: str
    ai_response: str
    reference_answer: Optional[str] = None
    top_k: int = 5

class JudgeRequest(BaseModel):
    question: str
    ai_response: str
    reference_answer: Optional[str] = None
    top_k: int = 4

class BenchmarkRequest(BaseModel):
    dataset: str

@app.get("/api/kb/stats")
def api_kb_stats():
    try:
        stats = get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kb/reset")
def api_kb_reset():
    try:
        reset_collection()
        return {"status": "success", "message": "Knowledge base reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kb/load-benchmark")
def api_kb_load_benchmark(req: BenchmarkRequest):
    try:
        from src.benchmark_loader import BENCHMARK_LOADERS
        if req.dataset not in BENCHMARK_LOADERS:
            raise HTTPException(status_code=400, detail=f"Invalid dataset. Choose from {list(BENCHMARK_LOADERS.keys())}")
        
        texts = BENCHMARK_LOADERS[req.dataset]()
        n = add_documents(texts, source_name=req.dataset)
        return {"status": "success", "chunks_added": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kb/upload")
async def api_kb_upload(files: List[UploadFile] = File(...)):
    try:
        from src.input_module import parse_uploaded_file
        texts = []
        for file in files:
            content = await file.read()
            text = parse_uploaded_file(file.filename, content)
            texts.append(text)
        n = add_documents(texts, source_name="user_docs")
        return {"status": "success", "chunks_added": n, "files_processed": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
def api_evaluate(req: EvaluationRequest):
    try:
        eval_input = build_evaluation_input(
            question=req.question,
            ai_response=req.ai_response,
            reference_answer=req.reference_answer
        )
        valid, msg = eval_input.is_valid()
        if not valid:
            raise HTTPException(status_code=400, detail=msg)
        
        stats = get_collection_stats()
        retrieved = retrieve(eval_input.question, top_k=req.top_k) if stats["total_chunks"] > 0 else []
        scores = score_response(
            question=eval_input.question,
            ai_response=eval_input.ai_response,
            reference_answer=eval_input.reference_answer,
            retrieved_chunks=retrieved
        )
        return {
            "scores": scores,
            "retrieved_chunks": retrieved
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/judge")
def api_judge(req: JudgeRequest):
    try:
        if not req.question.strip() or not req.ai_response.strip():
            raise HTTPException(status_code=400, detail="Question and AI Response are required.")
        
        stats = get_collection_stats()
        j_chunks = retrieve(req.question, top_k=req.top_k) if stats["total_chunks"] > 0 else []
        judgments = run_all_judges(
            question=req.question,
            ai_response=req.ai_response,
            reference_answer=req.reference_answer or None,
            retrieved_chunks=j_chunks,
        )
        return {
            "judgments": judgments,
            "retrieved_chunks": j_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validate")
def api_validate():
    try:
        stats = get_collection_stats()
        val = run_validation(use_kb=stats["total_chunks"] > 0)
        return val
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import FileResponse

@app.get("/")
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "FastAPI is running. index.html file not found in root."}
