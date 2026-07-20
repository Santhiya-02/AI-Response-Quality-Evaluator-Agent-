"""
Knowledge Base Module

Handles document chunking, embedding generation,
vector indexing, and semantic retrieval using ChromaDB.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    str(PROJECT_ROOT / "chroma_db")
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
COLLECTION_NAME = "reference_knowledge_base"
MIN_SIMILARITY_SCORE = 0.25

_embedder: Optional[SentenceTransformer] = None
_client = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    """Load and cache the embedding model."""
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)

    return _embedder


def _get_collection():
    """Create or return the persistent ChromaDB collection."""
    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": EMBEDDING_MODEL
            }
        )

    return _collection


def _clean_text(text: str) -> str:
    """Remove unnecessary whitespace from text."""
    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping word-safe chunks."""
    text = _clean_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            break_position = text.rfind(" ", start, end)

            if break_position > start:
                end = break_position

        chunk = text[start:end].strip()

        if len(chunk) > 30:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap

        while (
            next_start > start
            and next_start < len(text)
            and not text[next_start].isspace()
        ):
            next_start -= 1

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def add_documents(
    texts: list[str],
    source_name: str = "uploaded",
    source_type: str = "user_upload"
) -> int:
    """
    Chunk, embed and index documents.

    Returns the number of chunks indexed.
    """
    if not texts:
        return 0

    source_name = (source_name or "uploaded").strip()
    source_type = (source_type or "user_upload").strip()

    collection = _get_collection()
    embedder = _get_embedder()

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadata: list[dict] = []

    for document_index, text in enumerate(texts):
        cleaned_text = _clean_text(text)

        if not cleaned_text:
            continue

        document_hash = hashlib.sha256(
            cleaned_text.encode("utf-8")
        ).hexdigest()

        chunks = chunk_text(cleaned_text)

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(
                (
                    f"{source_type}|{source_name}|"
                    f"{document_hash}|{chunk_index}|{chunk}"
                ).encode("utf-8")
            ).hexdigest()

            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({
                "source": source_name,
                "source_type": source_type,
                "document_index": document_index,
                "document_hash": document_hash,
                "chunk_index": chunk_index
            })

    if not all_chunks:
        return 0

    embeddings = embedder.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata
    )

    return len(all_chunks)


def retrieve(
    query: str,
    top_k: int = 5,
    source_name: Optional[str] = None,
    source_type: Optional[str] = None,
    min_similarity: float = MIN_SIMILARITY_SCORE
) -> list[dict]:
    """Retrieve semantically relevant chunks."""
    query = (query or "").strip()

    if not query:
        raise ValueError("Retrieval query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError(
            "min_similarity must be between -1 and 1."
        )

    collection = _get_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    filters = []

    if source_name:
        filters.append({"source": source_name})

    if source_type:
        filters.append({"source_type": source_type})

    where_filter = None

    if len(filters) == 1:
        where_filter = filters[0]
    elif len(filters) > 1:
        where_filter = {"$and": filters}

    embedder = _get_embedder()

    query_embedding = embedder.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True
    ).tolist()

    query_arguments = {
        "query_embeddings": query_embedding,
        "n_results": min(top_k, total_chunks),
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if where_filter:
        query_arguments["where"] = where_filter

    results = collection.query(**query_arguments)

    documents = results.get("documents", [[]])[0]
    metadata = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output: list[dict] = []

    for document, meta, distance in zip(
        documents,
        metadata,
        distances
    ):
        similarity = 1.0 - float(distance)

        if similarity < min_similarity:
            continue

        output.append({
            "chunk": document,
            "source": meta.get("source", "unknown"),
            "source_type": meta.get(
                "source_type",
                "unknown"
            ),
            "document_index": meta.get(
                "document_index",
                0
            ),
            "chunk_index": meta.get(
                "chunk_index",
                0
            ),
            "similarity_score": round(similarity, 4)
        })

    return output


def get_collection_stats() -> dict:
    """Return basic knowledge-base statistics."""
    collection = _get_collection()

    return {
        "total_chunks": collection.count(),
        "collection_name": COLLECTION_NAME,
        "db_path": CHROMA_DB_PATH,
        "embedding_model": EMBEDDING_MODEL
    }


def reset_collection() -> None:
    """Clear all documents from the knowledge base."""
    global _client, _collection

    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL
        }
    )