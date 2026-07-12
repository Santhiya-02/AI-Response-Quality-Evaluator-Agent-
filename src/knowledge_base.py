"""
Knowledge Base Module
Handles document chunking, embedding generation, and vector indexing using ChromaDB.
"""

import os
import hashlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 400       # characters per chunk
CHUNK_OVERLAP = 80     # overlap between chunks
COLLECTION_NAME = "reference_knowledge_base"

_embedder = None
_client = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]


def add_documents(texts: list[str], source_name: str = "uploaded") -> int:
    """Chunk, embed, and index a list of text documents. Returns number of chunks added."""
    collection = _get_collection()
    embedder = _get_embedder()

    all_chunks, all_ids, all_metas = [], [], []
    for text_idx, text in enumerate(texts):
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = hashlib.md5(f"{source_name}_{text_idx}_{i}_{chunk}".encode()).hexdigest()
            all_chunks.append(chunk)
            all_ids.append(doc_id)
            all_metas.append({"source": source_name, "chunk_index": i})

    if not all_chunks:
        return 0

    embeddings = embedder.encode(all_chunks, show_progress_bar=False).tolist()
    collection.upsert(documents=all_chunks, embeddings=embeddings, ids=all_ids, metadatas=all_metas)
    return len(all_chunks)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve top-k most semantically relevant chunks for a query."""
    collection = _get_collection()
    embedder = _get_embedder()

    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "chunk": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "similarity_score": round(1 - dist, 4)   # cosine similarity
        })
    return output


def get_collection_stats() -> dict:
    """Return basic stats about the knowledge base."""
    collection = _get_collection()
    count = collection.count()
    return {"total_chunks": count, "collection_name": COLLECTION_NAME, "db_path": CHROMA_DB_PATH}


def reset_collection():
    """Clear all documents from the knowledge base."""
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
    _client.delete_collection(COLLECTION_NAME)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
