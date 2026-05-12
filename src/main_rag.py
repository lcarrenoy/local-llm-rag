"""
Local LLM & Private RAG — FastAPI Server
Puerto: 8002
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from src.rag import ingest_documents, query_rag, get_vectorstore

app = FastAPI(
    title="Local LLM & Private RAG",
    description="100% offline RAG: Ollama + nomic-embed + ChromaDB",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    model: str
    embed_model: str

class IngestResponse(BaseModel):
    status: str
    documents_loaded: Optional[int] = None
    chunks_created: Optional[int] = None
    message: Optional[str] = None

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Local LLM & Private RAG",
        "version": "1.0.0",
        "status": "running",
        "mode": "100% offline",
        "endpoints": ["/ingest", "/query", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "mode": "offline"}

@app.post("/ingest", response_model=IngestResponse)
def ingest(docs_dir: str = "./docs"):
    """Ingest documents from docs/ folder into ChromaDB"""
    try:
        result = ingest_documents(docs_dir)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Query the RAG system with a question"""
    try:
        result = query_rag(req.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
def list_collections():
    """List ChromaDB collections info"""
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        return {"documents_in_db": count, "chroma_dir": "./chroma_db"}
    except Exception as e:
        return {"documents_in_db": 0, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
