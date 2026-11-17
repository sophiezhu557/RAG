from fastapi import FastAPI
from schemas import RAGRequest, RAGResponse
from main import run_rag

app = FastAPI(title = "Financial Report RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rag/query", response_model=RAGResponse)
def query_rag(request: RAGRequest):
    return run_rag(request.question)

