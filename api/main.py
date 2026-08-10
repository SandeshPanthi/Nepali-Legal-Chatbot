from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.rag_pipeline import get_llm, rag_llm

app = FastAPI(title= "Legal RAG API")

# Initialize RAG components
embedding_manager = EmbeddingManager()
vector_store = VectorStore()

retriever = RAGRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager
)

llm = get_llm()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return{"message": "Backend is running."}


@app.post("/chat")
def chat(request: QueryRequest):
    try:
        result = rag_llm(
            query=request.query,
            retriever=retriever,
            llm=llm
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question."
        )

