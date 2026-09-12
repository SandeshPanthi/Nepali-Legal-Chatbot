from pydantic import BaseModel
from typing import List, Optional, Any


class ChatRequest(BaseModel):
    query: str


class Source(BaseModel):
    id: str
    source: str
    title: str
    page: str
    section: str
    rank: int


class Confidence(BaseModel):
    top_vector_score: float
    top_bm25_score: float
    top_rrf_score: float
    retrieved_documents: int
    agreement_count: int
    agreement_ratio: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: dict