from typing import Literal, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    language: Literal["vi", "en", "auto"] = "auto"


class Source(BaseModel):
    doc_id: str
    doc_type: Literal["product", "policy"]
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float
