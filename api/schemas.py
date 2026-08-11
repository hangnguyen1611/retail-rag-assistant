from typing import Literal
from pydantic import BaseModel


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    language: Literal["vi", "en", "auto"] = "auto"
    history: list[HistoryTurn] = []


class Source(BaseModel):
    doc_id: str
    doc_type: Literal["product", "policy"]
    title: str = ""
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float