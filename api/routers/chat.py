import time
from fastapi import APIRouter, Depends

from api.schemas import ChatRequest, ChatResponse, Source
from api.dependencies import get_retriever, get_generator

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, retriever=Depends(get_retriever), generator=Depends(get_generator)):
    start = time.perf_counter()

    results = retriever.search(req.query)
    context = "\n\n---\n\n".join(r["content"] for r in results)

    gen = await generator.generate(req.query, context, language=req.language)

    sources = [
        Source(
            doc_id=r["metadata"].get("doc_id", r["id"]),
            doc_type=r["metadata"].get("doc_type", "product"),
            score=r["score"],
        )
        for r in results
    ]

    latency_ms = (time.perf_counter() - start) * 1000

    return ChatResponse(answer=gen["answer"], sources=sources, latency_ms=latency_ms)
