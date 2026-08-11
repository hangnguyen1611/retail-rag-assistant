import json
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.dependencies import get_condenser, get_generator, get_retriever
from api.schemas import ChatRequest, ChatResponse, Source
from config.backend import ENABLE_CONDENSE, MAX_HISTORY_TURNS
from src.rag.query_filter import extract_product_filter

router = APIRouter()


def _title_of(hit):
    """
    Suy ra tiêu đề hiển thị cho khách từ 1 kết quả retrieval (hit).

    Product và policy có cấu trúc metadata khác nhau nên cần xử lý riêng:
    - policy: dùng tên file gốc (source_file) làm tiêu đề, dễ hiểu hơn cho khách so với doc_id kỹ thuật.
    Nếu thiếu source_file thì fallback về doc_id.
    - product: không có trường "title" sẵn trong metadata, nên lấy dòng ĐẦU TIÊN của content (chính là 
    dòng "Mã sản phẩm: SP{id}" do load_products_as_chunks() sinh ra) làm tiêu đề tạm.
    """
    metadata = hit.get("metadata", {})
    if metadata.get("doc_type") == "policy":
        return metadata.get("source_file", metadata.get("doc_id", ""))
    return hit["content"].split("\n", 1)[0].strip()


def _to_sources(results):
    """Chuyển list kết quả retrieval thô thành list Source (Pydantic schema)"""
    return [
        Source(
            doc_id=r["metadata"].get("doc_id", r["id"]),
            doc_type=r["metadata"].get("doc_type", "product"),
            title=_title_of(r),
            score=r["score"],
        )
        for r in results
    ]


def _messages_from_history(history):
    """
    Chuyển ChatRequest.history (list HistoryTurn) thành messages cho Groq.

    Giới hạn MAX_HISTORY_TURNS lượt gần nhất -- tránh prompt phình to theo độ dài phiên chat
    và context liên quan thường nằm ở vài lượt gần nhất.
    """
    recent = (history or [])[-MAX_HISTORY_TURNS:]
    return [{"role": h.role, "content": h.content} for h in recent]


async def _retrieve(retriever, condenser, query, history, language):
    """
    Condense query (nếu có history) rồi retrieval có filter, fallback nếu rỗng.

    QUAN TRỌNG: search_query (đã condense) chỉ dùng để RETRIEVE, KHÔNG dùng để hiển thị lại cho khách hay đưa cho
    Generator trả lời -- khách vẫn thấy đúng câu họ gõ, Generator vẫn nhận đúng câu đó (kèm history riêng để tự hiểu
    ngữ cảnh khi trả lời).
    """
    search_query = query
    if ENABLE_CONDENSE and history:
        search_query = await condenser.condense(query, history, language)

    product_filter = extract_product_filter(search_query)

    results = await run_in_threadpool(retriever.search, search_query, None, product_filter)
    if product_filter and not results:
        results = await run_in_threadpool(retriever.search, search_query)

    context = "\n\n---\n\n".join(r["content"] for r in results)
    return results, context, search_query


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, retriever=Depends(get_retriever), generator=Depends(get_generator), condenser=Depends(get_condenser),):
    """
    Endpoint trả lời không streaming — chờ toàn bộ câu trả lời rồi trả về 1 lần.

    Luồng xử lý:
    - Retrieval (có filter + fallback nếu cần) qua _retrieve().
    - Gọi Generator.generate() với context đã retrieve.
    - Đo tổng latency (bao gồm cả retrieval lẫn generation).
    - Trả về ChatResponse gồm câu trả lời, danh sách nguồn tham khảo và latency.
    """
    start = time.perf_counter()
    results, context, _ = await _retrieve(retriever, condenser, req.query, req.history, req.language)

    history_messages = _messages_from_history(req.history)
    gen = await generator.generate(req.query, context, language=req.language, history=history_messages)

    latency_ms = (time.perf_counter() - start) * 1000
    return ChatResponse(answer=gen["answer"], sources=_to_sources(results), latency_ms=latency_ms)


def _sse(event, payload):
    """Format 1 event theo chuẩn Server-Sent Events (SSE)"""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, retriever=Depends(get_retriever), generator=Depends(get_generator), condenser=Depends(get_condenser),):
    """
    Endpoint trả lời dạng streaming qua Server-Sent Events (SSE).

    Khác /chat, endpoint này trả về câu trả lời dần theo từng đoạn nhỏ ngay khi model sinh ra (typewriter effect)
    thay vì chờ toàn bộ rồi trả 1 lần — cải thiện cảm nhận độ trễ (time-to-first-byte) cho người dùng dù tổng thời
    gian xử lý có thể tương đương /chat.

    Luồng SSE events gửi về client theo thứ tự:
    - "sources" — danh sách nguồn tham khảo, gửi ngay sau khi retrieval xong (trước khi bắt đầu generate), để UI có thể
    hiển thị nguồn sớm trong lúc chờ câu trả lời.
    - "delta" — từng đoạn text của câu trả lời, gửi liên tục khi model sinh ra (0 hoặc nhiều event).
    - "error" — chỉ gửi nếu có exception xảy ra trong lúc generate, kèm thông báo lỗi. Nếu xảy ra, dừng stream ngay
    (không gửi "done" sau đó).
    - "done" — gửi khi hoàn tất thành công, kèm tổng latency.
    """
    start = time.perf_counter()
    results, context, _ = await _retrieve(retriever, condenser, req.query, req.history, req.language)
    sources = [s.model_dump() for s in _to_sources(results)]
    history_messages = _messages_from_history(req.history)

    async def event_stream():
        yield _sse("sources", {"sources": sources})
        try:
            async for delta in generator.generate_stream(req.query, context, language=req.language, history=history_messages):
                yield _sse("delta", {"text": delta})
        except Exception as e:
            yield _sse("error", {"message": str(e)})
            return
        yield _sse("done", {"latency_ms": (time.perf_counter() - start) * 1000})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )