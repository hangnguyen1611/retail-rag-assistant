import os
import re
import time
from dotenv import load_dotenv
from groq import AsyncGroq

from config.backend import (
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    GROQ_TEMPERATURE,
)
from src.rag.prompt import build_system_prompt

load_dotenv()

# Heuristic đơn giản để detect ngôn ngữ khi language="auto"
_VI_CHARS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


def detect_language(text):
    """Đoán ngôn ngữ của văn bản bằng heuristic đơn giản dựa trên dấu tiếng Việt"""
    return "vi" if _VI_CHARS.search(text) else "en"


class Generator:
    def __init__(self, model=GROQ_MODEL):
        """
        Khởi tạo Generator với model Groq chỉ định và client async.
        Args:
            model: Tên model Groq để dùng khi generate (mặc định lấy từ config GROQ_MODEL, vd "openai/gpt-oss-120b").
        """
        self.model = model
        api_key = os.getenv("GROQ_API_KEY")
        self._client = AsyncGroq(api_key=api_key) if api_key else None

    def _kwargs(self, system_prompt, query, history=None):
        """
        Build dict kwargs dùng chung cho cả generate() và generate_stream().
        Gom logic build tham số gọi API vào 1 chỗ để generate()/generate_stream() không lặp lại code — chỉ khác nhau ở việc 
        có truyền stream=True hay không.

        Args:
        - system_prompt: Chuỗi system prompt đã build (gồm context, hướng dẫn trả lời, ngôn ngữ...).
        - query: Câu hỏi gốc của người dùng.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages += (history or [])
        messages.append({"role": "user", "content": query})

        kwargs = {
            "model": self.model,
            "temperature": GROQ_TEMPERATURE,
            "max_tokens": GROQ_MAX_TOKENS,
            "messages": messages,
        }
        if "gpt-oss" in self.model or "qwen3" in self.model:
            kwargs["reasoning_effort"] = GROQ_REASONING_EFFORT
        return kwargs
    
    async def generate(self, query, context, language="vi", history=None):
        """
        Sinh câu trả lời đầy đủ (không streaming) từ query + context.
        Gọi Groq API 1 lần, chờ toàn bộ response, đo thời gian xử lý. Nếu câu trả lời bị cắt do vượt max_tokens, 
        thêm ghi chú cảnh báo vào cuối câu trả lời để người dùng biết.

        Args:
        - query: Câu hỏi của người dùng.
        - context: Đoạn văn bản context đã retrieve (từ ChromaDB), sẽ được ghép vào system prompt.
        - language: "vi", "en", hoặc "auto" (tự detect qua detect_language() dựa trên query). Mặc định "vi".
        """
        start = time.perf_counter()
        system_prompt = self._build(query, context, language)

        response = await self._client.chat.completions.create(
            **self._kwargs(system_prompt, query, history)
        )
        answer = response.choices[0].message.content
        if response.choices[0].finish_reason == "length":
            answer += "\n\n_(Câu trả lời bị cắt do giới hạn độ dài.)_"

        latency_ms = (time.perf_counter() - start) * 1000
        return {"answer": answer, "latency_ms": latency_ms}

    def _build(self, query, context, language):
        """Build system prompt, xác định ngôn ngữ thực tế cần dùng và kiểm tra client"""
        lang = detect_language(query) if language == "auto" else language
        if self._client is None:
            raise RuntimeError("GROQ_API_KEY chưa được set (.env). Không thể gọi generator.")
        return build_system_prompt(context, language=lang)

    async def generate_stream(self, query, context, language="vi", history=None):
        """
        Sinh câu trả lời dạng streaming, yield từng đoạn text khi model sinh ra.
        Dùng cho UI cần hiệu ứng "gõ chữ dần" (typewriter effect) thay vì chờ toàn bộ câu trả lời rồi hiển thị 1 lần. 
        Không đo/trả về latency như generate() vì bản chất streaming khiến khái niệm "1 latency tổng" ít ý nghĩa hơn 
        (thường đo time-to-first-token riêng ở tầng gọi hàm này).
        """
        system_prompt = self._build(query, context, language)

        stream = await self._client.chat.completions.create(
            **self._kwargs(system_prompt, query, history), stream=True
        )

        async for chunk in stream:
            choice = chunk.choices[0]
            if choice.delta.content:
                yield choice.delta.content
            if choice.finish_reason == "length":
                yield "\n\n_(Câu trả lời bị cắt do giới hạn độ dài.)_"