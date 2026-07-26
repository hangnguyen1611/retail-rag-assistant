"""
generator.py

Gọi Groq API để sinh câu trả lời cuối cùng, dựa trên context đã retrieve.

1. Khởi tạo Groq client (async) bằng GROQ_API_KEY
2. Ghép system prompt (từ prompt.py) + user query
3. Gọi API, trả về answer text
4. Đo latency (dùng cho eval + logging)
"""

import os
import re
import time
from dotenv import load_dotenv
from groq import AsyncGroq

from src.rag.prompt import build_system_prompt

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

# Heuristic đơn giản để detect ngôn ngữ khi language="auto":
# tiếng Việt luôn có ký tự có dấu, tiếng Anh (trong domain này) thì không.
_VI_CHARS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    return "vi" if _VI_CHARS.search(text) else "en"


class Generator:
    def __init__(self, model: str = GROQ_MODEL):
        self.model = model
        api_key = os.getenv("GROQ_API_KEY")
        self._client = AsyncGroq(api_key=api_key) if api_key else None

    async def generate(self, query: str, context: str, language: str = "vi") -> dict:
        start = time.perf_counter()

        lang = detect_language(query) if language == "auto" else language
        system_prompt = build_system_prompt(context, language=lang)

        if self._client is None:
            raise RuntimeError(
                "GROQ_API_KEY chưa được set (.env). Không thể gọi generator."
            )

        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=GROQ_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        answer = response.choices[0].message.content

        latency_ms = (time.perf_counter() - start) * 1000
        return {"answer": answer, "latency_ms": latency_ms}
