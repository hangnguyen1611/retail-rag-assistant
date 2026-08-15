import os

from dotenv import load_dotenv
from groq import AsyncGroq

from config.backend import CONDENSE_MODEL, MAX_HISTORY_TURNS

load_dotenv()

_CONDENSE_PROMPT_VI = """Dựa trên lịch sử hội thoại dưới đây, viết lại CÂU HỎI CUỐI \
thành một câu hỏi ĐỘC LẬP, đầy đủ ngữ cảnh (thay thế đại từ, bổ sung chủ ngữ/tên sản \
phẩm còn thiếu), để một người không đọc lịch sử vẫn hiểu được câu hỏi.

QUY TẮC:
- Nếu câu hỏi cuối đã đầy đủ ngữ cảnh, KHÔNG cần lịch sử để hiểu -> giữ nguyên, không đổi.
- Chỉ bổ sung thông tin THỰC SỰ có trong lịch sử, không suy diễn/bịa thêm.
- Giữ nguyên ngôn ngữ của câu hỏi cuối (không dịch).
- CHỈ trả về câu hỏi đã viết lại, không giải thích, không thêm dấu ngoặc kép.

Lịch sử hội thoại:
{history}

Câu hỏi cuối: {query}
Câu hỏi độc lập:"""

_CONDENSE_PROMPT_EN = """Given the conversation history below, rewrite the LAST QUESTION \
into a STANDALONE question with full context (resolve pronouns, add any missing subject/\
product name), so someone with no access to the history can understand it.

RULES:
- If the last question is already standalone, return it UNCHANGED.
- Only add information that is ACTUALLY present in the history, never invent details.
- Keep the same language as the last question (do not translate).
- Return ONLY the rewritten question, no explanation, no quotes.

Conversation history:
{history}

Last question: {query}
Standalone question:"""


def _format_history(history, max_turns=MAX_HISTORY_TURNS):
    """
    Format list các lượt hội thoại thành text đơn giản cho prompt.
    Dùng chung MAX_HISTORY_TURNS với chat.py._messages_from_history() thay vì hardcode riêng,
    để condense model và generation model luôn thấy cùng độ dài ngữ cảnh trong 1 request.
    """
    recent = history[-max_turns:] if history else []
    lines = []
    for turn in recent:
        role = turn.role if hasattr(turn, "role") else turn.get("role")
        content = turn.content if hasattr(turn, "content") else turn.get("content")
        label = "Khách" if role == "user" else "Trợ lý"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


class QueryCondenser:
    def __init__(self, model=CONDENSE_MODEL):
        self.model = model
        api_key = os.getenv("GROQ_API_KEY")
        self._client = AsyncGroq(api_key=api_key) if api_key else None

    async def condense(self, query, history, language="vi"):
        """Viết lại query thành câu hỏi độc lập nếu có history liên quan"""
        if not history or self._client is None:
            return query

        prompt_template = _CONDENSE_PROMPT_VI if language != "en" else _CONDENSE_PROMPT_EN
        prompt = prompt_template.format(
            history=_format_history(history),
            query=query,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = response.choices[0].message.content.strip().strip('"')
            return rewritten if rewritten else query
        except Exception:
            # Best-effort: lỗi ở bước condense KHÔNG được phép làm hỏng
            # toàn bộ request chat -- fallback về query gốc.
            return query