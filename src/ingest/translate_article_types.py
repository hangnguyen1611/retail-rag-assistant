"""
Dịch tự động các articleType chưa có trong ARTICLE_TYPE_VI (clean_products.py)
bằng Groq, ghi kết quả ra data/processed/article_type_vi_auto.json.

CHẠY 1 LẦN, không phải mỗi lần build pipeline (không tự động gọi trong clean_products.py):
    python -m src.ingest.translate_article_types

Cần GROQ_API_KEY trong .env (dùng chung với phần còn lại của project).
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from config.backend import CLEAN_DATA_PATH, DATA_DIR, GROQ_MODEL
from src.ingest.clean_products import ARTICLE_TYPE_VI

load_dotenv()

OUT_PATH = Path(DATA_DIR) / "processed" / "article_type_vi_auto.json"
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", GROQ_MODEL)

_PROMPT = """Bạn là chuyên gia bán lẻ thời trang tại Việt Nam. Dịch các tên loại sản phẩm 
(article type) tiếng Anh dưới đây sang tiếng Việt, theo đúng phong cách các ví dụ đã dịch sẵn.

VÍ DỤ đã dịch sẵn (giữ đúng phong cách này -- ngắn gọn, 2-4 từ, tên gọi thông dụng trong bán lẻ 
thời trang VN, không giải thích thêm):
{examples}

LƯU Ý CÁC TỪ DỄ NHẦM NGÀNH (toàn bộ danh sách thuộc ngành thời trang/mỹ phẩm, KHÔNG phải văn 
phòng phẩm/điện tử/đồ gia dụng -- một số từ tiếng Anh có nghĩa khác hẳn ở ngành khác):
- "Toner" -> "Nước hoa hồng" (mỹ phẩm dưỡng da), KHÔNG PHẢI "Mực in" (máy in)
- "Compact" -> "Phấn phủ nén" (mỹ phẩm), KHÔNG PHẢI "kem" hay đồ nhỏ gọn nói chung
- "Foundation" -> "Kem nền" (mỹ phẩm), KHÔNG PHẢI "nền móng" (xây dựng)
- "Concealer" -> "Kem che khuyết điểm", KHÔNG PHẢI "vật che giấu" nói chung
- "Blush" -> "Má hồng" (mỹ phẩm), KHÔNG PHẢI đỏ mặt/ửng đỏ
Nếu gặp từ khác có khả năng đa nghĩa tương tự (một nghĩa thuộc mỹ phẩm/thời trang, nghĩa khác 
thuộc ngành hoàn toàn khác), LUÔN chọn nghĩa thuộc mỹ phẩm/thời trang.

QUY TẮC BẮT BUỘC:
1. Dịch ĐÚNG NGHĨA, không đoán/bịa. Ví dụ "Ring" là NHẪN (đeo ngón tay), không phải vòng cổ.
   "Skirts" là VÁY, không phải quần. "Suits" là BỘ VEST, không phải áo ba lỗ.
2. Với trang phục/phụ kiện truyền thống Nam Á KHÔNG có tên tiếng Việt tương đương (Kurta, Kurti,
   Dupatta, Salwar, Saree, Lehenga, Churidar, Patiala...): GIỮ NGUYÊN tên gốc (phiên âm/mượn từ),
   KHÔNG gán nhầm thành "Áo dài" hay tên trang phục Việt Nam nào khác -- đây là lỗi văn hóa nghiêm
   trọng, tuyệt đối tránh. Ví dụ: "Kurtas" -> "Áo Kurta", "Dupatta" -> "Khăn Dupatta".
3. Nếu không chắc nghĩa chính xác, dịch sát nghĩa đen hơn là đoán mò.

Trả lời CHÍNH XÁC dưới dạng JSON object hợp lệ, key là tên tiếng Anh gốc (giữ nguyên y hệt), 
value là bản dịch tiếng Việt. KHÔNG thêm text/markdown/giải thích nào khác ngoài JSON object đó.

Dịch các loại sau ({n} loại):
{targets}"""


def _load_missing_article_types():
    """Đọc clean data, trả về list articleType chưa có trong ARTICLE_TYPE_VI (dict dịch tay)"""
    import pandas as pd

    df = pd.read_csv(CLEAN_DATA_PATH)
    all_types = sorted(df["articleType"].dropna().unique().tolist())
    return [t for t in all_types if t not in ARTICLE_TYPE_VI]


def _call_groq_json(client, batch, examples):
    """1 lần gọi Groq, trả về dict {en: vi} đã parse từ JSON hoặc {} nếu response không phải JSON hợp lệ."""
    prompt = _PROMPT.format(examples=examples, n=len(batch), targets="\n".join(f"- {t}" for t in batch))

    response = client.chat.completions.create(
        model=TRANSLATE_MODEL,
        temperature=0,
        max_tokens=1500,
        response_format={"type": "json_object"},  # ép model trả JSON hợp lệ, tránh lỗi parse/markdown fence
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [LỖI PARSE JSON] Response không parse được, bỏ qua batch này:\n{raw[:300]}")
        return {}


def translate_batch(client, missing_types, batch_size=15):
    """
    Dịch theo batch nhỏ (15 loại/lần, nhỏ hơn trước) để giảm rủi ro model bỏ sót/dịch sai khi prompt dài.
    Dùng response_format=json_object thay vì "mỗi dòng 1 kết quả" - JSON object có key rõ ràng nên dễ phát
    hiện thiếu key (không như đếm dòng, dễ lệch thứ tự/số lượng).
    Batch nào thiếu key thì retry đúng phần thiếu đó 1 lần.
    """
    examples = "\n".join(f"- {en} -> {vi}" for en, vi in list(ARTICLE_TYPE_VI.items())[:8])
    result = {}
    still_missing = []

    for i in range(0, len(missing_types), batch_size):
        batch = missing_types[i : i + batch_size]
        translated = _call_groq_json(client, batch, examples)
        result.update({k: v for k, v in translated.items() if k in batch})

        missed = [t for t in batch if t not in result]
        if missed:
            print(f"  [THIẾU] Batch {i}: thiếu {len(missed)}/{len(batch)} key -- retry riêng phần này")
            retry = _call_groq_json(client, missed, examples)
            result.update({k: v for k, v in retry.items() if k in missed})
            still_missing.extend(t for t in missed if t not in result)

    return result, still_missing


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Thiếu GROQ_API_KEY trong .env -- cần để gọi Groq dịch.")

    missing = _load_missing_article_types()
    print(f"Tìm thấy {len(missing)} articleType chưa có trong ARTICLE_TYPE_VI:")
    print(", ".join(missing))
    print()

    client = Groq(api_key=api_key)
    translated, still_missing = translate_batch(client, missing)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Đã dịch {len(translated)}/{len(missing)} loại -> {OUT_PATH}")
    if still_missing:
        print(f"[CẦN DỊCH TAY] {len(still_missing)} loại vẫn chưa dịch được: {sorted(still_missing)}")


if __name__ == "__main__":
    main()