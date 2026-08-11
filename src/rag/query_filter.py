import re

# Khớp với BASE_COLOUR gốc trong data/processed/products.csv (COLOR_VI trong clean_products.py, đảo chiều). 
# Chỉ cần đưa từ khoá tiếng Việt phổ biến nhất cho mỗi màu, không cần đầy đủ mọi biến thể.
_COLOR_VI_TO_EN = {
    "đen": "black",
    "trắng": "white",
    "xanh dương": "blue",
    "xanh biển": "blue",
    "nâu": "brown",
    "xám thép": "steel",
    "xám": "grey",
    "đỏ mận": "maroon",
    "đỏ": "red",
    "xanh lá": "green",
    "hồng": "pink",
    "xanh navy": "navy blue",
    "navy": "navy blue",
    "tím": "purple",
    "bạc": "silver",
    "vàng gold": "gold",
    "vàng": "yellow",
    "be": "beige",
    "cam": "orange",
    "xanh olive": "olive",
    "kem": "cream",
}

_GENDER_VI_TO_EN = {
    "bé trai": "boys",
    "bé gái": "girls",
    "nam": "men",
    "nữ": "women",
    "unisex": "unisex",
}

_SIZE_TOKENS = {"s", "m", "l", "xl", "xxl", "37", "38", "39", "40", "41", "42"}


def _to_vnd(number_str, unit=None):
    """
    Chuyển chuỗi số + đơn vị (nếu có) thành giá trị VND nguyên.
    Xử lý 3 dạng viết số tiền phổ biến trong tiếng Việt:
    - "500" + "k"      -> 500_000     (k = nghìn)
    - "5" + "triệu"    -> 5_000_000
    - "500.000" + None -> 500_000     (dấu chấm là phân cách hàng nghìn, không phải thập phân, khi không có đơn vị đi kèm)
    - "500" + None     -> 500         (số trần, không đơn vị, không dấu chấm -> hiểu là chính giá trị đó)

    Dấu chấm/phẩy trong số tiếng Việt có thể là phân cách hàng nghìn (500.000 = năm trăm nghìn) thay vì phân cách thập phân
    như tiếng Anh (500.000 = năm trăm phẩy không không không). Hàm này ưu tiên diễn giải kiểu VN khi không có đơn vị rõ ràng.
    """
    number_str = number_str.strip().replace(",", ".")

    if unit is None and "." in number_str:
        try:
            return int(number_str.replace(".", ""))
        except ValueError:
            return None

    try:
        base = float(number_str)
    except ValueError:
        return None

    if unit in ("k", "nghìn", "ngàn"):
        return int(base * 1_000)
    if unit == "triệu":
        return int(base * 1_000_000)
    return int(base)


_PRICE_RE = re.compile(
    r"(?P<dir>dưới|duoi|trên|tren|từ|tu)\s*"
    r"(?P<num>[\d]+(?:[.,]\d+)?)\s*"
    r"(?P<unit>k|nghìn|ngàn|triệu)?",
    re.IGNORECASE,
)


def _parse_price(q):
    """Trích điều kiện lọc giá từ câu hỏi, trả về ChromaDB where-clause."""
    m = _PRICE_RE.search(q)
    if not m:
        return None
    value = _to_vnd(m.group("num"), m.group("unit"))
    if value is None:
        return None
    direction = m.group("dir").lower()
    if direction in ("dưới", "duoi"):
        return {"price": {"$lt": value}}
    return {"price": {"$gt": value}}


def _parse_color(q):
    """
    Trích điều kiện lọc màu sắc từ câu hỏi bằng keyword matching.
    Ưu tiên khớp cụm từ DÀI trước cụm NGẮN (Vd "xanh navy" phải được thử khớp trước "xanh lá"/"xanh dương" đơn lẻ nếu chúng có phần chung),
    để tránh khớp nhầm phần đầu của 1 cụm dài với 1 màu đơn giản hơn.
    """
    # Ưu tiên cụm dài trước
    for phrase in sorted(_COLOR_VI_TO_EN, key=len, reverse=True):
        if phrase in q:
            return {"base_colour_lower": _COLOR_VI_TO_EN[phrase]}
    return None


def _parse_gender(q):
    """
    Trích điều kiện lọc giới tính từ câu hỏi bằng keyword matching có word boundary.

    Khác _parse_color(), dùng \\b (word boundary) để tránh khớp nhầm substring bên trong 1 từ khác (VD tránh "nam" khớp nhầm vào giữa 1
    từ tiếng Việt khác chứa "nam" như 1 phần âm tiết).
    """
    for phrase in sorted(_GENDER_VI_TO_EN, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return {"gender_lower": _GENDER_VI_TO_EN[phrase]}
    return None


def _parse_size(q):
    """
    Trích điều kiện lọc size từ câu hỏi, chỉ khớp khi có từ khoá "size" đứng trước.

    Khác với color/gender (chỉ cần từ khoá xuất hiện đâu đó), size yêu cầu cấu trúc rõ ràng "size <giá trị>" để tránh khớp nhầm các số/
    chữ cái đơn lẻ không liên quan xuất hiện tình cờ trong câu (vd số trong giá tiền hoặc chữ cái đơn lẻ ngẫu nhiên).
    """
    m = re.search(r"\bsize\s+([a-zA-Z0-9]+)\b", q)
    if not m:
        return None
    token = m.group(1).lower()
    if token not in _SIZE_TOKENS:
        return None
    return {"size": token.upper() if token.isalpha() else token}


def extract_product_filter(query):
    """
    Trả về ChromaDB `where` clause hoặc None nếu không tìm thấy điều kiện nào. Best-effort, KHÔNG đảm bảo bắt hết mọi cách diễn đạt.

    Đây là điểm vào (entry point) duy nhất của module, được gọi bởi tầng API (api/routers/chat.py) trước khi gọi Retriever.search(). 
    Gộp cả 4 loại điều kiện (giá, màu, giới tính, size) nếu tìm thấy, kết hợp bằng $and nếu có nhiều hơn 1. 
    QUAN TRỌNG: nơi gọi hàm này bắt buộc phải có cơ chế fallback về tìm kiếm KHÔNG filter nếu kết quả filter trả về rỗng — vì đây chỉ 
    là parse heuristic, có thể sai/thiếu, không nên để 1 lần parse sai làm mất hoàn toàn kết quả tìm kiếm.
    """
    q = (query or "").lower()

    clauses = [
        c for c in (
            _parse_price(q),
            _parse_color(q),
            _parse_gender(q),
            _parse_size(q),
        ) if c
    ]

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}