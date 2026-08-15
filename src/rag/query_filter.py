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

# Khớp với articleType gốc (tiếng Anh, viết thường) lưu trong metadata article_type_lower, không phải type_vi
# 1 cụm từ có thể khớp nhiều articleType (vd "áo khoác" khớp cả "jackets" lẫn "rain jacket") -> dùng $in để filter theo danh sách.
# Không cố phủ hết 142 articleType trong catalog, chỉ cover các nhóm quần áo/giày/túi phổ biến nhất
_ARTICLE_TYPE_ALIASES = {
    # Áo khoác ngoài
    "áo ghi lê": ["waistcoat"],
    "áo gile": ["waistcoat"],
    "áo vest": ["waistcoat"],
    "waistcoat": ["waistcoat"],
    "bộ vest": ["suits"],
    "suits": ["suits"],
    "suit": ["suits"],
    "áo blazer": ["blazers"],
    "blazers": ["blazers"],
    "blazer": ["blazers"],
    "áo khoác mưa": ["rain jacket"],
    "rain jacket": ["rain jacket"],
    "áo khoác nehru": ["nehru jackets"],
    "nehru jacket": ["nehru jackets"],
    "áo khoác": ["jackets", "rain jacket", "nehru jackets"],
    "jackets": ["jackets", "rain jacket", "nehru jackets"],
    "jacket": ["jackets", "rain jacket", "nehru jackets"],
    # Áo
    "áo sơ mi": ["shirts"],
    "shirts": ["shirts"],
    "shirt": ["shirts"],
    "áo thun": ["tshirts"],
    "t-shirt": ["tshirts"],
    "tshirts": ["tshirts"],
    "tshirt": ["tshirts"],
    "áo len": ["sweaters"],
    "sweaters": ["sweaters"],
    "sweater": ["sweaters"],
    "áo nỉ": ["sweatshirts"],
    "sweatshirts": ["sweatshirts"],
    "sweatshirt": ["sweatshirts"],
    "áo kiểu": ["tops"],
    "tops": ["tops"],
    # Quần
    "quần jean co giãn": ["jeggings"],
    "jeggings": ["jeggings"],
    "quần jean": ["jeans"],
    "jeans": ["jeans"],
    "quần short": ["shorts"],
    "shorts": ["shorts"],
    "quần tây": ["trousers"],
    "trousers": ["trousers"],
    "trouser": ["trousers"],
    "quần legging": ["leggings"],
    "leggings": ["leggings"],
    "legging": ["leggings"],
    # Váy/đầm
    "váy đầm": ["dresses"],
    "đầm": ["dresses"],
    "dresses": ["dresses"],
    "dress": ["dresses"],
    "váy": ["skirts"],
    "skirts": ["skirts"],
    "skirt": ["skirts"],
    # Giày
    "giày sneaker": ["casual shoes"],
    "giày thể thao": ["sports shoes"],
    "sports shoes": ["sports shoes"],
    "giày tây": ["formal shoes"],
    "formal shoes": ["formal shoes"],
    "giày cao gót": ["heels"],
    "heels": ["heels"],
    "giày bệt": ["flats"],
    "flats": ["flats"],
    "dép xỏ ngón": ["flip flops"],
    "flip flops": ["flip flops"],
    "sandal": ["sandals"],
    "sandals": ["sandals"],
    # Túi & phụ kiện
    "túi xách": ["handbags"],
    "handbags": ["handbags"],
    "handbag": ["handbags"],
    "balo": ["backpacks"],
    "backpacks": ["backpacks"],
    "backpack": ["backpacks"],
    "ví": ["wallets"],
    "wallets": ["wallets"],
    "wallet": ["wallets"],
    "thắt lưng": ["belts"],
    "belts": ["belts"],
    "belt": ["belts"],
    "kính râm": ["sunglasses"],
    "sunglasses": ["sunglasses"],
    "đồng hồ": ["watches"],
    "watches": ["watches"],
    "watch": ["watches"],
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
    """
    Trích điều kiện lọc giá từ câu hỏi, trả về ChromaDB where-clause.

    "dưới X" -> $lt (loại trừ đúng giá X, đúng nghĩa "dưới").
    "trên X" -> $gt (loại trừ đúng giá X, đúng nghĩa "trên", tương tự "dưới").
    "từ X"   -> $gte (bao gồm đúng giá X, vì "từ X" nghĩa là "bắt đầu từ X trở lên" —
    khác "trên X", "từ" không loại trừ giá trị biên).
    """
    m = _PRICE_RE.search(q)
    if not m:
        return None
    value = _to_vnd(m.group("num"), m.group("unit"))
    if value is None:
        return None
    direction = m.group("dir").lower()
    if direction in ("dưới", "duoi"):
        return {"price": {"$lt": value}}
    if direction in ("từ", "tu"):
        return {"price": {"$gte": value}}
    return {"price": {"$gt": value}}


# Các âm tiết màu ngắn, trùng với từ tiếng Việt phổ biến khác ngoài nghĩa màu sắc
_AMBIGUOUS_COLOR_PHRASES = {"cam", "be", "kem", "bạc", "vàng", "hồng"}

_COLOR_CUE_RE = re.compile(r"\b(màu|color|colour)\b")


def _parse_color(q):
    """
    Trích điều kiện lọc màu sắc từ câu hỏi bằng keyword matching.
    Ưu tiên khớp cụm từ DÀI trước cụm NGẮN (Vd "xanh navy" phải được thử khớp trước "xanh lá"/"xanh dương" đơn lẻ nếu chúng có phần chung),
    để tránh khớp nhầm phần đầu của 1 cụm dài với 1 màu đơn giản hơn.

    Dùng word boundary (\\b) như _parse_gender(), không phải substring thuần — giúp tránh khớp nhầm khi
    1 màu là substring bên trong 1 từ KHÔNG có dấu cách (vd "cam" bên trong "camera").
    Nhưng \\b không giúp được khi màu ngắn trùng với 1 từ tiếng Việt khác ĐỨNG RIÊNG (vd "cam" trong
    "cam kết" cũng là 1 "từ" hoàn chỉnh theo boundary) -> với các phrase trong _AMBIGUOUS_COLOR_PHRASES,
    bắt buộc phải có từ khoá "màu"/"color" xuất hiện đâu đó trong câu mới nhận, giống cách _parse_size()
    bắt buộc từ khoá "size" đứng trước token.
    """
    # Ưu tiên cụm dài trước
    for phrase in sorted(_COLOR_VI_TO_EN, key=len, reverse=True):
        if not re.search(rf"\b{re.escape(phrase)}\b", q):
            continue
        if phrase in _AMBIGUOUS_COLOR_PHRASES and not _COLOR_CUE_RE.search(q):
            continue
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


def _parse_article_type(q):
    """
    Trích điều kiện lọc category (articleType) từ câu hỏi bằng keyword matching.

    Giống _parse_color(): ưu tiên khớp cụm DÀI trước cụm NGẮN (vd "áo khoác mưa" phải thử khớp
    trước "áo khoác" để không bị cụm ngắn nuốt mất, dẫn tới lọc sai/quá rộng). Cũng dùng word
    boundary (\\b) như _parse_gender()/_parse_color() để tránh khớp nhầm vào giữa 1 từ khác.

    1 cụm có thể ánh xạ tới NHIỀU articleType (vd "áo khoác" -> jackets + rain jacket + nehru
    jackets), nên dùng $in thay vì so bằng trực tiếp.
    """
    for phrase in sorted(_ARTICLE_TYPE_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            types = _ARTICLE_TYPE_ALIASES[phrase]
            if len(types) == 1:
                return {"article_type_lower": types[0]}
            return {"article_type_lower": {"$in": types}}
    return None


def extract_product_filter(query):
    """
    Trả về ChromaDB `where` clause hoặc None nếu không tìm thấy điều kiện nào. Best-effort, không đảm bảo bắt hết mọi cách diễn đạt.

    Đây là điểm vào (entry point) duy nhất của module, được gọi bởi tầng API (api/routers/chat.py) trước khi gọi Retriever.search(). 
    Gộp cả 5 loại điều kiện (category, giá, màu, giới tính, size) nếu tìm thấy, kết hợp bằng $and nếu có nhiều hơn 1. 
    Nơi gọi hàm này bắt buộc phải có cơ chế fallback về tìm kiếm KHÔNG filter nếu kết quả filter trả về rỗng — vì đây chỉ 
    là parse heuristic, có thể sai/thiếu, không nên để 1 lần parse sai làm mất hoàn toàn kết quả tìm kiếm.
    """
    q = (query or "").lower()

    clauses = [
        c for c in (
            _parse_article_type(q),
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