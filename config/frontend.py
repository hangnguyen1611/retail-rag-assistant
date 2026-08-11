import os

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)

CHAT_API = f"{BACKEND_URL}/chat"
STREAM_API = f"{BACKEND_URL}/chat/stream"

TIMEOUT = int(
    os.getenv(
        "FRONTEND_TIMEOUT",
        "45"
    )
)

APP_NAME = "Fashion Assistant"
PAGE_TITLE = APP_NAME
PAGE_ICON = "🛍️"
LAYOUT = "wide"

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🛍️"

WELCOME_MESSAGE = """
Xin chào!
Mình là trợ lý AI của cửa hàng thời trang.
Mình có thể giúp gì cho bạn?
"""

STARTER_QUESTIONS = [
    {"text": "Áo sơ mi trắng còn size M không?", "icon": "👕", "group": "product"},
    {"text": "Áo khoác dưới 500.000đ", "icon": "🧥", "group": "product"},
    {"text": "Giày thể thao nam màu đen", "icon": "👟", "group": "product"},
    {"text": "Đổi trả trong bao nhiêu ngày?", "icon": "↻", "group": "policy"},
    {"text": "Bao nhiêu tiền thì được miễn phí vận chuyển?", "icon": "⇢", "group": "policy"},
    {"text": "Chính sách bảo hành như thế nào?", "icon": "✓", "group": "policy"},
]

POLICY_LABELS = {
    "return_policy.md": "Chính sách đổi trả",
    "shipping_policy.md": "Chính sách vận chuyển",
    "promotion_policy.md": "Chính sách khuyến mãi",
    "warranty_policy.md": "Chính sách bảo hành",
    "size_guide.md": "Bảng size",
}

PRIMARY = "#7C3AED"          # tím chủ đạo
PRIMARY_DARK = "#5B21B6"
GRADIENT_FROM = "#7C3AED"
GRADIENT_TO = "#EC4899"

ACCENT_PRODUCT = "#2563EB"   # xanh dương — thông tin sản phẩm
ACCENT_POLICY = "#059669"    # xanh lá — chính sách
ACCENT_OTHER = "#F59E0B"     # cam — gợi ý khác / linh tinh

BACKGROUND = "#F6F5FC"
CARD = "#FFFFFF"
BORDER = "#E4E0F7"
TEXT = "#1F2033"
SUBTEXT = "#6B6C89"

SUCCESS = "#16A34A"
WARNING = "#F59E0B"
ERROR = "#DC2626"

GROUP_COLORS = {
    "product": ACCENT_PRODUCT,
    "policy": ACCENT_POLICY,
    "other": ACCENT_OTHER,
}

GROUP_LABELS = {
    "product": "🧢 Sản phẩm",
    "policy": "🗊  Chính sách",
    "other": "✨ Khác",
}