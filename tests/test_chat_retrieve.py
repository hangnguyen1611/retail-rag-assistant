"""
Unit test cho _has_product_hit trong api/routers/chat.py.
Chạy:
    pytest tests/test_chat_retrieve.py -v

_has_product_hit() tách riêng việc kiểm tra product-slot để fallback quyết định đúng. 
Test này không cần Groq/ChromaDB thật, chỉ cần list dict giả lập đúng shape mà retriever.search() trả về.
"""
from api.routers.chat import _has_product_hit


def _hit(doc_type):
    """Tạo 1 hit giả lập tối thiểu đúng shape mà retriever.search() trả về."""
    return {"id": "x", "content": "...", "metadata": {"doc_type": doc_type}, "score": 0.9}


class TestHasProductHit:
    def test_only_policy_hits_returns_false(self):
        # Đây chính là case bug thật: product_filter sai/quá chặt -> product ra 0 nhưng policy-slot vẫn có 2 hit
        # -> merged results không rỗng
        results = [_hit("policy"), _hit("policy")]
        assert _has_product_hit(results) is False

    def test_mixed_hits_returns_true(self):
        results = [_hit("policy"), _hit("product"), _hit("policy")]
        assert _has_product_hit(results) is True

    def test_only_product_hits_returns_true(self):
        results = [_hit("product"), _hit("product"), _hit("product")]
        assert _has_product_hit(results) is True

    def test_empty_results_returns_false(self):
        assert _has_product_hit([]) is False

    def test_missing_metadata_key_does_not_crash(self):
        # metadata.get("doc_type") an toàn với dict rỗng/thiếu key, không raise.
        results = [{"id": "x", "content": "...", "metadata": {}, "score": 0.5}]
        assert _has_product_hit(results) is False