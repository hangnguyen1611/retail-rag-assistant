"""
Unit test cho src/rag/query_filter.py

Chạy: pytest tests/test_query_filter.py -v

Test riêng từng hàm pure (_to_vnd, _parse_price, _parse_color, _parse_gender, _parse_size) và hàm entry point (extract_product_filter). 
Không gọi Groq, không gọi ChromaDB, không cần .env — chạy được offline trong CI.
"""
import pytest

from src.rag.query_filter import (
    _parse_article_type,
    _parse_color,
    _parse_gender,
    _parse_price,
    _parse_size,
    _to_vnd,
    extract_product_filter,
)


# ---------------------------------------------------------------------------
# _to_vnd: parse số tiền VN (k, triệu, dấu chấm phân cách hàng nghìn)
# ---------------------------------------------------------------------------
class TestToVnd:
    def test_k_suffix(self):
        assert _to_vnd("500", "k") == 500_000

    def test_nghin_suffix(self):
        assert _to_vnd("300", "nghìn") == 300_000

    def test_trieu_suffix(self):
        assert _to_vnd("5", "triệu") == 5_000_000

    def test_dotted_thousand_separator_no_unit(self):
        # "500.000" không có unit -> dấu chấm là phân cách hàng nghìn kiểu VN
        assert _to_vnd("500.000", None) == 500_000

    def test_bare_number_no_unit(self):
        assert _to_vnd("500", None) == 500

    def test_decimal_k_suffix(self):
        # "1.5" + "k" -> 1500 (ở đây dấu chấm là thập phân vì CÓ unit đi kèm)
        assert _to_vnd("1.5", "k") == 1_500

    def test_comma_as_decimal(self):
        # hàm tự đổi "," -> "." trước khi parse
        assert _to_vnd("1,5", "k") == 1_500

    def test_invalid_string_returns_none(self):
        assert _to_vnd("abc", None) is None

    def test_invalid_dotted_string_returns_none(self):
        assert _to_vnd("abc.def", None) is None


# ---------------------------------------------------------------------------
# _parse_price: trích where-clause giá từ câu hỏi tiếng Việt
# ---------------------------------------------------------------------------
class TestParsePrice:
    def test_duoi_with_k(self):
        assert _parse_price("áo khoác dưới 500k") == {"price": {"$lt": 500_000}}

    def test_tren_with_trieu(self):
        assert _parse_price("giày trên 1 triệu") == {"price": {"$gt": 1_000_000}}

    def test_no_diacritics_duoi(self):
        # "duoi" không dấu vẫn phải khớp (khách gõ không dấu rất phổ biến)
        assert _parse_price("ao duoi 300k") == {"price": {"$lt": 300_000}}

    def test_no_price_mentioned(self):
        assert _parse_price("áo sơ mi màu đen") is None

    def test_tu_treated_as_gte(self):
        # "từ X" bao gồm đúng giá X (>=), khác "trên X" (>) loại trừ đúng giá X
        assert _parse_price("áo từ 200k") == {"price": {"$gte": 200_000}}

    def test_tren_treated_as_gt(self):
        assert _parse_price("áo trên 200k") == {"price": {"$gt": 200_000}}


# ---------------------------------------------------------------------------
# _parse_color: ưu tiên cụm dài trước cụm ngắn
# ---------------------------------------------------------------------------
class TestParseColor:
    def test_simple_color(self):
        assert _parse_color("áo màu đen size m") == {"base_colour_lower": "black"}

    def test_long_phrase_priority_over_short(self):
        # "xanh navy" phải thắng "xanh dương"/"xanh lá" dù cùng bắt đầu bằng "xanh"
        assert _parse_color("quần xanh navy") == {"base_colour_lower": "navy blue"}

    def test_xanh_la_not_confused_with_xanh_duong(self):
        assert _parse_color("áo xanh lá") == {"base_colour_lower": "green"}

    def test_no_color_mentioned(self):
        assert _parse_color("áo size m giá dưới 300k") is None

    def test_ambiguous_color_without_cue_not_matched(self):
        # "cam" trong "cam kết" không phải màu -> không có từ khoá "màu" đi kèm nên không nhận
        assert _parse_color("tôi cam kết mua sản phẩm này") is None

    def test_ambiguous_color_with_cue_matched(self):
        assert _parse_color("áo khoác màu cam size m") == {"base_colour_lower": "orange"}

    def test_ambiguous_color_kem_without_cue_not_matched(self):
        # "kem" ở đây là món ăn (ice cream), không phải màu kem
        assert _parse_color("đi ăn kem với bạn") is None

    def test_unambiguous_color_without_cue_still_matched(self):
        # màu dài/rõ nghĩa (không nằm trong _AMBIGUOUS_COLOR_PHRASES) vẫn nhận dù không có từ khoá "màu"
        assert _parse_color("áo đen size m") == {"base_colour_lower": "black"}


# ---------------------------------------------------------------------------
# _parse_gender: dùng word boundary để tránh khớp nhầm substring
# ---------------------------------------------------------------------------
class TestParseGender:
    def test_nam(self):
        assert _parse_gender("áo nam size l") == {"gender_lower": "men"}

    def test_nu(self):
        assert _parse_gender("váy nữ màu đen") == {"gender_lower": "women"}

    def test_be_trai_priority_over_nothing_else(self):
        assert _parse_gender("quần bé trai") == {"gender_lower": "boys"}

    def test_no_gender_mentioned(self):
        assert _parse_gender("áo màu đen size m") is None


# ---------------------------------------------------------------------------
# _parse_size: chỉ khớp khi có "size <token>" tường minh
# ---------------------------------------------------------------------------
class TestParseSize:
    def test_letter_size_uppercased(self):
        assert _parse_size("áo size m màu đen") == {"size": "M"}

    def test_numeric_size_kept_as_is(self):
        assert _parse_size("giày size 40") == {"size": "40"}

    def test_no_size_keyword_no_match(self):
        # số "40" xuất hiện nhưng KHÔNG có từ "size" đứng trước -> không khớp
        assert _parse_size("áo giá 40k") is None

    def test_size_token_not_in_whitelist(self):
        # "size 99" không nằm trong _SIZE_TOKENS -> None, tránh khớp rác
        assert _parse_size("giày size 99") is None


# ---------------------------------------------------------------------------
# _parse_article_type: category filter, khớp theo articleType gốc (tiếng Anh, lowercase)
# ---------------------------------------------------------------------------
class TestParseArticleType:
    def test_exact_english_term(self):
        assert _parse_article_type("waistcoat") == {"article_type_lower": "waistcoat"}

    def test_vietnamese_colloquial_alias(self):
        # "áo ghi lê" là từ tiếng Việt thông dụng cho waistcoat -- đây chính là câu hỏi
        # thực tế từng bị miss (không match string nào trong data) trước khi có alias này.
        assert _parse_article_type("áo ghi lê") == {"article_type_lower": "waistcoat"}

    def test_ambiguous_vest_term_maps_to_waistcoat_not_suits(self):
        # "áo vest" dễ gây nhầm với "suits" (bộ vest) -- phải map đúng về waistcoat theo
        # _ARTICLE_TYPE_ALIASES, không lẫn sang suits.
        assert _parse_article_type("áo vest") == {"article_type_lower": "waistcoat"}

    def test_multi_type_alias_returns_in_clause(self):
        # "áo khoác" ánh xạ tới nhiều articleType (jackets, rain jacket, nehru jackets)
        # -> phải dùng $in, không phải so bằng trực tiếp.
        result = _parse_article_type("áo khoác")
        assert result == {
            "article_type_lower": {"$in": ["jackets", "rain jacket", "nehru jackets"]}
        }

    def test_longer_phrase_priority_over_shorter(self):
        # "áo khoác mưa" phải khớp cụm dài "áo khoác mưa" (-> chỉ rain jacket), KHÔNG bị cụm
        # ngắn "áo khoác" (-> cả 3 loại) nuốt mất trước.
        assert _parse_article_type("áo khoác mưa màu đen") == {
            "article_type_lower": "rain jacket"
        }

    def test_no_category_mentioned(self):
        assert _parse_article_type("màu đen size m dưới 300k") is None


# ---------------------------------------------------------------------------
# extract_product_filter: entry point, gộp nhiều điều kiện bằng $and
# ---------------------------------------------------------------------------
class TestExtractProductFilter:
    def test_empty_query_returns_none(self):
        assert extract_product_filter("") is None

    def test_none_query_returns_none(self):
        assert extract_product_filter(None) is None

    def test_single_condition_not_wrapped_in_and(self):
        result = extract_product_filter("áo màu đen")
        assert result == {"base_colour_lower": "black"}

    def test_multiple_conditions_wrapped_in_and(self):
        result = extract_product_filter("áo nam màu đen dưới 500k size l")
        assert "$and" in result
        assert {"gender_lower": "men"} in result["$and"]
        assert {"base_colour_lower": "black"} in result["$and"]
        assert {"price": {"$lt": 500_000}} in result["$and"]
        assert {"size": "L"} in result["$and"]

    def test_query_is_lowercased_before_parsing(self):
        # Màu/giới tính viết hoa vẫn phải khớp vì hàm tự lowercase trước
        # Không assert đúng thứ tự trong $and (order = price, color, gender, size theo thứ tự gọi trong extract_product_filter)
        # -> so sánh bằng set để test, không vỡ nếu thứ tự nội bộ đổi
        result = extract_product_filter("Áo NAM Màu Đen")
        assert result is not None
        assert {"gender_lower": "men"} in result["$and"]
        assert {"base_colour_lower": "black"} in result["$and"]

    def test_irrelevant_query_returns_none(self):
        assert extract_product_filter("chính sách đổi trả như thế nào") is None