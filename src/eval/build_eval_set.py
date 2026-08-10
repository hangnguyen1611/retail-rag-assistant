"""
Build data/eval/eval_set.csv = (nửa product SINH TỪ DATA) + (nửa manual đã verify).

Ba loại câu product:
    strict: Có brand/tên trong câu hỏi -> đúng 1 sản phẩm khớp.
            Dùng để đo recall@1 một cách sắc nét.
    loose: Chỉ mô tả thuộc tính (loại/giới tính/màu) -> nhiều sản phẩm khớp. 
            Giống cách khách thật hỏi.
    product_not_found: Tổ hợp thuộc tính không tồn tại trong catalog. 
            Test hallucination trực tiếp: trợ lý phải nói không có.
"""

import csv
import os
import sys
from collections import Counter
from itertools import cycle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.backend import EVAL_SET_PATH, PROCESSED_PRODUCTS_PATH, SEED

MANUAL_SET_PATH = "data/eval/eval_set_manual.csv"
N_STRICT = int(os.getenv("EVAL_N_STRICT", "30"))
N_LOOSE = int(os.getenv("EVAL_N_LOOSE", "30"))
N_NOT_FOUND = int(os.getenv("EVAL_N_NOT_FOUND", "15"))

FIELDNAMES = [
    "id",
    "question",
    "language",
    "category",
    "strictness",
    "expected_doc_type",
    "product_filter",
    "expected_relevant_ids",
    "expected_answer_keypoints",
]

# -------------------------- Templates -------------------------------------- 
# Mỗi intent có bản vi + en. {d} = mô tả sản phẩm đã dựng theo ngôn ngữ.
STRICT_TEMPLATES = {
    "price": {"vi": "{d} giá bao nhiêu?", "en": "What's the price of the {d}?"},
    "stock": {"vi": "{d} còn hàng không?", "en": "Is the {d} in stock?"},
    "size": {"vi": "{d} có size nào?", "en": "What size is the {d} available in?"},
    "full": {"vi": "Cho tôi thông tin về {d}.", "en": "Tell me about the {d}."},
}

LOOSE_TEMPLATES = {
    "price": {"vi": "{d} giá bao nhiêu?", "en": "How much does {d} cost?"},
    "stock": {"vi": "Shop còn {d} không?", "en": "Do you have {d} in stock?"},
    "options": {"vi": "Shop có những mẫu {d} nào?", "en": "What options do you have for {d}?"},
    "size": {"vi": "{d} có sẵn size nào?", "en": "What sizes are available for {d}?"},
}

NOT_FOUND_TEMPLATES = {
    "price": {"vi": "{d} giá bao nhiêu?", "en": "What's the price of {d}?"},
    "stock": {"vi": "Shop có {d} không?", "en": "Do you sell {d}?"},
}


def _sentence(text):
    """Descriptor bị lowercase để ghép vào giữa câu -> viết hoa lại nếu nó rơi vào đầu câu."""
    return text[:1].upper() + text[1:] if text else text


def _fmt_money(value, language):
    """Format số tiền theo ngôn ngữ: Dùng dấu . cho VI và , cho EN."""
    sep = "." if language == "vi" else ","
    return f"{int(value):,}".replace(",", sep) + " VND"


def describe_strict(row, language):
    """
    Mô tả có brand -> định danh duy nhất 1 sản phẩm.
    Dùng cho câu hỏi strict: trả thẳng tên đầy đủ của sản phẩm trong catalog, 
    không phụ thuộc ngôn ngữ vì tên sản phẩm không dịch.
    """
    return str(row["productDisplayName"]).strip()


def describe_loose(row, language):
    """
    Mô tả chỉ bằng thuộc tính -> cố tình khớp nhiều sản phẩm.
    Ghép 3 thuộc tính chung chung (loại - giới tính - màu), không nêu tên/brand cụ thể, 
    để mô phỏng cách khách hàng thật thường hỏi (ambiguous, có thể khớp nhiều sản phẩm khác nhau trong catalog).
    """
    if language == "vi":
        return f"{row['type_vi'].lower()} {str(row['gender_vi']).lower()} màu {str(row['color_vi']).lower()}"
    return f"{row['articleType'].lower()} for {row['gender'].lower()} in {row['baseColour'].lower()}"


def describe_combo(gender, article, colour, gender_vi, type_vi, color_vi, language):
    """
    Giống describe_loose nhưng nhận giá trị rời thay vì một row thật.
    Dùng cho gen_not_found(), nơi tổ hợp (gender, articleType, baseColour) đang xét KHÔNG tồn tại trong catalog 
    -> không có row thật nào để lấy nên phải truyền tay từng giá trị (đã tra cứu bản dịch từ nơi khác).
    """
    if language == "vi":
        return f"{type_vi.lower()} {gender_vi.lower()} màu {color_vi.lower()}"
    return f"{article.lower()} for {gender.lower()} in {colour.lower()}"


# ---------------------------- Filters ------------------------------------ 
def parse_filter(spec):
    """Parse chuỗi filter dạng "key1=value1;key2=value2" thành dict."""
    out = {}
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def apply_filter(df, spec):
    """
    Lọc DataFrame sản phẩm theo chuỗi filter spec.
    Hỗ trợ lọc theo id (khớp tuyệt đối), gender/articleType/baseColour/size (khớp tuyệt đối dạng string)
    và khoảng giá price_min/price_max.
    """
    f = parse_filter(spec)
    m = df
    if "id" in f:
        m = m[m["id"].astype(str) == f["id"]]
    for col, key in (("gender", "gender"), ("articleType", "articleType"),
                     ("baseColour", "baseColour"), ("size", "size")):
        if key in f:
            m = m[m[col].astype(str) == f[key]]
    if "price_min" in f:
        m = m[m["price"] >= int(f["price_min"])]
    if "price_max" in f:
        m = m[m["price"] <= int(f["price_max"])]
    return m


# ---------------------------- Keypoints ------------------------------------ 
def keypoints_strict(match, language):
    """Sinh ground truth keypoints cho câu hỏi strict (khớp đúng 1 sản phẩm)."""
    row = match.iloc[0]
    price = _fmt_money(row["price"], language)
    stock = int(row["stock"])
    if language == "vi":
        stock_text = "hết hàng" if stock == 0 else f"còn {stock} sản phẩm"
        return (f"{row['productDisplayName']} — size {row['size']}, giá {price}, {stock_text}")
    stock_text = "out of stock" if stock == 0 else f"{stock} in stock"
    return (f"{row['productDisplayName']} — size {row['size']}, price {price}, {stock_text}")


def keypoints_loose(match, language):
    """
    Sinh ground-truth keypoints cho câu hỏi loose (khớp nhiều sản phẩm).
    Ground truth cho câu ambiguous = mô tả CẢ TẬP sản phẩm khớp, không phải 1 sản phẩm cụ thể.
    """
    n = len(match)
    lo = _fmt_money(match["price"].min(), language)
    hi = _fmt_money(match["price"].max(), language)
    in_stock = int((match["stock"] > 0).sum())
    sizes = sorted({str(s) for s in match["size"]})
    examples = "; ".join(str(x) for x in match["productDisplayName"].head(3))

    if language == "vi":
        return (f"Có {n} sản phẩm khớp, giá từ {lo} đến {hi}, {in_stock} mẫu còn hàng, "
                f"size có: {', '.join(sizes)}. Ví dụ hợp lệ: {examples}. "
                f"Trả lời đúng nếu nêu được ít nhất một sản phẩm trong tập này kèm giá/tồn kho chính xác.")
    return (f"{n} matching products, price from {lo} to {hi}, {in_stock} in stock, "
            f"sizes: {', '.join(sizes)}. Valid examples: {examples}. "
            f"Correct if it names at least one product from this set with accurate price/stock.")


def keypoints_not_found(descriptor, language):
    """Sinh ground-truth keypoints cho câu hỏi product_not_found."""
    if language == "vi":
        return (f"Catalog KHÔNG có sản phẩm nào khớp '{descriptor}'. Trả lời đúng = nói rõ "
                f"không có/không tìm thấy và đề nghị liên hệ nhân viên. Bịa ra một sản phẩm là SAI.")
    return (f"The catalog has NO product matching '{descriptor}'. Correct answer = clearly state "
            f"it is unavailable and suggest contacting support. Inventing a product is WRONG.")


# ------------------------------- Generation --------------------------------- 
def _stratified_rows(df, n, unique_only, rng):
    """
    Chọn n sản phẩm đại diện, rải đều theo articleType và có cả hết hàng.
 
    Gồm 3 bước:
    - Nhóm theo (gender, articleType, baseColour), lọc tổ hợp phù hợp (unique_only=True: đúng 1 sản phẩm/tổ hợp, dùng cho strict; 
    unique_only=False: >=5 sản phẩm/tổ hợp, dùng cho loose).
    - Round-robin interleave các tổ hợp theo articleType để tránh dồn hết vào 1-2 loại có nhiều tổ hợp nhất khi duyệt tuần tự.
    - Duyệt 2 lượt: lượt 1 chỉ nhận tổ hợp có sản phẩm hết hàng (stock=0) cho tới khi đủ oos_quota, lượt 2 lấp phần còn lại không
    phân biệt tồn kho. Cần 2 lượt vì nếu chỉ duyệt 1 lượt ngẫu nhiên, xác suất trúng sản phẩm hết hàng rất thấp (đặc biệt khi 
    unique_only=True), khiến eval set mất hẳn case hết hàng.
    """
    combo = df.groupby(["gender", "articleType", "baseColour"]).size()
    wanted = combo[combo == 1] if unique_only else combo[combo >= 5]
    keys = list(wanted.index)
    rng.shuffle(keys)

    # Rải theo articleType: mỗi type góp tối đa 1 lần trước khi lặp lại
    by_type = {}
    for g, a, c in keys:
        by_type.setdefault(a, []).append((g, a, c))
    order = []
    pools = [by_type[a] for a in sorted(by_type, key=lambda a: -len(by_type[a]))]
    while any(pools):
        for pool in pools:
            if pool:
                order.append(pool.pop())

    # Hai lượt: lượt 1 chỉ nhặt tổ hợp CÓ sản phẩm hết hàng cho tới đủ quota, lượt 2 lấp phần còn lại. 
    oos_quota = max(2, n // 5)
    picked, seen_types, used = [], Counter(), set()

    def _take(key, prefer_oos):
        """
        Thử thêm 1 tổ hợp vào `picked`, tôn trọng các ràng buộc.
        Bỏ qua nếu đã đủ n, articleType này đã xuất hiện >=2 lần, tổ hợp đã dùng, tổ hợp không có sản phẩm nào
        hoặc (khi prefer_oos=True) tổ hợp không có sản phẩm hết hàng nào.
        """
        g, a, c = key
        if len(picked) >= n or seen_types[a] >= 2 or key in used:
            return
        m = df[(df.gender == g) & (df.articleType == a) & (df.baseColour == c)]
        if m.empty:
            return
        oos = m[m.stock == 0]
        if prefer_oos:
            if oos.empty:
                return
            row = oos.iloc[0]
        else:
            row = m.iloc[0]
        picked.append((row, m))
        seen_types[a] += 1
        used.add(key)

    for key in order:
        if len(picked) >= oos_quota:
            break
        _take(key, prefer_oos=True)
    for key in order:
        _take(key, prefer_oos=False)
    return picked


def gen_strict(df, rng, start=1):
    """
    Sinh danh sách câu hỏi strict (khớp đúng 1 sản phẩm).
    Với mỗi sản phẩm đại diện từ _stratified_rows(unique_only=True), sinh 1 câu hỏi bằng cách luân phiên ngôn ngữ (vi/en) và 
    intent (price/stock/size/full) theo cycle(), filter theo id sản phẩm.
    """
    rows = []
    langs = cycle(["vi", "en"])
    intents = cycle(list(STRICT_TEMPLATES))
    for i, (row, _) in enumerate(_stratified_rows(df, N_STRICT, True, rng), start=start):
        lang, intent = next(langs), next(intents)
        rows.append({
            "id": f"ps{i:03d}",
            "question": _sentence(STRICT_TEMPLATES[intent][lang].format(d=describe_strict(row, lang))),
            "language": lang,
            "category": "product",
            "strictness": "strict",
            "expected_doc_type": "product",
            "product_filter": f"id={row['id']}",
        })
    return rows


def gen_loose(df, rng, start=1):
    """
    Sinh danh sách câu hỏi loose (khớp nhiều sản phẩm cùng thuộc tính).
    Với mỗi sản phẩm đại diện từ _stratified_rows(unique_only=False), sinh 1 câu hỏi mô tả chung chung (không nêu tên/brand),
    filter theo 3 thuộc tính (gender, articleType, baseColour) thay vì id.
    """
    rows = []
    langs = cycle(["en", "vi"])
    intents = cycle(list(LOOSE_TEMPLATES))
    for i, (row, _) in enumerate(_stratified_rows(df, N_LOOSE, False, rng), start=start):
        lang, intent = next(langs), next(intents)
        rows.append({
            "id": f"pl{i:03d}",
            "question": _sentence(LOOSE_TEMPLATES[intent][lang].format(d=describe_loose(row, lang))),
            "language": lang,
            "category": "product",
            "strictness": "loose",
            "expected_doc_type": "product",
            "product_filter": (f"gender={row['gender']};articleType={row['articleType']};"
                               f"baseColour={row['baseColour']}"),
        })
    return rows


def gen_not_found(df, rng, start=1):
    """
    Sinh danh sách câu hỏi product_not_found (test hallucination).
    Tổ hợp (gender, articleType, baseColour) có 0 sản phẩm, nhưng cả articleType lẫn màu đều tồn tại trong catalog 
    -> câu hỏi nghe rất hợp lý, chỉ có điều không có hàng. Đây là bài test hallucination sắc nhất.
 
    Chỉ dùng articleType/baseColour phổ biến (>=20 sản phẩm mỗi loại) để đảm bảo từng thành phần riêng lẻ quen thuộc,
    chỉ tổ hợp cụ thể là không tồn tại."""
    present = set(map(tuple, df[["gender", "articleType", "baseColour"]].drop_duplicates().values))
    vi_gender = dict(zip(df.gender, df.gender_vi))
    vi_type = dict(zip(df.articleType, df.type_vi))
    vi_color = dict(zip(df.baseColour, df.color_vi))

    types = [t for t, n in df.articleType.value_counts().items() if n >= 20]
    colours = [c for c, n in df.baseColour.value_counts().items() if n >= 20]
    genders = ["Men", "Women"]

    candidates = [(g, a, c) for a in types for c in colours for g in genders
                  if (g, a, c) not in present]
    rng.shuffle(candidates)

    rows, seen_types = [], Counter()
    langs = cycle(["vi", "en"])
    intents = cycle(list(NOT_FOUND_TEMPLATES))
    i = start
    for g, a, c in candidates:
        if len(rows) >= N_NOT_FOUND:
            break
        if seen_types[a] >= 2:
            continue
        lang, intent = next(langs), next(intents)
        d = describe_combo(g, a, c, vi_gender.get(g, g), vi_type.get(a, a), vi_color.get(c, c), lang)
        rows.append({
            "id": f"pn{i:03d}",
            "question": _sentence(NOT_FOUND_TEMPLATES[intent][lang].format(d=d)),
            "descriptor": d,
            "language": lang,
            "category": "product_not_found",
            "strictness": "not_found",
            "expected_doc_type": "product",
            "product_filter": f"gender={g};articleType={a};baseColour={c}",
        })
        seen_types[a] += 1
        i += 1
    return rows


# ------------------------------- Resolve --------------------------------- 
def resolve(rows, df):
    """
    Điền expected_relevant_ids + expected_answer_keypoints cho mỗi row. Điền keypoints từ data. 
    Với mỗi row, chạy lại apply_filter để lấy danh sách id khớp, rồi gọi đúng hàm keypoints_*
    tương ứng với category/strictness của row để sinh ground truth.
    """
    for r in rows:
        match = apply_filter(df, r["product_filter"])
        ids = [str(x) for x in match["id"]]
        r["expected_relevant_ids"] = ";".join(ids)
        lang = r["language"]

        if r["category"] == "product_not_found":
            r["expected_answer_keypoints"] = keypoints_not_found(r["descriptor"], lang)
        elif r["strictness"] == "strict":
            r["expected_answer_keypoints"] = keypoints_strict(match, lang)
        else:
            r["expected_answer_keypoints"] = keypoints_loose(match, lang)
    return rows


def validate(rows, df):
    """
    Kiểm tra tính đúng đắn của các row đã resolve, đối chiếu với catalog.
    Chạy lại apply_filter để đếm số sản phẩm khớp thực tế, so với kỳ vọng của từng loại câu hỏi 
    (strict phải =1, loose phải >=2, not_found phải =0) và kiểm tra không có câu hỏi trùng lặp.
    """
    errors = []
    seen_q = set()
    for r in rows:
        n = len(apply_filter(df, r["product_filter"])) if r["product_filter"] else None
        if r["category"] == "product" and r["strictness"] == "strict" and n != 1:
            errors.append(f"{r['id']}: strict nhưng khớp {n} sản phẩm (phải là 1)")
        if r["category"] == "product" and r["strictness"] == "loose" and (n or 0) < 2:
            errors.append(f"{r['id']}: loose nhưng chỉ khớp {n} (phải >= 2)")
        if r["category"] == "product_not_found" and n != 0:
            errors.append(f"{r['id']}: not_found nhưng khớp {n} sản phẩm (phải là 0)")
        if r["question"] in seen_q:
            errors.append(f"{r['id']}: câu hỏi trùng — {r['question']!r}")
        seen_q.add(r["question"])
    return errors


def report(rows, df):
    """
    In thống kê độ đa dạng của eval set ra console (chỉ để debug).
    Thống kê phân bố theo category/language/strictness, cùng số lượng articleType/baseColour khác nhau xuất hiện, 
    phân bố gender và số câu hỏi liên quan sản phẩm hết hàng.
    """
    print("\n" + "=" * 62)
    print("EVAL SET DIVERSITY")
    print("=" * 62)
    print("category    :", dict(Counter(r["category"] for r in rows)))
    print("language    :", dict(Counter(r["language"] for r in rows)))
    print("strictness  :", dict(Counter(r.get("strictness") or "-" for r in rows)))

    prod = [r for r in rows if r["product_filter"]]
    types, genders, colours = set(), Counter(), set()
    oos = 0
    for r in prod:
        f = parse_filter(r["product_filter"])
        m = apply_filter(df, r["product_filter"])
        if not m.empty:
            types |= set(m.articleType)
            genders[m.iloc[0].gender] += 1
            colours |= set(m.baseColour)
            if int(m.iloc[0].stock) == 0:
                oos += 1
        elif "articleType" in f:
            types.add(f["articleType"])
            genders[f.get("gender", "?")] += 1
            colours.add(f.get("baseColour", "?"))
    print(f"articleType : {len(types)} loại khác nhau")
    print(f"màu         : {len(colours)} màu khác nhau")
    print("gender      :", dict(genders))
    print(f"hết hàng    : {oos} câu hỏi về sản phẩm stock=0")
    print("=" * 62)


def main():
    """
    Luồng xử lý:
    - Đọc catalog đã xử lý (PROCESSED_PRODUCTS_PATH).
    - Sinh câu hỏi strict + loose + not_found, resolve ground truth.
    - Validate, nếu có lỗi thì in ra và thoát chương trình (không ghi file sai ra ngoài).
    - Đọc thêm bộ câu hỏi manual đã verify tay (eval_set_manual.csv).
    - Gộp generated + manual, ghi ra EVAL_SET_PATH theo đúng FIELDNAMES.
    - In log tổng kết và báo cáo độ đa dạng.
    """
    df = pd.read_csv(PROCESSED_PRODUCTS_PATH)
    rng = np.random.default_rng(SEED)

    generated = gen_strict(df, rng) + gen_loose(df, rng) + gen_not_found(df, rng)
    generated = resolve(generated, df)

    errors = validate(generated, df)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    with open(MANUAL_SET_PATH, "r", encoding="utf-8") as f:
        manual = [{**{k: "" for k in FIELDNAMES}, **row} for row in csv.DictReader(f)]

    all_rows = generated + manual
    with open(EVAL_SET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in FIELDNAMES} for r in all_rows)

    print(f"Generated {len(generated)} product rows + {len(manual)} manual rows "
          f"-> {EVAL_SET_PATH} ({len(all_rows)} total)")
    report(all_rows, df)


if __name__ == "__main__":
    main()