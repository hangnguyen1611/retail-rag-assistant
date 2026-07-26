"""
build_eval_set.py

Build data/eval/eval_set.csv = (nửa product SINH TỪ DATA) + (nửa manual đã verify).

Vì sao tồn tại: nửa product của eval set cũ được LLM sinh mà không có
products.csv trong context -> bịa product_id, bịa giá, và không biết rằng một
mô tả kiểu "giày tây đen nam" khớp tới 46 sản phẩm. Nửa policy/out_of_scope thì
ngược lại: policy docs nhỏ, nằm trong context, nên chính xác 100% -> giữ nguyên
trong data/eval/eval_set_manual.csv.

Nguyên tắc: câu hỏi sinh TỪ sản phẩm thật, còn expected_relevant_ids và
expected_answer_keypoints thì KHÔNG BAO GIỜ viết tay — luôn resolve bằng code
từ products.csv qua cột product_filter.

Ba loại câu product:
  strict          — có brand/tên trong câu hỏi -> đúng 1 sản phẩm khớp.
                    Dùng để đo recall@1 một cách sắc nét.
  loose           — chỉ mô tả thuộc tính (loại/giới tính/màu) -> nhiều sản phẩm
                    khớp, ground truth là CẢ TẬP. Giống cách khách thật hỏi.
  product_not_found — tổ hợp thuộc tính không tồn tại trong catalog. Test
                    hallucination trực tiếp: trợ lý phải nói không có.

Chạy:
    python -m src.ingest.clean_products
    python -m src.eval.build_eval_set
"""

import csv
import os
import sys
from collections import Counter
from itertools import cycle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import EVAL_SET_PATH, PROCESSED_PRODUCTS_PATH, SEED

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

# ---------------------------------------------------------------- templates
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


def _sentence(text: str) -> str:
    """Descriptor bị lowercase để ghép vào giữa câu -> viết hoa lại nếu nó
    rơi vào đầu câu."""
    return text[:1].upper() + text[1:] if text else text


def _fmt_money(value: int, language: str) -> str:
    sep = "." if language == "vi" else ","
    return f"{int(value):,}".replace(",", sep) + " VND"


def describe_strict(row, language: str) -> str:
    """Mô tả có brand -> định danh duy nhất 1 sản phẩm."""
    return str(row["productDisplayName"]).strip()


def describe_loose(row, language: str) -> str:
    """Mô tả chỉ bằng thuộc tính -> cố tình khớp nhiều sản phẩm."""
    if language == "vi":
        return f"{row['type_vi'].lower()} {str(row['gender_vi']).lower()} màu {str(row['color_vi']).lower()}"
    return f"{row['articleType'].lower()} for {row['gender'].lower()} in {row['baseColour'].lower()}"


def describe_combo(gender, article, colour, gender_vi, type_vi, color_vi, language: str) -> str:
    if language == "vi":
        return f"{type_vi.lower()} {gender_vi.lower()} màu {color_vi.lower()}"
    return f"{article.lower()} for {gender.lower()} in {colour.lower()}"


# ---------------------------------------------------------------- filters
def parse_filter(spec: str) -> dict:
    out = {}
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def apply_filter(df: pd.DataFrame, spec: str) -> pd.DataFrame:
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


# ---------------------------------------------------------------- keypoints
def keypoints_strict(match: pd.DataFrame, language: str) -> str:
    row = match.iloc[0]
    price = _fmt_money(row["price"], language)
    stock = int(row["stock"])
    if language == "vi":
        stock_text = "hết hàng" if stock == 0 else f"còn {stock} sản phẩm"
        return (f"{row['productDisplayName']} — size {row['size']}, giá {price}, {stock_text}")
    stock_text = "out of stock" if stock == 0 else f"{stock} in stock"
    return (f"{row['productDisplayName']} — size {row['size']}, price {price}, {stock_text}")


def keypoints_loose(match: pd.DataFrame, language: str) -> str:
    """Ground truth cho câu ambiguous = mô tả CẢ TẬP, không phải 1 sản phẩm."""
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


def keypoints_not_found(descriptor: str, language: str) -> str:
    if language == "vi":
        return (f"Catalog KHÔNG có sản phẩm nào khớp '{descriptor}'. Trả lời đúng = nói rõ "
                f"không có/không tìm thấy và đề nghị liên hệ nhân viên. Bịa ra một sản phẩm là SAI.")
    return (f"The catalog has NO product matching '{descriptor}'. Correct answer = clearly state "
            f"it is unavailable and suggest contacting support. Inventing a product is WRONG.")


# ---------------------------------------------------------------- generation
def _stratified_rows(df: pd.DataFrame, n: int, unique_only: bool, rng) -> list:
    """Chọn n sản phẩm, rải đều theo articleType rồi tới gender, và cố tình
    chèn cả sản phẩm hết hàng để eval không chỉ toàn case 'còn hàng'."""
    combo = df.groupby(["gender", "articleType", "baseColour"]).size()
    wanted = combo[combo == 1] if unique_only else combo[combo >= 5]
    keys = list(wanted.index)
    rng.shuffle(keys)

    # rải theo articleType: mỗi type góp tối đa 1 lần trước khi lặp lại
    by_type = {}
    for g, a, c in keys:
        by_type.setdefault(a, []).append((g, a, c))
    order = []
    pools = [by_type[a] for a in sorted(by_type, key=lambda a: -len(by_type[a]))]
    while any(pools):
        for pool in pools:
            if pool:
                order.append(pool.pop())

    # Hai lượt: lượt 1 chỉ nhặt tổ hợp CÓ sản phẩm hết hàng cho tới đủ quota,
    # lượt 2 lấp phần còn lại. Nếu chỉ đi một lượt thì với tổ hợp duy nhất
    # (strict) gần như không bao giờ gặp stock=0 -> eval mất hẳn case hết hàng.
    oos_quota = max(2, n // 5)
    picked, seen_types, used = [], Counter(), set()

    def _take(key, prefer_oos):
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
    """Tổ hợp (gender, articleType, baseColour) có 0 sản phẩm, nhưng cả
    articleType lẫn màu đều tồn tại trong catalog -> câu hỏi nghe rất hợp lý,
    chỉ có điều không có hàng. Đây là bài test hallucination sắc nhất."""
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


# ---------------------------------------------------------------- resolve
def resolve(rows, df):
    """Điền expected_relevant_ids + keypoints từ data. Không tay người ở đây."""
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