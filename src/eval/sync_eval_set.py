"""
sync_eval_set.py

Sinh lại cột `expected_answer_keypoints` của data/eval/eval_set.csv từ
data/processed/products.csv (chỉ cho các câu category=product).

Vì sao cần: keypoints chứa số cụ thể (giá, tồn kho, size) được viết tay dựa
trên một lần generate products.csv trước đó. Mỗi lần regenerate data là số cũ
hết khớp -> LLM-as-judge chấm correctness thấp vì lý do dữ liệu, không phải
vì model trả lời sai. Script này làm cho products.csv là source of truth.

Chạy sau mỗi lần `python -m src.ingest.clean_products`:
    python -m src.eval.sync_eval_set
"""

import csv
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import EVAL_SET_PATH, PROCESSED_PRODUCTS_PATH

FIELDNAMES = [
    "id",
    "question",
    "language",
    "category",
    "expected_doc_type",
    "expected_relevant_ids",
    "expected_answer_keypoints",
]


def _fmt_money(value: int, language: str) -> str:
    sep = "." if language == "vi" else ","
    return f"{value:,}".replace(",", sep) + " VND"


def describe_product(row, language: str) -> str:
    """Một dòng fact đầy đủ cho 1 sản phẩm, theo ngôn ngữ của câu hỏi."""
    price = _fmt_money(int(row["price"]), language)
    stock = int(row["stock"])

    if language == "vi":
        stock_text = "hết hàng" if stock == 0 else f"còn {stock} sản phẩm"
        return (
            f"{row['name_en']} ({row['type_vi']} {str(row['gender_vi']).lower()} "
            f"màu {str(row['color_vi']).lower()}), size {row['size']}, "
            f"giá {price}, {stock_text}"
        )

    stock_text = "out of stock" if stock == 0 else f"{stock} in stock"
    return (
        f"{row['name_en']} ({row['articleType']} for {row['gender']}, "
        f"color {row['baseColour']}), size {row['size']}, "
        f"price {price}, {stock_text}"
    )


def build_keypoints(row, products_by_id) -> tuple[str, list[str]]:
    language = (row.get("language") or "vi").strip() or "vi"
    ids = [x.strip() for x in (row.get("expected_relevant_ids") or "").split(";") if x.strip()]

    parts, missing = [], []
    for pid in ids:
        product = products_by_id.get(pid)
        if product is None:
            missing.append(pid)
            continue
        parts.append(describe_product(product, language))

    return " | ".join(parts), missing


def main():
    products = pd.read_csv(PROCESSED_PRODUCTS_PATH)
    products["id"] = products["id"].astype(str)
    products_by_id = {r["id"]: r for _, r in products.iterrows()}

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    updated, all_missing = 0, {}
    for row in rows:
        if row.get("category") != "product":
            continue
        keypoints, missing = build_keypoints(row, products_by_id)
        if missing:
            all_missing[row["id"]] = missing
        if keypoints and keypoints != row["expected_answer_keypoints"]:
            row["expected_answer_keypoints"] = keypoints
            updated += 1

    shutil.copyfile(EVAL_SET_PATH, EVAL_SET_PATH + ".bak")
    with open(EVAL_SET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated keypoints for {updated} product question(s). Backup: {EVAL_SET_PATH}.bak")
    if all_missing:
        print("[warn] id không có trong products.csv (chạy lại clean_products.py để pin):")
        for qid, ids in all_missing.items():
            print(f"  {qid}: {ids}")
    else:
        print("Tất cả expected_relevant_ids đều tồn tại trong products.csv.")


if __name__ == "__main__":
    main()