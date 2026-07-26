import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    RAW_PRODUCTS_PATH,
    PROCESSED_PRODUCTS_PATH,
    SEED,
    set_seed,
)

RAW_PATH = Path(RAW_PRODUCTS_PATH)
OUT_PATH = Path(PROCESSED_PRODUCTS_PATH)

DEFAULT_N_SAMPLES = int(os.getenv("PRODUCTS_N_SAMPLES", "100"))


# ---- Dictionaries ----
COLOR_VI = {
    "Black": "Đen",
    "White": "Trắng",
    "Blue": "Xanh dương",
    "Brown": "Nâu",
    "Grey": "Xám",
    "Red": "Đỏ",
    "Green": "Xanh lá",
    "Pink": "Hồng",
    "Navy Blue": "Xanh navy",
    "Purple": "Tím",
    "Silver": "Bạc",
    "Yellow": "Vàng",
    "Beige": "Be",
    "Gold": "Vàng gold",
    "Maroon": "Đỏ mận",
    "Orange": "Cam",
    "Olive": "Xanh olive",
    "Multi": "Phối nhiều màu",
    "Cream": "Kem",
    "Steel": "Xám thép",
}

ARTICLE_TYPE_VI = {
    "Tshirts": "Áo thun",
    "Shirts": "Áo sơ mi",
    "Casual Shoes": "Giày sneaker",
    "Sports Shoes": "Giày thể thao",
    "Formal Shoes": "Giày tây",
    "Tops": "Áo kiểu",
    "Handbags": "Túi xách",
    "Heels": "Giày cao gót",
    "Sunglasses": "Kính râm",
    "Wallets": "Ví",
    "Flip Flops": "Dép xỏ ngón",
    "Sandals": "Sandal",
    "Belts": "Thắt lưng",
    "Backpacks": "Balo",
    "Jeans": "Quần jeans",
    "Shorts": "Quần short",
    "Trousers": "Quần tây",
    "Flats": "Giày bệt",
    "Dresses": "Váy đầm",
    "Watches": "Đồng hồ",
}

GENDER_VI = {
    "Men": "Nam",
    "Women": "Nữ",
    "Unisex": "Unisex",
    "Boys": "Bé trai",
    "Girls": "Bé gái",
}

USAGE_VI = {
    "Casual": "thường ngày",
    "Formal": "công sở",
    "Sports": "thể thao",
    "Ethnic": "truyền thống",
    "Party": "dự tiệc",
    "Travel": "du lịch",
    "Smart Casual": "lịch sự thường ngày",
}

PRICE_RANGE = {
    "Tshirts": (100, 300),
    "Shirts": (200, 500),
    "Tops": (150, 400),
    "Jeans": (300, 650),
    "Shorts": (150, 350),
    "Trousers": (250, 550),
    "Dresses": (300, 700),
    "Casual Shoes": (500, 900),
    "Sports Shoes": (600, 1200),
    "Formal Shoes": (500, 1200),
    "Flats": (300, 650),
    "Heels": (350, 800),
    "Flip Flops": (80, 200),
    "Sandals": (150, 350),
    "Handbags": (350, 1000),
    "Backpacks": (300, 800),
    "Wallets": (150, 450),
    "Belts": (150, 400),
    "Sunglasses": (200, 600),
    "Watches": (400, 2500),
}

DEFAULT_PRICE_RANGE = (150, 450)

SIZE_OPTIONS = {
    "Topwear": ["S", "M", "L", "XL"],
    "Bottomwear": ["S", "M", "L", "XL"],
    "Dress": ["S", "M", "L", "XL"],
    "Shoes": ["37", "38", "39", "40", "41", "42"],
    "Sandal": ["37", "38", "39", "40", "41", "42"],
    "Flip Flops": ["37", "38", "39", "40", "41", "42"],
}



def _vi(mapping, key, fallback=None):
    if pd.isna(key):
        return fallback if fallback is not None else "Không xác định"
    return mapping.get(key, fallback if fallback is not None else key)


def load_raw(path=RAW_PATH):

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, on_bad_lines="skip")

    df["usage"] = df["usage"].fillna("Casual")
    df["gender"] = df["gender"].fillna("Unisex")
    df["baseColour"] = df["baseColour"].fillna("Black")

    return df


def filter_fashion_scope(df, n_samples=None):

    apparel = df[df.masterCategory.isin(["Apparel", "Footwear"])]

    accessories = df[
        (df.masterCategory == "Accessories") & (df.subCategory.isin(["Bags", "Watches", "Belts", "Wallets", "Eyewear"]))
    ]

    scoped = pd.concat([apparel, accessories])

    scoped = scoped.dropna(
        subset=[
            "productDisplayName",
            "articleType",
            "baseColour",
        ]
    )

    if n_samples is None:
        return scoped.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_samples = min(n_samples, len(scoped))
    n_types = scoped.articleType.nunique()
    per_type = max(1, n_samples // n_types)

    sampled = []

    for _, group in scoped.groupby("articleType"):
        sampled.append(group.sample(min(len(group), per_type), random_state=SEED))

    sampled = pd.concat(sampled)

    if len(sampled) < n_samples:
        remaining = scoped.loc[~scoped.index.isin(sampled.index)]

        extra = remaining.sample(
            min(n_samples - len(sampled), len(remaining)),
            random_state=SEED,
        )

        sampled = pd.concat([sampled, extra])

    return sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)


def enrich_bilingual_fields(df):
    df = df.copy()

    df["color_vi"] = df.baseColour.apply(lambda x: _vi(COLOR_VI, x))
    df["type_vi"] = df.articleType.apply(lambda x: _vi(ARTICLE_TYPE_VI, x))
    df["gender_vi"] = df.gender.apply(lambda x: _vi(GENDER_VI, x))
    df["usage_vi"] = df.usage.apply(
        lambda x: _vi(USAGE_VI, x, "thường ngày")
    )

    df["name_vi"] = (
        df["type_vi"] + " " + df["gender_vi"].str.lower() + " màu " + df["color_vi"].str.lower()
    )

    df["name_en"] = df.productDisplayName

    df["description_vi"] = (
        df["type_vi"] + " dành cho " + df["gender_vi"].str.lower() + ", màu "
        + df["color_vi"].str.lower() + ", phù hợp cho dịp " + df["usage_vi"] + "."
    )

    df["description_en"] = (
        df.articleType + " for " + df.gender + ", color " + df.baseColour
        + ", suitable for " + df.usage.str.lower() + "."
    )

    return df


def generate_price_stock_size(df):
    rng = np.random.default_rng(SEED)
    df = df.copy()

    def make_price(article):
        low, high = PRICE_RANGE.get(
            article,
            DEFAULT_PRICE_RANGE,
        )

        return int(rng.integers(low, high + 1)) * 1000

    def make_size(sub):
        options = SIZE_OPTIONS.get(sub)
        if options is None:
            return "Free size"
        return rng.choice(options)

    df["price"] = df.articleType.apply(make_price)
    df["stock"] = rng.integers(1, 51, len(df))
    sold_out = rng.random(len(df)) < 0.10
    df.loc[sold_out, "stock"] = 0
    df["size"] = df.subCategory.apply(make_size)

    return df


def main():
    set_seed(SEED)
    df = load_raw()
    df = filter_fashion_scope(df, DEFAULT_N_SAMPLES)
    df = enrich_bilingual_fields(df)
    df = generate_price_stock_size(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} products")


if __name__ == "__main__":
    main()