import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd

from config.backend import (
    PROCESSED_PRODUCTS_PATH,
    CLEAN_DATA_PATH,
    SEED,
    set_seed,
)

DATA_PATH = Path(CLEAN_DATA_PATH)
OUT_PATH = Path(PROCESSED_PRODUCTS_PATH)
DEFAULT_N_SAMPLES = int(os.getenv("PRODUCTS_N_SAMPLES", "5000"))


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
    "Tshirts": (200, 500),
    "Shirts": (300, 1000),
    "Tops": (250, 600),
    "Jeans": (300, 650),
    "Shorts": (250, 450),
    "Trousers": (250, 850),
    "Dresses": (300, 2000),
    "Casual Shoes": (500, 1000),
    "Sports Shoes": (600, 2200),
    "Formal Shoes": (500, 2200),
    "Flats": (300, 650),
    "Heels": (350, 800),
    "Flip Flops": (80, 200),
    "Sandals": (150, 550),
    "Handbags": (350, 3000),
    "Backpacks": (300, 800),
    "Wallets": (250, 950),
    "Belts": (250, 500),
    "Sunglasses": (200, 700),
    "Watches": (500, 5500),
}

DEFAULT_PRICE_RANGE = (250, 550)

SIZE_OPTIONS = {
    "Topwear": ["S", "M", "L", "XL"],
    "Bottomwear": ["S", "M", "L", "XL"],
    "Dress": ["S", "M", "L", "XL"],
    "Shoes": ["37", "38", "39", "40", "41", "42"],
    "Sandal": ["37", "38", "39", "40", "41", "42"],
    "Flip Flops": ["37", "38", "39", "40", "41", "42"],
}


def _row_seed(product_id):
    """
    Sinh seed số nguyên ổn định (deterministic) từ product id.
    Dùng hash MD5 thay vì hash() built-in của Python, vì hash() bị salt ngẫu nhiên mỗi lần khởi động process (PYTHONHASHSEED) 
    -> cùng 1 id sẽ ra seed khác nhau giữa các lần chạy, phá vỡ tính reproducible.
    """
    digest = hashlib.md5(str(product_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2 ** 32)


def _vi(mapping, key, fallback=None):
    """Tra cứu bản dịch VI từ dict mapping, có xử lý NaN và fallback"""
    if pd.isna(key):
        return fallback if fallback is not None else "Không xác định"
    return mapping.get(key, fallback if fallback is not None else key)


def load_data(path=DATA_PATH):
    """Đọc file CSV sản phẩm đã tiền xử lý."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def filter_fashion_scope(df, n_samples=None):
    """
    Lọc DataFrame về đúng phạm vi thời trang rồi lấy mẫu cân bằng theo loại.

    Chỉ giữ 2 nhóm: Apparel/Footwear (toàn bộ), và Accessories nhưng chỉ các subCategory liên quan thời trang 
    (Bags, Watches, Belts, Wallets, Eyewear) — loại bỏ các phụ kiện không phù hợp use case retail, vẫn đa dạng
    và không bị mất cân bằng.

    Nếu n_samples được chỉ định, lấy mẫu theo kiểu stratified: chia đều theo articleType (mỗi loại lấy tối đa 
    n_samples/n_types sản phẩm) để tránh 1-2 loại phổ biến (Vd Tshirts) chiếm áp đảo toàn bộ catalog, sau đó 
    lấp đầy phần thiếu (nếu do 1 số loại ít sản phẩm hơn quota) bằng cách lấy thêm ngẫu nhiên từ phần dư.
    """
    apparel = df[df.masterCategory.isin(["Apparel", "Footwear"])]
    accessories = df[(df.masterCategory == "Accessories") & (df.subCategory.isin(["Bags", "Watches", "Belts", "Wallets", "Eyewear"]))]
    scoped = pd.concat([apparel, accessories])

    if n_samples is None: # Nếu không giới hạn số lượng mẫu (lấy toàn bộ)
        return scoped.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_samples = min(n_samples, len(scoped))

    n_types = max(1, scoped.articleType.nunique())
    per_type = max(1, n_samples // n_types)

    sampled = []
    for _, group in scoped.groupby("articleType"):
        sampled.append(group.sample(min(len(group), per_type), random_state=SEED))
    sampled = pd.concat(sampled)

    if len(sampled) < n_samples: # Nếu chưa đủ mẫu
        remaining = scoped.loc[~scoped.index.isin(sampled.index)]
        extra = remaining.sample(
            min(n_samples - len(sampled), len(remaining)),
            random_state=SEED,
        )
        sampled = pd.concat([sampled, extra])

    return sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)


def enrich_bilingual_fields(df):
    """
    Làm giàu DataFrame với các trường song ngữ VI/EN.
    Thêm các cột dịch thuật (color_vi, type_vi, gender_vi, usage_vi) tra từ các dictionary tĩnh ở đầu file,
    rồi ghép chúng lại thành tên (name_vi/name_en) và mô tả đầy đủ (description_vi/description_en) cho từng sản phẩm.
    """
    df = df.copy()

    df["color_vi"] = df.baseColour.apply(lambda x: _vi(COLOR_VI, x))
    df["type_vi"] = df.articleType.apply(lambda x: _vi(ARTICLE_TYPE_VI, x))
    df["gender_vi"] = df.gender.apply(lambda x: _vi(GENDER_VI, x))
    df["usage_vi"] = df.usage.apply(lambda x: _vi(USAGE_VI, x, "thường ngày"))

    df["name_vi"] = (df["type_vi"] + " " + df["gender_vi"].str.lower() + " màu " + df["color_vi"].str.lower())
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
    """
    Sinh giá/tồn kho/size giả một cách deterministic cho từng sản phẩm.

    Logic sinh:
    - price: random uniform trong khoảng PRICE_RANGE[articleType] (nghìn VND), nhân 1000. Dùng DEFAULT_PRICE_RANGE nếu loại
    không có trong bảng.
    - stock: random uniform [1, 50], nhưng có 10% xác suất ép về 0 (hết hàng) — đảm bảo luôn có case out-of-stock trong catalog.
    - size: random chọn 1 trong SIZE_OPTIONS[subCategory]. Nếu subCategory không có trong bảng, gán "Free size".
    """
    df = df.copy()
    prices, stocks, sizes = [], [], []

    for _, row in df.iterrows():
        rng = np.random.default_rng(_row_seed(row["id"]))

        low, high = PRICE_RANGE.get(row["articleType"], DEFAULT_PRICE_RANGE)
        prices.append(int(rng.integers(low, high + 1)) * 1000)

        stock = int(rng.integers(1, 51))
        if rng.random() < 0.10:
            stock = 0
        stocks.append(stock)

        options = SIZE_OPTIONS.get(row["subCategory"])
        sizes.append("Free size" if options is None else str(rng.choice(options)))

    df["price"], df["stock"], df["size"] = prices, stocks, sizes
    return df


def main():
    """
    Luồng xử lý:
    - set_seed(SEED) - cố định seed toàn cục (ảnh hưởng các bước sampling dùng random_state=SEED trong filter_fashion_scope).
    - Đọc data (load_data).
    - Lọc phạm vi thời trang + lấy mẫu cân bằng (filter_fashion_scope).
    - Làm giàu song ngữ VI/EN (enrich_bilingual_fields).
    - Sinh giá/tồn kho/size deterministic theo id (generate_price_stock_size).
    - Ghi ra PROCESSED_PRODUCTS_PATH, tạo thư mục cha nếu chưa có.
    """
    set_seed(SEED)
    df = load_data()
    df = filter_fashion_scope(df, DEFAULT_N_SAMPLES)
    df = enrich_bilingual_fields(df)
    df = generate_price_stock_size(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} products")


if __name__ == "__main__":
    main()