import os
import glob
import hashlib

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from src.config import (
    PROCESSED_PRODUCTS_PATH,
    POLICIES_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    POLICY_CHUNK_SIZE,
    POLICY_CHUNK_OVERLAP,
    set_seed,
)

load_dotenv()


def _stable_id(text):
    """Hash nội dung -> id ổn định, dùng để upsert idempotent (không duplicate khi chạy lại)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_products_as_chunks(path=PROCESSED_PRODUCTS_PATH):
    df = pd.read_csv(path)
    chunks = []

    for _, row in df.iterrows():
        # Mỗi sản phẩm = 1 chunk, gộp đủ field quan trọng để retrieval bắt được
        # cả câu hỏi tiếng Việt lẫn tiếng Anh (name/description song ngữ trong cùng 1 chunk)
        content = (
            f"{row['name_vi']} / {row['name_en']}\n"
            f"Loại: {row['type_vi']} | Type: {row['articleType']}\n"
            f"Màu: {row['color_vi']} | Color: {row['baseColour']}\n"
            f"Giới tính: {row['gender_vi']} | Gender: {row['gender']}\n"
            f"Size: {row['size']}\n"
            f"Giá: {row['price']:,} VND | Tồn kho: {row['stock']}\n"
            f"{row['description_vi']}\n{row['description_en']}"
        )

        chunks.append({
            "id": f"product_{row['id']}",
            "content": content,
            "metadata": {
                "doc_type": "product",
                "doc_id": str(row["id"]),
                "price": int(row["price"]),
                "stock": int(row["stock"]),
            },
        })

    return chunks


def _split_text(text, chunk_size, overlap):
    """Chunk theo số từ (proxy đơn giản cho token), có overlap để không cắt đứt ngữ cảnh."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_policies_as_chunks(dir_path: str = POLICIES_DIR):
    chunks = []

    for filepath in sorted(glob.glob(os.path.join(dir_path, "*.md"))):
        source_file = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        pieces = _split_text(text, POLICY_CHUNK_SIZE, POLICY_CHUNK_OVERLAP)

        for i, piece in enumerate(pieces):
            chunk_id = f"policy_{source_file}_{i}"
            chunks.append({
                "id": chunk_id,
                "content": piece,
                "metadata": {
                    "doc_type": "policy",
                    "doc_id": chunk_id,
                    "source_file": source_file,
                },
            })

    return chunks


def build_and_persist_index(chunks, batch_size=128, use_multiprocess=None):
    """
    batch_size: tăng lên (64-128+) giúp CPU/GPU encode hiệu quả hơn so với
        mặc định 32 của sentence-transformers — ít overhead giữa các batch.
    use_multiprocess: chia việc encode ra nhiều tiến trình CPU song song.
        Mặc định None -> tự bật khi có >1000 chunk và máy có >1 core (lợi ích
        rõ nhất ở dataset lớn như 5000 sản phẩm; với dataset nhỏ, overhead
        khởi tạo pool còn tốn hơn cả lợi ích).
    """
    if not chunks:
        print("Không có chunk nào để index.")
        return

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    ids = [c["id"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    if use_multiprocess is None:
        use_multiprocess = len(documents) > 1000 and os.cpu_count() and os.cpu_count() > 1

    print(f"Encoding {len(documents)} chunks (batch_size={batch_size}, multiprocess={use_multiprocess})...")

    if use_multiprocess:
        # API mới: encode() nhận thẳng tham số pool thay vì gọi encode_multi_process
        # riêng (encode_multi_process đã bị deprecate) — normalize_embeddings vẫn
        # hoạt động bình thường khi truyền pool.
        pool = model.start_multi_process_pool()
        embeddings = model.encode(
            documents, pool=pool, batch_size=batch_size, normalize_embeddings=True
        ).tolist()
        model.stop_multi_process_pool(pool)
    else:
        embeddings = model.encode(
            documents, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        ).tolist()

    upsert_batch = 500
    for start in range(0, len(ids), upsert_batch):
        end = start + upsert_batch
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  upserted {min(end, len(ids))}/{len(ids)}")


def main():
    set_seed()
    chunks = []
    chunks += load_products_as_chunks()
    chunks += load_policies_as_chunks()
    build_and_persist_index(chunks)
    print(f"Indexed {len(chunks)} chunks into {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()