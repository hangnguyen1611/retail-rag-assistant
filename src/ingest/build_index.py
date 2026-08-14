import glob
import os
import chromadb
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from config.backend import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_SPACE,
    EMBEDDING_MODEL,
    EMBEDDING_PASSAGE_PREFIX,
    POLICIES_DIR,
    POLICY_CHUNK_OVERLAP,
    POLICY_CHUNK_SIZE,
    PROCESSED_PRODUCTS_PATH,
    set_seed,
)

load_dotenv()


def load_products_as_chunks(path=PROCESSED_PRODUCTS_PATH):
    """
    Đọc products.csv và biến mỗi sản phẩm thành 1 chunk để index vào ChromaDB.
    Mỗi sản phẩm là 1 chunk riêng (không cắt nhỏ thêm, vì mô tả sản phẩm đủ ngắn để nằm gọn trong 1 embedding). 
    Nội dung chunk gộp cả tiếng Việt lẫn tiếng Anh (song ngữ) để retrieval hoạt động tốt bất kể câu hỏi bằng 
    ngôn ngữ nào. Metadata đi kèm để phục vụ filter/rerank sau khi retrieve (vd lọc theo article_type, còn hàng hay không).
    """
    df = pd.read_csv(path)
    chunks = []

    for _, row in df.iterrows():
        content = (
            f"Mã sản phẩm: SP{row['id']}\n"
            f"{row['name_vi']} | {row['name_en']}\n"
            f"Loại: {row['type_vi']} | Type: {row['articleType']}\n"
            f"Màu: {row['color_vi']} | Color: {row['baseColour']}\n"
            f"Giới tính: {row['gender_vi']} | Gender: {row['gender']}\n"
            f"Size: {row['size']}\n"
            f"Giá: {row['price']:,} VND | Price: {row['price']:,}\n"
            f"Tồn kho: {row['stock']} | Stock: {row['stock']}\n"
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
                "article_type": str(row["articleType"]),
                "base_colour": str(row["baseColour"]),
                "gender": str(row["gender"]),
                "sub_category": str(row["subCategory"]),
                "size": str(row["size"]),
                "article_type_lower": str(row["articleType"]).lower(),
                "base_colour_lower": str(row["baseColour"]).lower(),
                "gender_lower": str(row["gender"]).lower(),
                "in_stock": bool(int(row["stock"]) > 0),
            },
        })

    return chunks


def _split_text(text, chunk_size, overlap):
    """
    Chunk văn bản dài thành các đoạn chồng lấp theo số từ.
    Chunk theo số từ (proxy đơn giản cho token), có overlap để không cắt đứt ngữ cảnh.
    Câu/ý ở ranh giới giữa 2 chunk vẫn xuất hiện đầy đủ trong ít nhất 1 chunk nhờ phần chồng lấp.
    """
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


def load_policies_as_chunks(dir_path=POLICIES_DIR):
    """
    Đọc tất cả file .md trong thư mục policy và chunk hoá chúng.
    Mỗi file .md (vd chính sách đổi trả, vận chuyển) được đọc toàn bộ rồi cắt nhỏ bằng _split_text() 
    theo cấu hình POLICY_CHUNK_SIZE/OVERLAP, vì các văn bản chính sách thường dài hơn nhiều so với 
    1 embedding có thể biểu diễn tốt trong 1 lần.
    """
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
    Encode danh sách chunks thành embedding và upsert vào ChromaDB.
    Dùng SentenceTransformer để encode nội dung chunk (có thêm prefix dành cho passage nếu model yêu cầu, 
    Vd BGE-style "passage: "), normalize embedding để cosine similarity là metric đúng về mặt toán học 
    (distance ChromaDB trả về nằm trong [0, 2], convert sang similarity bằng công thức 1 - distance). 
    Tự động bật multiprocess encoding nếu số lượng chunk đủ lớn và máy có nhiều CPU core.
    """
    if not chunks:
        print("Không có chunk nào để index.")
        return

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": CHROMA_SPACE},
    )

    ids = [c["id"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Prefix chỉ để encode, documents upsert vào Chroma vẫn là text gốc.
    to_encode = [EMBEDDING_PASSAGE_PREFIX + d for d in documents]

    if use_multiprocess is None:
        use_multiprocess = len(documents) > 1000 and os.cpu_count() and os.cpu_count() > 1

    print(f"Encoding {len(documents)} chunks (batch_size={batch_size}, multiprocess={use_multiprocess})...")

    if use_multiprocess:
        pool = model.start_multi_process_pool()
        embeddings = model.encode(
            to_encode, pool=pool, batch_size=batch_size, normalize_embeddings=True
        ).tolist()
        model.stop_multi_process_pool(pool)
    else:
        embeddings = model.encode(
            to_encode, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
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
    """
    Luồng xử lý:
    - set_seed() - cố định seed (giữ để nhất quán với các script khác trong pipeline và phòng hờ nếu sau này thêm bước có random).
    - Load chunks từ products.csv (load_products_as_chunks).
    - Load chunks từ các file policy .md (load_policies_as_chunks).
    - Gộp cả 2 nguồn, encode và upsert vào ChromaDB (build_and_persist_index).
    - In tổng số chunk đã index.
    """
    set_seed()
    chunks = []
    chunks += load_products_as_chunks()
    chunks += load_policies_as_chunks()
    build_and_persist_index(chunks, use_multiprocess=False)
    print(f"Indexed {len(chunks)} chunks into {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()