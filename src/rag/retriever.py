"""
retriever.py

Vector search runtime: nhận query -> trả top-k chunk liên quan từ ChromaDB.

1. Load lại persisted ChromaDB collection (CHROMA_PERSIST_DIR)
2. Embed query bằng cùng EMBEDDING_MODEL đã dùng lúc index (build_index.py)
3. Query top-k, trả về list[{id, content, metadata, score}]
4. Score: convert distance -> cosine similarity ("cao hơn = liên quan hơn").
   Công thức phụ thuộc metric của collection, nên đọc metric từ chính collection
   thay vì hardcode:
     - "cosine" / "ip": Chroma trả về 1 - cos_sim  =>  cos_sim = 1 - distance
     - "l2":            Chroma trả về SQUARED L2 (không phải L2 thường!). Với
                        vector đơn vị: ||a-b||^2 = 2 - 2*cos  =>  cos = 1 - d/2
"""

from sentence_transformers import SentenceTransformer
import chromadb

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_SPACE,
    EMBEDDING_QUERY_PREFIX,
    RETRIEVE_K_POLICY,
    RETRIEVE_K_PRODUCT,
    SPLIT_BY_DOC_TYPE,
)

def _distance_to_cosine_sim(distance: float, space: str = "cosine") -> float:
    """Convert Chroma distance -> cosine similarity, theo metric của collection."""
    if space == "l2":
        sim = 1 - distance / 2      # distance là squared L2 trên vector đã normalize
    else:                            # "cosine" hoặc "ip"
        sim = 1 - distance
    return max(-1.0, min(1.0, sim))


class Retriever:
    def __init__(self, persist_dir: str, embedding_model: str, top_k: int = 5):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.top_k = top_k

        self._model = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": CHROMA_SPACE},
        )
        # Collection có sẵn từ trước có thể đang dùng metric khác -> tôn trọng nó.
        self._space = (self._collection.metadata or {}).get("hnsw:space", CHROMA_SPACE)

    def search(self, query: str, top_k: int | None = None):
        """Trả về list[dict] đã rank theo độ liên quan giảm dần:
        {"id": str, "content": str, "metadata": dict, "score": float}
        """
        return self.search_many([query], top_k=top_k)[0]

    def search_many(self, queries: list[str], top_k: int | None = None, batch_size: int = 64):
        """Batch nhiều query trong MỘT lần encode.

        Nếu SPLIT_BY_DOC_TYPE: chạy hai truy vấn có `where` lọc doc_type và giữ
        slot riêng cho product/policy. Không có bước này thì 15 chunk policy
        phải cạnh tranh cùng ranking với 5.000 chunk sản phẩm và gần như luôn
        thua ở những câu dùng từ vựng miền sản phẩm.
        """
        k = top_k or self.top_k
        embeddings = self._encode(queries, batch_size)

        if not SPLIT_BY_DOC_TYPE:
            return self._query(embeddings, k, None, len(queries))

        merged = [[] for _ in queries]
        for doc_type, k_type in (("product", RETRIEVE_K_PRODUCT),
                                 ("policy", RETRIEVE_K_POLICY)):
            if k_type <= 0:
                continue
            part = self._query(embeddings, k_type, {"doc_type": doc_type}, len(queries))
            for i, hits in enumerate(part):
                merged[i].extend(hits)

        for hits in merged:
            hits.sort(key=lambda h: h["score"], reverse=True)
        return merged

    def _encode(self, queries, batch_size):
        return self._model.encode(
            [EMBEDDING_QUERY_PREFIX + q for q in queries],
            normalize_embeddings=True,
            batch_size=batch_size,
        ).tolist()

    def _query(self, embeddings, n_results, where, n_queries):
        kwargs = {
            "query_embeddings": embeddings,
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)

        out = []
        for i in range(n_queries):
            hits = []
            for doc_id, content, metadata, distance in zip(
                result["ids"][i], result["documents"][i],
                result["metadatas"][i], result["distances"][i],
            ):
                hits.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "score": _distance_to_cosine_sim(distance, self._space),
                })
            out.append(hits)
        return out
