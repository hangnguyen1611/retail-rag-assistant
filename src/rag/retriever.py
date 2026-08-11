import chromadb
from sentence_transformers import SentenceTransformer

from config.backend import (
    CHROMA_COLLECTION_NAME,
    CHROMA_SPACE,
    EMBEDDING_QUERY_PREFIX,
    RETRIEVE_K_POLICY,
    RETRIEVE_K_PRODUCT,
    SPLIT_BY_DOC_TYPE,
)


def _distance_to_cosine_sim(distance, space="cosine"):
    """
    Convert khoảng cách ChromaDB trả về thành điểm cosine similarity.

    ChromaDB trả về "distance" (càng nhỏ càng giống nhau) nhưng để dễ hiểu/so sánh/threshold ta cần "similarity".
    Công thức chuyển đổi phụ thuộc vào metric (hnsw:space) mà collection dùng khi index:
    - "l2": distance là squared L2 trên vector đã normalize (unit vector), nên có quan hệ toán học: d^2 = 2 - 2*cos(θ)
    => cos(θ) = 1 - d^2/2 = 1 - distance/2. Vì ChromaDB trả "distance" ở đây thực chất đã là squared L2.
    - "cosine" hoặc "ip" (inner product trên vector đã normalize): ChromaDB định nghĩa cosine distance = 1 - cosine_similarity
    => chỉ cần đảo ngược lại: sim = 1 - distance.
    """
    if space == "l2":
        sim = 1 - distance / 2      
    else:                           # "cosine" hoặc "ip"
        sim = 1 - distance
    return max(-1.0, min(1.0, sim))


def _combine_where(*wheres):
    """
    Gộp nhiều điều kiện `where` (filter metadata) của ChromaDB thành 1.
    ChromaDB yêu cầu combine nhiều điều kiện filter bằng toán tử "$and" khi truyền query. 
    Hàm này lọc bỏ các điều kiện rỗng/None, rồi tự quyết định trả về None (không filter gì), 1 điều kiện duy nhất
    (không cần bọc $and) hay dict {"$and": [...]} khi có từ 2 điều kiện trở lên.
    """
    parts = [w for w in wheres if w]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


class Retriever:
    def __init__(self, persist_dir, embedding_model, top_k=5):
        """Khởi tạo Retriever: load embedding model và kết nối tới ChromaDB collection"""
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.top_k = top_k

        self._model = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": CHROMA_SPACE},
        )
        self._space = (self._collection.metadata or {}).get("hnsw:space", CHROMA_SPACE)

    def search(self, query, top_k=None, product_filter=None):
        """
        Tìm kiếm các chunk liên quan nhất tới 1 câu query duy nhất.
        Wrapper tiện lợi quanh search_many() cho trường hợp chỉ có 1 query 
        (tránh người gọi phải tự bọc query vào list rồi lấy phần tử [0] ra).
        """
        return self.search_many([query], top_k=top_k, product_filter=product_filter)[0]

    def search_many(self, queries,top_k=None, batch_size=64, product_filter=None):
        """
        Tìm kiếm các chunk liên quan nhất cho nhiều câu query cùng lúc (batch).

        Encode tất cả query 1 lần (hiệu quả hơn encode từng cái), rồi query ChromaDB theo 1 trong 2 chiến lược tùy SPLIT_BY_DOC_TYPE:
        - SPLIT_BY_DOC_TYPE=False: query 1 lần trên toàn collection (có thể lọc theo product_filter nếu có), trả về top_k kết quả bất kể
        doc_type là product hay policy — nghĩa là 2 loại tài liệu cạnh tranh trực tiếp nhau theo điểm similarity.
        - SPLIT_BY_DOC_TYPE=True: query RIÊNG cho từng doc_type (product và policy), mỗi loại lấy k riêng theo RETRIEVE_K_PRODUCT/
        RETRIEVE_K_POLICY, rồi GỘP kết quả lại và sort theo score. Đảm bảo luôn có đại diện của cả 2 loại tài liệu trong kết quả cuối (tránh
        trường hợp product áp đảo hết vì catalog rất lớn so với vài file policy, khiến policy không bao giờ lọt vào top_k nếu query chung)
        """
        k = top_k or self.top_k
        embeddings = self._encode(queries, batch_size)

        if not SPLIT_BY_DOC_TYPE:
            where = _combine_where({"doc_type": "product"} if product_filter else None, product_filter)
            return self._query(embeddings, k, where, len(queries))

        merged = [[] for _ in queries]
        for doc_type, k_type in (("product", RETRIEVE_K_PRODUCT), ("policy", RETRIEVE_K_POLICY)):
            if k_type <= 0:
                continue
            doc_where = {"doc_type": doc_type}
            if doc_type == "product":
                doc_where = _combine_where(doc_where, product_filter)
            part = self._query(embeddings, k_type, doc_where, len(queries))
            for i, hits in enumerate(part):
                merged[i].extend(hits)

        for hits in merged:
            hits.sort(key=lambda h: h["score"], reverse=True)
        return merged

    def _encode(self, queries, batch_size):
        """
        Encode danh sách query thành embedding vector đã normalize.
        Thêm prefix dành cho query (EMBEDDING_QUERY_PREFIX) trước khi encode, khác với prefix dùng khi index passage
        (EMBEDDING_PASSAGE_PREFIX ở bước build index) — vì các embedding model kiểu instruction-tuned (vd BGE) cần 
        prefix riêng biệt cho query vs passage để tối ưu không gian vector cho tác vụ retrieval.
        """
        return self._model.encode(
            [EMBEDDING_QUERY_PREFIX + q for q in queries],
            normalize_embeddings=True,
            batch_size=batch_size,
        ).tolist()

    def _query(self, embeddings, n_results, where, n_queries):
        """Gọi ChromaDB collection.query() và format lại kết quả thô thành dict dễ dùng"""
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