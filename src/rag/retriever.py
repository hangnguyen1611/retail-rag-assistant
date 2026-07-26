"""
retriever.py

Vector search runtime: nhận query -> trả top-k chunk liên quan từ ChromaDB.

1. Load lại persisted ChromaDB collection (CHROMA_PERSIST_DIR)
2. Embed query bằng cùng EMBEDDING_MODEL đã dùng lúc index (build_index.py)
3. Query top-k, trả về list[{id, content, metadata, score}]
4. Score: build_index.py không set metric khi tạo collection nên Chroma
   dùng mặc định "l2". Vì embedding đã normalize_embeddings=True (build_index.py),
   với vector đơn vị: l2_distance^2 = 2 - 2*cos_sim  =>  cos_sim = 1 - l2^2/2.
   Ta convert distance -> cosine similarity để "score" có nghĩa "cao hơn = liên quan hơn".
"""

from sentence_transformers import SentenceTransformer
import chromadb

from src.config import CHROMA_COLLECTION_NAME


def _distance_to_cosine_sim(distance: float) -> float:
    """Chroma default space là l2 trên embedding đã normalize -> suy ra cosine similarity."""
    sim = 1 - (distance ** 2) / 2
    return max(-1.0, min(1.0, sim))


class Retriever:
    def __init__(self, persist_dir: str, embedding_model: str, top_k: int = 5):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.top_k = top_k

        self._model = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    def search(self, query: str, top_k: int | None = None):
        """Trả về list[dict] đã rank theo độ liên quan giảm dần:
        {"id": str, "content": str, "metadata": dict, "score": float}
        """
        k = top_k or self.top_k

        query_embedding = self._model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits = []
        for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            hits.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "score": _distance_to_cosine_sim(distance),
            })

        return hits
