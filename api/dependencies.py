"""
dependencies.py

Khởi tạo Retriever + Generator MỘT LẦN lúc startup, inject vào routes
qua FastAPI Depends() -- tránh load lại model mỗi request (latency).
"""

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, TOP_K

_retriever = None
_generator = None


def init_dependencies():
    global _retriever, _generator
    from src.rag.retriever import Retriever
    from src.rag.generator import Generator

    _retriever = Retriever(
        persist_dir=CHROMA_PERSIST_DIR,
        embedding_model=EMBEDDING_MODEL,
        top_k=TOP_K,
    )
    _generator = Generator()


def get_retriever():
    return _retriever


def get_generator():
    return _generator