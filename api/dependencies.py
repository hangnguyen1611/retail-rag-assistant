"""
dependencies.py

Khởi tạo Retriever + Generator MỘT LẦN lúc startup, inject vào routes
qua FastAPI Depends() -- tránh load lại model mỗi request (latency).
"""

import os

_retriever = None
_generator = None


def init_dependencies():
    global _retriever, _generator
    from src.rag.retriever import Retriever
    from src.rag.generator import Generator

    _retriever = Retriever(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    )
    _generator = Generator()


def get_retriever():
    return _retriever


def get_generator():
    return _generator
