from config.backend import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, TOP_K

_retriever = None
_generator = None
_condenser = None


def init_dependencies():
    """
    Khởi tạo Retriever và Generator, gán vào biến toàn cục module.

    Gọi hàm này đúng 1 lần lúc FastAPI khởi động, trước khi bất kỳ request nào được xử lý. Sau khi gọi, get_retriever()/get_generator() sẽ trả
    về các instance đã khởi tạo sẵn, tránh phải load lại model embedding (SentenceTransformer) hoặc kết nối lại ChromaDB/Groq client mỗi request,
    chi phí khởi tạo model embedding đặc biệt cao (vài giây), không thể chấp nhận được nếu lặp lại cho mỗi request.

    Import Generator/Retriever được đặt bên trong hàm thay vì đầu file để tránh import các thư viện nặng (sentence_transformers,
    chromadb, groq...) ngay khi module dependencies.py được import — chỉ thực sự tải chúng khi init_dependencies() được gọi (thường lúc app
    khởi động thật, không phải lúc chỉ import module để test/kiểm tra routing).

    Side effects: Gán giá trị cho 2 biến toàn cục module-level _retriever và _generator thay thế giá trị None ban đầu.
    """
    global _retriever, _generator, _condenser
    from src.rag.condense import QueryCondenser
    from src.rag.generator import Generator
    from src.rag.retriever import Retriever

    _retriever = Retriever(
        persist_dir=CHROMA_PERSIST_DIR,
        embedding_model=EMBEDDING_MODEL,
        top_k=TOP_K,
    )
    _generator = Generator()
    _condenser = QueryCondenser()


def get_retriever():
    """
    Trả về instance Retriever đã khởi tạo sẵn (dependency cho FastAPI).

    Dùng làm callable trong FastAPI Depends(get_retriever) tại các route handler — FastAPI sẽ tự gọi hàm này cho mỗi request và truyền kết
    quả vào tham số tương ứng của route (VD retriever=Depends(get_retriever) trong chat.py).
    Phải gọi init_dependencies() trước đó (lúc app startup), nếu không hàm này sẽ trả về None — các route dùng Depends(get_retriever) sẽ
    nhận None thay vì Retriever thật, dẫn tới lỗi AttributeError khi cố gọi phương thức trên đó (vd retriever.search(...)).
    """
    if _retriever is None:
        raise RuntimeError("init_dependencies() chưa được gọi trước khi xử lý request.")
    return _retriever


def get_generator():
    """
    Trả về instance Generator đã khởi tạo sẵn (dependency cho FastAPI).
    Tương tự get_retriever(), dùng làm Depends(get_generator) trong route handler. Cùng yêu cầu init_dependencies() phải được gọi trước đó.
    """
    if _generator is None:
        raise RuntimeError("init_dependencies() chưa được gọi trước khi xử lý request.")
    return _generator


def get_condenser():
    """
    Trả về instance QueryCondenser đã khởi tạo sẵn (dependency cho FastAPI).
    Tương tự get_retriever()/get_generator(), cùng yêu cầu init_dependencies() phải được gọi trước đó.
    """
    if _condenser is None:
        raise RuntimeError("init_dependencies() chưa được gọi trước khi xử lý request.")
    return _condenser