import os
import random
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR                = "data"
RAW_PRODUCTS_PATH       = f"{DATA_DIR}/raw/styles.csv"
PROCESSED_PRODUCTS_PATH = f"{DATA_DIR}/processed/products.csv"
POLICIES_DIR            = f"{DATA_DIR}/processed/policies"
EVAL_SET_PATH           = f"{DATA_DIR}/eval/eval_set.csv"
EVAL_RESULTS_PATH       = f"{DATA_DIR}/eval/results.csv"

# --- Vector store / embedding ---
CHROMA_PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME  = "retail_assistant"
CHROMA_SPACE            = "cosine"   # set tường minh, không dựa vào default "l2" của Chroma
EMBEDDING_MODEL          = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_QUERY_PREFIX   = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")

# --- Retrieval ---
TOP_K                = 5
# Policy chỉ có ~15 chunk trên tổng ~5015. Để chung một ranking thì câu hỏi
# policy bị chunk sản phẩm nhấn chìm mỗi khi nó dùng từ ngữ miền sản phẩm
# ("size L", "trousers", "sản phẩm da"). Giữ slot riêng cho từng doc_type.
# Tổng = 3 + 2 = TOP_K nên kích thước context không đổi.
SPLIT_BY_DOC_TYPE    = os.getenv("SPLIT_BY_DOC_TYPE", "1") == "1"
RETRIEVE_K_PRODUCT   = int(os.getenv("RETRIEVE_K_PRODUCT", "3"))
RETRIEVE_K_POLICY    = int(os.getenv("RETRIEVE_K_POLICY", "2"))
POLICY_CHUNK_SIZE    = 150   # ~token, dùng khi chunk policy docs
POLICY_CHUNK_OVERLAP = 30

# --- LLM (Groq) ---
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = 0.2   # thấp để giảm hallucination, ưu tiên bám context

# --- App ---
DEFAULT_LANGUAGE = "auto"   # "vi" | "en" | "auto"
 
# --- Reproducibility ---
SEED = 42
 
 
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
