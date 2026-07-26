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
EMBEDDING_MODEL         = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- Retrieval ---
TOP_K                = 5
POLICY_CHUNK_SIZE    = 250   # ~token, dùng khi chunk policy docs
POLICY_CHUNK_OVERLAP = 50

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
    try:
        np.random.seed(seed)
    except ImportError:
        pass
