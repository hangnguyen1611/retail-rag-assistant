import os
import random
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR                = "data"
RAW_DATA_PATH           = f"{DATA_DIR}/raw/styles.csv"
CLEAN_DATA_PATH         = f"{DATA_DIR}/processed/clean_styles.csv"
PROCESSED_PRODUCTS_PATH = f"{DATA_DIR}/processed/products.csv"
POLICIES_DIR            = f"{DATA_DIR}/processed/policies"
EVAL_SET_PATH           = f"{DATA_DIR}/eval/eval_set.csv"
EVAL_RESULTS_PATH       = f"{DATA_DIR}/eval/results.csv"

# --- Vector store / embedding ---
CHROMA_PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME  = "retail_assistant"
CHROMA_SPACE            = "cosine"
EMBEDDING_MODEL          = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_QUERY_PREFIX   = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")

# --- Retrieval ---
SPLIT_BY_DOC_TYPE  = os.getenv("SPLIT_BY_DOC_TYPE", "1") == "1"
RETRIEVE_K_PRODUCT = int(os.getenv("RETRIEVE_K_PRODUCT", "3"))
RETRIEVE_K_POLICY  = int(os.getenv("RETRIEVE_K_POLICY", "2"))

TOP_K = (RETRIEVE_K_PRODUCT + RETRIEVE_K_POLICY) if SPLIT_BY_DOC_TYPE else int(os.getenv("TOP_K", "5"))

POLICY_CHUNK_SIZE    = 150   # ~350 token với tiếng Việt, an toàn dưới 512
POLICY_CHUNK_OVERLAP = 30

# --- LLM (Groq) ---
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")
GROQ_MODEL            = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE      = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
GROQ_MAX_TOKENS       = int(os.getenv("GROQ_MAX_TOKENS", "600"))
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

CONDENSE_MODEL = os.getenv("CONDENSE_MODEL", "llama-3.1-8b-instant")
ENABLE_CONDENSE = os.getenv("ENABLE_CONDENSE", "1") == "1"
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# --- App ---
DEFAULT_LANGUAGE = "auto"   # "vi" | "en" | "auto"

# --- Reproducibility ---
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
