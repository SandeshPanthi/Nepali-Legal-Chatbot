"""
Central configuration for the Legal RAG pipeline.

All paths and constants that were hard-coded inline in the original
notebook are collected here so they only need to be changed in one place.
"""

from pathlib import Path

# Project root (folder containing this file)
BASE_DIR = Path(__file__).resolve().parent

# --- Data locations -------------------------------------------------------
DATA_DIR = BASE_DIR / "data" / "acts"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

CIVIL_CODE_PATH = DATA_DIR / "civil_code.pdf"
PENAL_CODE_PATH = DATA_DIR / "penal_code.pdf"
CONSTITUTION_PATH = DATA_DIR / "constitution.pdf"

# --- Embedding model --------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Vector store -----------------------------------------------------------
COLLECTION_NAME = "pdf_documents"

# --- LLM (Groq) ---------------------------------------------------------
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS = 1024

# --- Default retrieval / RAG settings ---------------------------------------
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.0
