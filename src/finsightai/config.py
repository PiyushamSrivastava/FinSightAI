import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Models
LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "jina-embeddings-v5-text-small"

# ChromaDB configuration
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

# New collection: do not reuse the Gemini collection
COLLECTION_NAME = "finsight_jina_v5"





def get_secret(key):
    value = os.getenv(key)

    if value:
        return value

    try:
        return st.secrets.get(key)
    except Exception:
        return None