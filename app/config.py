"""Central place for all settings. Values come from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

COMPANY_NAME = "The Chronos Collective"
SUPPORT_EMAIL = "support@thechronoscollective.com"

# LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "smartsupport")

# RAG settings
TOP_K = int(os.getenv("TOP_K", "4"))                                   # chunks sent to the LLM
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.62"))  # below this = out of scope
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "6"))                   # past messages used as memory
