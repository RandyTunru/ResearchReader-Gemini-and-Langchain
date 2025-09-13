from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from project root
load_dotenv()

# Expose config via simple attributes
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-pro")

# Local storage
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR = BASE_DIR / "faiss_index"
FAISS_DIR.mkdir(parents=True, exist_ok=True)