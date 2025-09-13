# faiss_utils.py
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path
from typing import List
from langchain.schema import Document

def build_faiss_index(docs: List[Document], embeddings_model: str, faiss_dir: Path):
    """
    Build and persist a FAISS index using the LangChain Google-GenAI embeddings wrapper.
    embeddings_model should be something like 'models/gemini-embedding-001' or 'gemini-embedding-001'
    depending on the naming you prefer.
    """
    if not docs:
        return None
    # LangChain's GoogleGenerativeAIEmbeddings wraps the GenAI SDK
    embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(faiss_dir))
    return vs

def load_faiss_if_exists(embeddings_model: str, faiss_dir: Path):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
        vs = FAISS.load_local(str(faiss_dir), embeddings)
        return vs
    except Exception:
        return None
