from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    """Save an uploaded Streamlit file to dest_dir and return Path."""
    dest = dest_dir / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def load_and_split_pdf(path: Path, chunk_size=1000, chunk_overlap=200):
    """Load a PDF and split into chunks. Attach metadata 'source' and 'page'.

    This implementation assumes PyPDFLoader returns page-level Documents. If your
    LangChain version behaves differently, the code still enumerates pages and
    assigns page numbers.
    """
    loader = PyPDFLoader(str(path))
    page_docs = loader.load()  # often returns one Document per page; if not, adjust

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for i, page_doc in enumerate(page_docs):
        page_num = page_doc.metadata.get("page", i + 1)
        page_doc.metadata["source"] = path.name
        page_doc.metadata["page"] = int(page_num)
        # Chunk the page (keeps page-level mapping)
        chunks = splitter.split_documents([page_doc])
        for c in chunks:
            c.metadata.setdefault("source", path.name)
            c.metadata.setdefault("page", int(page_num))
            all_chunks.append(c)
    return all_chunks
