import streamlit as st
from pathlib import Path
import json
import re

from config.env_var import UPLOAD_DIR, FAISS_DIR, EMBEDDING_MODEL, LLM_MODEL
from config.context_prompt import SYSTEM_CONTEXT_PROMPT
from utils.pdf_utils import save_uploaded_file, load_and_split_pdf
from utils.faiss_utils import build_faiss_index, load_faiss_if_exists
from utils.parser_utils import FORMAT_INSTRUCTIONS, try_parse_structured, fallback_extract

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except Exception:
    raise ImportError("Install langchain-google-genai and langchain: pip install langchain-google-genai langchain")

# Session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "docs" not in st.session_state:
    st.session_state.docs = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.set_page_config(page_title="Gemini PDF RAG", layout="wide")
st.title("PDF Research Assistant with Gemini 2.5 & LangChain")

left, right = st.columns([1, 2])

with left:
    st.header("Upload PDFs")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        added = []
        for f in uploaded:
            if f.name in st.session_state.uploaded_files:
                st.info(f"Already uploaded: {f.name}")
                continue
            path = save_uploaded_file(f, UPLOAD_DIR)
            chunks = load_and_split_pdf(path)
            st.session_state.docs.extend(chunks)
            st.session_state.uploaded_files.append(f.name)
            added.append(f.name)
        if added:
            st.success(f"Added: {', '.join(added)} building index...")
            st.session_state.vectorstore = build_faiss_index(st.session_state.docs, EMBEDDING_MODEL, FAISS_DIR)

    st.markdown("---")
    st.header("Uploaded Files")
    if st.session_state.uploaded_files:
        for name in st.session_state.uploaded_files:
            st.write(f"- {name}")
    else:
        st.write("_No files uploaded._")

with right:
    user_q = st.text_area("Ask a question about the uploaded documents:")
    if st.button("Ask"):
        # lazy load index if app restarted
        if not st.session_state.vectorstore:
            vs = load_faiss_if_exists(EMBEDDING_MODEL, FAISS_DIR)
            if vs:
                st.session_state.vectorstore = vs

        if not st.session_state.vectorstore:
            st.error("No index available. Upload PDFs first.")
        elif not user_q.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Retrieving & querying Gemini..."):
                retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":5})
                llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

                safe_format_instructions = FORMAT_INSTRUCTIONS.replace("{", "{{").replace("}", "}}")

                prompt_template = (
                    SYSTEM_CONTEXT_PROMPT
                    + "\n\nIMPORTANT (do not change):\n"
                    + "- Use ONLY the provided context below. Do not invent facts.\n"
                    + "- For every factual claim include a bracketed citation like: [filename.pdf, p.3]\n"
                    + "- Return the final output EXACTLY in the JSON format described by the format_instructions below.\n\n"
                    + safe_format_instructions
                    + "\n- If you cannot find the answer in the context, say so.\n"
                    + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer (JSON):"
                )
                prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"]) 

                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt}, return_source_documents=True)

                out = qa({"query": user_q})
                raw = out.get("result") or out.get("answer") or ""

                # try structured parse
                try:
                    parsed = try_parse_structured(raw)
                except Exception:
                    parsed = fallback_extract(raw)

                # === Normalize parsed to a dict if needed ===
                if not isinstance(parsed, dict):
                    try:
                        parsed = json.loads(parsed)
                    except Exception:
                        # If parsing fails, fallback to raw text answer
                        parsed = {"answer": str(parsed), "found": False, "citations": [], "follow_up": None}

                # === Normalize citations into list[dict] with keys: source, page, excerpt ===
                raw_citations = parsed.get("citations") or []
                normalized = []

                # Accept dict or string forms as well
                if isinstance(raw_citations, dict):
                    raw_citations = [raw_citations]
                if isinstance(raw_citations, str):
                    raw_citations = [raw_citations]

                for item in raw_citations:
                    if isinstance(item, dict):
                        src = item.get("source") or item.get("file") or item.get("filename") or None
                        page = item.get("page") if item.get("page") is not None else item.get("p") if item.get("p") is not None else None
                        excerpt = item.get("excerpt", "")
                        normalized.append({"source": src, "page": page, "excerpt": excerpt})
                    elif isinstance(item, str):
                        # try bracketed pattern: [paper.pdf, p.3]
                        m = re.findall(r"\[([^,\]]+),\s*p\.?\s*(\d+)\]", item)
                        if m:
                            for src, pg in m:
                                normalized.append({"source": src.strip(), "page": int(pg), "excerpt": ""})
                            continue
                        # try JSON-encoded item
                        try:
                            obj = json.loads(item)
                            if isinstance(obj, dict):
                                normalized.append({"source": obj.get("source"), "page": obj.get("page"), "excerpt": obj.get("excerpt", "")})
                                continue
                            if isinstance(obj, list):
                                for ele in obj:
                                    if isinstance(ele, dict):
                                        normalized.append({"source": ele.get("source"), "page": ele.get("page"), "excerpt": ele.get("excerpt", "")})
                                continue
                        except Exception:
                            pass
                        # fallback: treat the whole string as a filename (no page)
                        normalized.append({"source": item.strip(), "page": None, "excerpt": ""})
                    else:
                        # ignore unexpected types
                        continue

                parsed["citations"] = normalized

                # === Enrich citations using source_documents (if available) ===
                src_docs = out.get("source_documents") or []
                lookup = {}
                for d in src_docs:
                    src = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page")
                    key = (src, int(page)) if page is not None else (src, None)
                    lookup.setdefault(key, d.page_content[:400])

                for c in parsed.get("citations", []):
                    key = (c.get("source"), c.get("page"))
                    if not c.get("excerpt"):
                        c["excerpt"] = lookup.get(key, "")

                # Display
                st.markdown("### Answer")
                st.write(parsed.get("answer"))

                st.markdown("### Citations")
                if parsed.get("citations"):
                    for c in parsed["citations"]:
                        src = c.get("source")
                        page = c.get("page")
                        excerpt = c.get("excerpt", "")[:300]
                        st.write(f"- **{src}** (p.{page}) : {excerpt}...")
                else:
                    st.write("_No citations returned._")

                st.markdown("---")
                st.markdown("### Raw model output (debug)")
                st.code(raw[:4000])

# Try to load persisted index on cold start
if st.session_state.vectorstore is None:
    vs = load_faiss_if_exists(EMBEDDING_MODEL, FAISS_DIR)
    if vs:
        st.session_state.vectorstore = vs
        st.success("Loaded persisted FAISS index from disk.")
