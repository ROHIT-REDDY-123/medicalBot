
"""
Streamlit MediBot (Resume PDF specific)
- Accepts only PDF uploads (saved to ./data)
- Processes PDFs -> chunks -> embeddings -> FAISS DB saved to vectorstore/db_faiss
- Serves a QA UI that answers only from the resume content
"""

import os
import time
from pathlib import Path
from typing import List

import streamlit as st
from pypdf import PdfReader

# LangChain & embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA

# Optional: local Ollama LLM - used as default if available
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# -----------------------
# Config
# -----------------------
DATA_DIR = Path("data")
DB_FAISS_PATH = Path("vectorstore/db_faiss")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
K = 3

# Resume-specific prompt — concise and on-source only
CUSTOM_PROMPT_TEMPLATE = """
You are ResumeAssistant. Use ONLY the provided resume context to answer the question.
If the answer is not present in the resume, reply: "I don't have that information in the resume."

Context:
{context}

Question:
{question}

Answer (concise):
"""

PROMPT = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# -----------------------
# Helpers: File & PDF handling
# -----------------------
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_pdf_to_data(uploaded_file) -> Path:
    """
    Save uploaded PDF file (streamlit UploadedFile) to ./data and return path
    Only accepts .pdf
    """
    ensure_dirs()
    fname = uploaded_file.name
    if not fname.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are accepted.")
    dest = DATA_DIR / fname
    # Overwrite if exists
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest

def extract_text_from_pdf(pdf_path: Path) -> List[str]:
    """
    Return a list of page texts for given PDF path.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return pages

# -----------------------
# Helpers: Chunking & Vectorstore
# -----------------------
def docs_from_pdf(pdf_path: Path) -> List[Document]:
    """
    Read PDF and return list[Document] with metadata: source, page, chunk
    """
    pages = extract_text_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs: List[Document] = []
    chunk_index = 0
    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text or page_text.strip() == "":
            continue
        chunks = splitter.split_text(page_text)
        for c in chunks:
            docs.append(Document(page_content=c, metadata={"source": pdf_path.name, "page": page_idx, "chunk": chunk_index}))
            chunk_index += 1
    return docs

def build_faiss_from_pdfs(pdf_paths: List[Path]):
    """
    Given one or more PDF file paths, build embeddings and a FAISS DB and save locally.
    Overwrites existing DB at DB_FAISS_PATH.
    """
    if not pdf_paths:
        raise ValueError("No PDF paths provided for building FAISS.")
    all_docs: List[Document] = []
    for p in pdf_paths:
        all_docs.extend(docs_from_pdf(p))
    if not all_docs:
        raise RuntimeError("No text extracted from provided PDFs.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(all_docs, embeddings)
    db.save_local(str(DB_FAISS_PATH))
    return db

def load_vectorstore() -> FAISS:
    """
    Load existing FAISS DB if present, otherwise raise.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if DB_FAISS_PATH.exists() and any(DB_FAISS_PATH.iterdir()):
        db = FAISS.load_local(str(DB_FAISS_PATH), embeddings, allow_dangerous_deserialization=True)
        return db
    raise FileNotFoundError("FAISS DB not found. Upload PDF and build index first.")

# -----------------------
# QA Chain
# -----------------------
def create_qa_chain(llm, db: FAISS):
    """
    Create a RetrievalQA chain using the provided LLM and vectorstore db.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

def get_default_llm():
    """
    Return a default LLM. Preference: Ollama local if available; otherwise raise and ask user to configure.
    """
    if OLLAMA_AVAILABLE:
        return OllamaLLM(model=os.getenv("OLLAMA_MODEL", "phi3:mini"), temperature=0.1, num_predict=200)
    raise RuntimeError("No local LLM available. Install or run Ollama, or change get_default_llm to use HF/inference client.")

# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="MediBot — Resume QA (PDF only)", layout="centered")
    st.title("🏥 MediBot — Resume PDF Q&A")
    st.write("Upload a resume PDF. The app will index it and let you ask questions (answers come from the resume only).")

    ensure_dirs()

    # Left: upload & build; Right: chat
    uploaded = st.file_uploader("Upload resume (PDF only)", type=["pdf"], accept_multiple_files=False)
    if uploaded:
        try:
            saved_path = save_pdf_to_data(uploaded)
            st.success(f"Saved to data/{saved_path.name}")
            with st.spinner("Processing PDF and building index (this may take a few seconds)..."):
                db = build_faiss_from_pdfs([saved_path])
            st.success("Index built and saved to vectorstore/db_faiss")
        except Exception as e:
            st.error(f"Failed to process upload: {e}")
            st.stop()

    # Try to load DB (if exists)
    try:
        db = load_vectorstore()
    except FileNotFoundError:
        st.info("No index found — upload a PDF to build one.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        st.stop()

    # LLM
    try:
        llm = get_default_llm()
        st.info("Using local Ollama LLM")
    except Exception as e:
        st.error(str(e))
        st.stop()

    # chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # input
    if prompt_text := st.chat_input("Ask a question about the resume..."):
        st.chat_message("user").markdown(prompt_text)
        st.session_state.messages.append({"role": "user", "content": prompt_text})

        with st.chat_message("assistant"):
            with st.spinner("Searching resume and generating answer..."):
                try:
                    qa_chain = create_qa_chain(llm, db)
                    # invoke chain (invoke preferred; .run fallback)
                    try:
                        resp = qa_chain.invoke({"query": prompt_text})
                        answer = resp.get("result") or resp.get("answer") or str(resp)
                        source_docs = resp.get("source_documents", [])
                    except Exception:
                        answer = qa_chain.run(prompt_text)
                        source_docs = []

                    # format sources snippet
                    src_info = ""
                    for i, d in enumerate(source_docs[:3], 1):
                        meta = d.metadata
                        src_info += f"{i}. {meta.get('source')} (page {meta.get('page')}, chunk {meta.get('chunk')})\n"

                    full = f"{answer}\n\n**Sources:**\n{src_info}"
                    st.markdown(full)
                    st.session_state.messages.append({"role": "assistant", "content": full})
                except Exception as e:
                    st.error(f"Error during QA: {e}")

if __name__ == "__main__":
    main()
