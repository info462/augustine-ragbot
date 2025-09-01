# ingest.py (FAISS + diagnostics)
import os
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- Constants (must match app.py) ----------
DATA_DIR = Path("data/clean_final")
FAISS_DIR = "faiss_index/augustine"
INDEX_NAME = "index"
EMBED_MODEL = "text-embedding-3-small"

def _read_all_txt_files(root: Path) -> List[Document]:
    docs: List[Document] = []
    for p in sorted(root.rglob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs

def rebuild_vectorstore(progress: Optional[Callable[[str], None]] = None) -> int:
    """
    Build a FAISS index from DATA_DIR and save under FAISS_DIR/INDEX_NAME.
    Returns number of chunks written.
    Optionally accepts a progress(str) callback for UI logging.
    """
    log = progress or (lambda _: None)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR.resolve()}")

    log("Reading .txt files…")
    raw_docs = _read_all_txt_files(DATA_DIR)
    if not raw_docs:
        raise RuntimeError(f"No .txt files found under {DATA_DIR}")

    log(f"Found {len(raw_docs)} files. Chunking…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)
    for i, d in enumerate(chunks):
        d.metadata["chunk"] = i
    log(f"Created {len(chunks)} chunks. Initializing embeddings…")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    # quick probe to fail fast if API/key is wrong
    _ = embeddings.embed_query("probe")  # will raise if key/perm is bad
    log("Embeddings OK. Building FAISS index…")

    vs = FAISS.from_documents(chunks, embeddings)

    Path(FAISS_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(FAISS_DIR, index_name=INDEX_NAME)
    log(f"Saved index to {FAISS_DIR} with base name '{INDEX_NAME}'.")
    return len(chunks)
