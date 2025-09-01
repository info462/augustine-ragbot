# ingest.py (FAISS)
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- Constants (must match app.py) ----------
DATA_DIR = Path("data/clean_final")          # your .txt files
FAISS_DIR = "faiss_index/augustine"          # folder to save index
INDEX_NAME = "index"                         # base filename for FAISS files
EMBED_MODEL = "text-embedding-3-small"

def _read_all_txt_files(root: Path) -> List[Document]:
    docs: List[Document] = []
    for p in sorted(root.rglob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs

def rebuild_vectorstore() -> int:
    """
    Build a FAISS index from DATA_DIR and save under FAISS_DIR/INDEX_NAME.
    Returns number of chunks written.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR.resolve()}")

    raw_docs = _read_all_txt_files(DATA_DIR)
    if not raw_docs:
        raise RuntimeError(f"No .txt files found under {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    # tag chunks for nicer source display
    for i, d in enumerate(chunks):
        d.metadata["chunk"] = i

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)

    Path(FAISS_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(FAISS_DIR, index_name=INDEX_NAME)
    return len(chunks)
