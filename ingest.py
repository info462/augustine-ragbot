# ingest.py
import os
import shutil
from pathlib import Path
from typing import List
from chromadb.config import Settings
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

# when you create/load Chroma (in BOTH files)
Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    client_settings=CHROMA_SETTINGS,
)

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# ---------- Constants (app + ingest must match) ----------
DATA_DIR = Path("data/clean_final")
PERSIST_DIR = "chroma_db/augustine"
COLLECTION_NAME = "augustine"
EMBED_MODEL = "text-embedding-3-small"  # cheap + good

def _read_all_txt_files(root: Path) -> List[Document]:
    docs: List[Document] = []
    for p in sorted(root.rglob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs

def rebuild_vectorstore() -> int:
    """
    Wipe and rebuild the persistent Chroma index from DATA_DIR (.txt files).
    Returns number of chunks written.
    """
    # 0) Sanity
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR.resolve()}")

    # 1) Fresh start
    if Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)

    # 2) Load raw docs
    raw_docs = _read_all_txt_files(DATA_DIR)
    if not raw_docs:
        raise RuntimeError(f"No .txt files found under {DATA_DIR}")

    # 3) Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)

    # 4) Embed + persist
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    vs.persist()
    return len(chunks)
