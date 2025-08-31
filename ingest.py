# ingest.py (TXT-only)
from __future__ import annotations

import os
import re
import shutil
import hashlib
import logging
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

logging.getLogger("pypdf").setLevel(logging.ERROR)

# --------- Config (Augustine) ---------
DOC_ROOT = Path("data/clean_final")    # folder with your cleaned .txt files
DB_DIR = "chroma_db/augustine"         # app looks here first (with fallback)
COLLECTION = "augustine"

# Chunking tuned for Augustine
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# --------- Env / API key ---------
# Streamlit app sets OPENAI_API_KEY in env before calling rebuild_vectorstore()
load_dotenv()  # for local runs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Put it in your .env (OPENAI_API_KEY=sk-...) "
        "or let app.py export it to the environment before calling ingest."
    )

# --------- Helpers ---------
_ws_collapse = re.compile(r"[ \t\u00A0]+")

def normalize_text(t: str) -> str:
    """Collapse odd spaces, trim line tails, keep paragraph breaks."""
    if not t:
        return ""
    lines = [_ws_collapse.sub(" ", ln).rstrip() for ln in t.splitlines()]
    return "\n".join(lines).strip()

def content_hash(text: str, meta: Dict) -> str:
    """Stable hash to de-dupe identical content across files."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    h.update(("|" + meta.get("work_title", "") + "|" + meta.get("author", "")).encode("utf-8"))
    return h.hexdigest()[:16]

def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

def _load_txt_like_file(p: Path) -> List[Document]:
    """Load a .txt (or .md) file and split into chunks with consistent metadata."""
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    norm = normalize_text(text)
    if not norm:
        return []
    base_meta = {
        "author": "Augustine of Hippo",
        "work_title": p.stem,     # e.g., Confessions_Bk10
        "source_path": str(p),
    }
    doc = Document(page_content=norm, metadata=base_meta)
    chunks = _splitter().split_documents([doc])
    # add stable IDs & page-like label for UI
    for idx, ch in enumerate(chunks):
        ch.metadata.setdefault("chunk_id", f"{p.stem}:{idx:05d}")
        # optional: expose a human-ish page label for the UI
        ch.metadata.setdefault("page", f"chunk {idx+1}")
    return chunks

def load_and_split_docs(root: Path) -> List[Document]:
    """Load *.txt (and optionally *.md) under DOC_ROOT and return deduped chunks."""
    if not root.exists():
        raise RuntimeError(f"Input folder not found: {root.resolve()}")

    txt_paths = sorted(root.rglob("*.txt"))
    md_paths  = sorted(root.rglob("*.md"))  # remove this line if you truly want txt-only

    if not (txt_paths or md_paths):
        raise RuntimeError(f"No .txt/.md files found under {root.resolve()}")

    all_chunks: List[Document] = []
    for p in txt_paths + md_paths:
        all_chunks.extend(_load_txt_like_file(p))

    # De-duplicate by content hash
    deduped: List[Document] = []
    seen = set()
    for d in all_chunks:
        h = content_hash(d.page_content, d.metadata)
        if h in seen:
            continue
        d.metadata["content_hash"] = h
        seen.add(h)
        deduped.append(d)

    print(f"[INGEST] Found {len(txt_paths)} .txt and {len(md_paths)} .md under {root}.")
    print(f"[INGEST] Produced {len(all_chunks)} chunks; kept {len(deduped)} unique after de-dup.")
    return deduped

def rebuild_vectorstore():
    """Wipe and rebuild the persistent Chroma index from DOC_ROOT (txt only)."""
    # Clear existing to avoid stale state
    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR, ignore_errors=True)

    docs = load_and_split_docs(DOC_ROOT)
    if not docs:
        raise RuntimeError("No documents were loaded. Check your data folder.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"  # key picked up from env
    )

    _ = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,            # correct kwarg for langchain_community
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
    )

    print(f"[INGEST] Vector store rebuilt at '{DB_DIR}' (collection '{COLLECTION}').")
    print(f"[INGEST] Loaded {len(docs)} docs into vectorstore.")
    return len(docs)

if __name__ == "__main__":
    rebuild_vectorstore()
