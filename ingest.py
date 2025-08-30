# ingest.py
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
# Use cleaned TEXT files produced by your extraction/cleanup pipeline
DOC_ROOT = Path("data/clean_final")   # <-- change if you keep your final texts elsewhere
DB_DIR = "chroma_db/augustine"        # keep separate; safe to change to "chroma_db" if you prefer
COLLECTION = "augustine"

# Chunking tuned for Augustine's long sentences/paragraphs
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# --------- Env / API key ---------
load_dotenv()  # read .env if present
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Put it in your .env (OPENAI_API_KEY=sk-...) "
        "or set it in the environment before running ingest.py."
    )

# --------- Helpers ---------
_ws_collapse = re.compile(r"[ \t\u00A0]+")  # collapse weird intra-line spaces

def normalize_text(t: str) -> str:
    """
    Light cleanup to help chunking & retrieval:
    - Collapse runs of spaces/tabs/nbsp
    - Strip trailing spaces on lines
    - Keep newlines (preserves paragraph structure for better splits)
    """
    if not t:
        return ""
    lines = [_ws_collapse.sub(" ", ln).rstrip() for ln in t.splitlines()]
    return "\n".join(lines).strip()

def content_hash(text: str, meta: Dict) -> str:
    """
    Stable hash of content + a few metadata fields to avoid dupes.
    """
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    h.update(("|" + meta.get("work_title", "") + "|" + meta.get("author", "")).encode("utf-8"))
    return h.hexdigest()[:16]

def load_and_split_docs(root: Path) -> List[Document]:
    """
    Load *.txt files from DOC_ROOT, attach metadata, and split into chunks.
    """
    if not root.exists():
        raise RuntimeError(f"Input folder not found: {root.resolve()}")

    txt_paths = sorted(list(root.rglob("*.txt")))
    if not txt_paths:
        raise RuntimeError(f"No .txt files found under {root.resolve()}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks: List[Document] = []
    for p in txt_paths:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        norm = normalize_text(raw)
        if not norm:
            continue

        # One big document per file; splitting happens next
        base_meta = {
            "author": "Augustine of Hippo",
            "work_title": p.stem,      # e.g., Confessions_Bk10
            "source_path": str(p),
        }
        doc = Document(page_content=norm, metadata=base_meta)

        chunks = splitter.split_documents([doc])

        kept: List[Document] = []
        for idx, ch in enumerate(chunks):
            if not ch.page_content.strip():
                continue
            ch.metadata.update({
                "chunk_id": f"{p.stem}:{idx:05d}"
            })
            kept.append(ch)

        all_chunks.extend(kept)

    # De-duplicate by content hash (helps if multiple files overlap)
    deduped: List[Document] = []
    seen = set()
    for d in all_chunks:
        h = content_hash(d.page_content, d.metadata)
        if h in seen:
            continue
        d.metadata["content_hash"] = h
        seen.add(h)
        deduped.append(d)

    print(f"[INGEST] Found {len(txt_paths)} text files under {root}.")
    print(f"[INGEST] Produced {len(all_chunks)} chunks; kept {len(deduped)} unique after de-dup.")
    return deduped

def rebuild_vectorstore():
    """Wipe and rebuild the persistent Chroma index from /data/clean_final."""
    # Hard wipe to avoid stale collections
    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR, ignore_errors=True)

    docs = load_and_split_docs(DOC_ROOT)
    if not docs:
        raise RuntimeError("No documents were loaded. Check your /data/clean_final folder.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",   # match your Lewis bot if you used large; downgrade if needed
        api_key=OPENAI_API_KEY,           # ingest runs outside Streamlit; pass key explicitly
    )

    _ = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,              # keep same kwarg style as your working setup
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
    )

    print(f"[INGEST] Vector store rebuilt at '{DB_DIR}' (collection '{COLLECTION}').")

if __name__ == "__main__":
    rebuild_vectorstore()
