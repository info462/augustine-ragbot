# app.py (FAISS lazy-load + diagnostics)
import os
import shutil
from pathlib import Path
from typing import List, Optional

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ----- Page first to avoid blank UI -----
st.set_page_config(page_title="Ask St. Augustine Anything", layout="wide")

# ---------- Constants (must match ingest.py) ----------
DATA_DIR = Path("data/clean_final")
FAISS_DIR = Path("faiss_index/augustine")
INDEX_NAME = "index"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---------- Secrets ----------
def get_secret(key: str, default: str | None = None) -> str | None:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ELEVEN_API_KEY = get_secret("ELEVENLABS_API_KEY")  # optional
audio_enabled = bool(ELEVEN_API_KEY)

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing. Add it in Streamlit Secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ---------- Helpers ----------
def faiss_files_exist() -> bool:
    return (FAISS_DIR / f"{INDEX_NAME}.faiss").exists() and (FAISS_DIR / f"{INDEX_NAME}.pkl").exists()

def reset_index_folder():
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)

@st.cache_resource(show_spinner=False)
def load_vectordb() -> Optional[FAISS]:
    if not faiss_files_exist():
        return None
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    try:
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        reset_index_folder()
        return None

def collection_count(vs: Optional[FAISS]) -> int:
    try:
        return vs.index.ntotal if vs is not None else 0
    except Exception:
        return 0

# ---------- Sidebar ----------
st.sidebar.header("Dataset")
st.sidebar.write(f"**Index path:** `{FAISS_DIR}/{INDEX_NAME}`")
txts = sorted([str(p) for p in DATA_DIR.rglob("*.txt")])
st.sidebar.write(f"**Found {len(txts)} .txt files under** `data/clean_final`.")
with st.sidebar.expander("Examples", expanded=False):
    st.code(txts[:10] if txts else "[]", language="json")

# Buttons
if st.sidebar.button("Rebuild index from data/clean_final"):
    with st.status("Rebuilding vector store…", expanded=True) as s:
        logs = []
        def log(msg: str):
            logs.append(msg)
            s.update(label="Rebuilding vector store…", state="running")
            st.write("• " + msg)

        try:
            from ingest import rebuild_vectorstore
            chunks = rebuild_vectorstore(progress=log)
            s.update(label=f"Rebuild complete. {chunks} chunks.", state="complete")
            st.success(f"Done. Chunks in index: {chunks}")
        except Exception as e:
            s.update(label="Rebuild failed.", state="error")
            st.error(f"Ingest failed: {e}")
    load_vectordb.clear()

if st.sidebar.button("Reset / delete index folder"):
    reset_index_folder()
    load_vectordb.clear()
    st.sidebar.success("Index folder deleted. Click Rebuild next.")

# Diagnostics
with st.sidebar.expander("Diagnostics", expanded=False):
    if st.button("Test embeddings now"):
        try:
            emb = OpenAIEmbeddings(model=EMBED_MODEL)
            v = emb.embed_query("hello")
            st.success(f"Embeddings OK. Vector length: {len(v)}")
        except Exception as e:
            st.error(f"Embedding call FAILED: {e}")

# ---------- Load DB (non-blocking) ----------
vectordb = load_vectordb()
st.sidebar.write(f"**Chunks in index:** {collection_count(vectordb)}")
st.sidebar.write("Audio: " + ("enabled" if audio_enabled else "disabled"))

# ---------- RAG (only when ready) ----------
SYSTEM_PROMPT = """You are St. Augustine scholar-bot. Answer strictly from the retrieved context.
If unsure, say you don't know. Always cite sources with file path and chunk number.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT + "\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
)

def build_qa(vs: FAISS) -> RetrievalQA:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )

def render_sources(src_docs: List):
    if not src_docs:
        st.info("No sources returned.")
        return
    st.subheader("Sources")
    for i, d in enumerate(src_docs, 1):
        meta = d.metadata or {}
        source = meta.get("source", "unknown")
        chunk_id = meta.get("chunk")
        label = f"{source}" + (f" — chunk {chunk_id}" if chunk_id is not None else "")
        with st.expander(f"{i}. {label}", expanded=False):
            body = d.page_content or ""
            st.write(body[:1200] + ("..." if len(body) > 1200 else ""))

# ---------- Main ----------
st.title("Ask St. Augustine Anything")
st.caption("Answers are drawn exclusively from his writings")

if vectordb is None:
    st.warning("Index not found. Use the **Rebuild index** button in the sidebar. Live logs will appear during rebuild.")
else:
    qa = build_qa(vectordb)
    q = st.text_input("Ask:", value="Teach me about grace", help="The bot cites exact files/chunks")
    if st.button("Ask"):
        if collection_count(vectordb) == 0:
            st.error("Index is empty. Click **Rebuild index** in the sidebar first.")
        else:
            with st.spinner("Thinking…"):
                out = qa({"query": q})
            st.markdown("### Answer")
            st.write(out["result"])
            render_sources(out.get("source_documents", []))
