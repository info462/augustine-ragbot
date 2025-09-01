# app.py
import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- Constants (MUST MATCH ingest.py) ----------
DATA_DIR = Path("data/clean_final")
PERSIST_DIR = "chroma_db/augustine"
COLLECTION_NAME = "augustine"
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
audio_enabled = bool(ELEVEN_API_KEY)  # <-- define this once so it exists

if not OPENAI_API_KEY:
    st.stop()  # streamlit will render an informative error

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ---------- Vector DB ----------
@st.cache_resource(show_spinner=False)
def load_vectordb():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

def collection_count(vs: Chroma) -> int:
    try:
        return vs._collection.count()  # pyright: ignore
    except Exception:
        return 0

# ---------- UI ----------
st.set_page_config(page_title="Ask St. Augustine Anything", layout="wide")

st.sidebar.header("Dataset")
st.sidebar.write(f"**Index path:** `{PERSIST_DIR}`")

# Show file list found under data/clean_final
txts = sorted([str(p) for p in DATA_DIR.rglob("*.txt")])
st.sidebar.write(f"**Found {len(txts)} .txt files under** `data/clean_final`.")
with st.sidebar.expander("Examples:", expanded=False):
    st.code(txts[:10] if txts else "[]", language="json")

# Rebuild button
with st.sidebar:
    if st.button("Rebuild index from data/clean_final", type="primary"):
        with st.spinner("Rebuilding vector store…"):
            from ingest import rebuild_vectorstore
            try:
                chunk_count = rebuild_vectorstore()
                st.success(f"Done. Chunks in index: {chunk_count}")
            except Exception as e:
                st.error(f"Ingest failed: {e}")
        # clear cache so the next call reloads the DB
        load_vectordb.clear()

# Load DB
vectordb = load_vectordb()
st.sidebar.write(f"**Chunks in index:** {collection_count(vectordb)}")
st.sidebar.write(f"**Chroma index loaded from** `{PERSIST_DIR}`.")

# ---------- RAG Chain ----------
SYSTEM_PROMPT = """You are St. Augustine scholar-bot. Answer strictly from the retrieved context.
If unsure, say you don't know. Always cite sources with their file path and chunk number.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        SYSTEM_PROMPT
        + "\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
)

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
)

# ---------- Page ----------
st.title("Ask St. Augustine Anything")
st.caption("Answers are drawn exclusively from his writings")

q = st.text_input("Ask:", value="Teach me about grace", help="The bot cites exact files/chunks")
ask = st.button("Ask")

def render_sources(src_docs: List):
    if not src_docs:
        st.info("No sources returned.")
        return
    st.subheader("Sources")
    for i, d in enumerate(src_docs, 1):
        meta = d.metadata or {}
        source = meta.get("source", "unknown")
        # attach a 'chunk' index if present
        chunk_id = meta.get("chunk", "")
        label = f"{source}" + (f" — chunk {chunk_id}" if chunk_id != "" else "")
        with st.expander(f"{i}. {label}", expanded=False):
            st.write(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))

if ask:
    if collection_count(vectordb) == 0:
        st.error("Index is empty (0 chunks). Click **Rebuild index** in the sidebar.")
    else:
        with st.spinner("Thinking…"):
            out = qa({"query": q})
        st.markdown("### Answer")
        st.write(out["result"])
        render_sources(out.get("source_documents", []))

# ---------- Optional: Audio (safe-guarded) ----------
# Only reference audio_enabled after it is defined above
if audio_enabled:
    st.sidebar.success("ElevenLabs audio enabled.")
else:
    st.sidebar.info("ElevenLabs audio disabled (no API key).")
