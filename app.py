# app.py (FAISS)
import os
from pathlib import Path
from typing import List

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- Constants (must match ingest.py) ----------
DATA_DIR = Path("data/clean_final")
FAISS_DIR = "faiss_index/augustine"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---------- Secrets / keys ----------
def get_secret(key: str, default: str | None = None) -> str | None:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ELEVEN_API_KEY = get_secret("ELEVENLABS_API_KEY")  # optional
audio_enabled = bool(ELEVEN_API_KEY)               # avoid NameError

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing. Add it in Streamlit Secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ---------- Vector store ----------
@st.cache_resource(show_spinner=False)
def load_vectordb():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    # allow_dangerous_deserialization is required on Streamlit Cloud
    return FAISS.load_local(
        FAISS_DIR, embeddings, allow_dangerous_deserialization=True
    )

def collection_count(vs: FAISS) -> int:
    try:
        return vs.index.ntotal
    except Exception:
        return 0

# ---------- UI: sidebar ----------
st.set_page_config(page_title="Ask St. Augustine Anything", layout="wide")

st.sidebar.header("Dataset")
st.sidebar.write(f"**Index path:** `{FAISS_DIR}`")

txts = sorted([str(p) for p in DATA_DIR.rglob("*.txt")])
st.sidebar.write(f"**Found {len(txts)} .txt files under** `data/clean_final`.")
with st.sidebar.expander("Examples", expanded=False):
    st.code(txts[:10] if txts else "[]", language="json")

if st.sidebar.button("Rebuild index from data/clean_final"):
    with st.spinner("Rebuilding vector store…"):
        try:
            from ingest import rebuild_vectorstore
            chunks = rebuild_vectorstore()
            st.success(f"Done. Chunks in index: {chunks}")
        except Exception as e:
            st.error(f"Ingest failed: {e}")
    load_vectordb.clear()

# ---------- Load DB ----------
vectordb = load_vectordb()
st.sidebar.write(f"**Chunks in index:** {collection_count(vectordb)}")
st.sidebar.write(f"**FAISS index loaded from** `{FAISS_DIR}`.")
st.sidebar.write("Audio: " + ("enabled" if audio_enabled else "disabled"))

# ---------- RAG chain ----------
SYSTEM_PROMPT = """You are St. Augustine scholar-bot. Answer strictly from the retrieved context.
If unsure, say you don't know. Always cite sources with file path and chunk number.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT + "\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
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

# ---------- Helpers ----------
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

# ---------- Page ----------
st.title("Ask St. Augustine Anything")
st.caption("Answers are drawn exclusively from his writings")

q = st.text_input("Ask:", value="Teach me about grace", help="The bot cites exact files/chunks")
if st.button("Ask"):
    if collection_count(vectordb) == 0:
        st.error("Index is empty (0 chunks). Click **Rebuild index** in the sidebar.")
    else:
        with st.spinner("Thinking…"):
            out = qa({"query": q})
        st.markdown("### Answer")
        st.write(out["result"])
        render_sources(out.get("source_documents", []))
