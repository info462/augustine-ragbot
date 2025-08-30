# app.py
# --- DO NOT MOVE: SQLite JSON1/FTS5 shim for Chroma on Streamlit Cloud ---
# (Some hosts ship Python's sqlite3 without JSON1; this swaps in pysqlite3.)
try:
    import sys
    import pysqlite3  # SQLite build that includes JSON1/FTS5
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass
# -------------------------------------------------------------------------

import os
import base64
import logging
import warnings
import re
from pathlib import Path
from typing import Tuple

import streamlit as st
from openai import OpenAI
import requests

# Optional: verify JSON1 at runtime (shows up in Streamlit logs)
try:
    import sqlite3
    opts = [r[0] for r in sqlite3.connect(":memory:").execute("PRAGMA compile_options;").fetchall()]
    logging.info("SQLite compile options: %s", opts)
except Exception:
    logging.exception("Could not inspect SQLite compile options")

# ---------- Quiet noisy libs / warnings ----------
logging.getLogger("pypdf").setLevel(logging.ERROR)
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

# ---------- Secrets loader ----------
def get_secret(key: str, default: str | None = None) -> str | None:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    val = os.getenv(key)
    if val:
        return val
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(key, default)
    except Exception:
        return default

# ---------- API keys / voice selection ----------
OPENAI_API_KEY  = (get_secret("OPENAI_API_KEY") or "").strip()
ELEVEN_API_KEY  = (get_secret("ELEVEN_API_KEY") or "").strip()
ELEVEN_VOICE_ID = (get_secret("ELEVEN_VOICE_ID") or "").strip()

try:
    ELEVEN_VERBOSE_ERRORS = bool(st.secrets.get("ELEVEN_VERBOSE_ERRORS", False))
except Exception:
    ELEVEN_VERBOSE_ERRORS = False

if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY. Add to `.streamlit/secrets.toml` or Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ---------- OpenAI client ----------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- ElevenLabs preflight ----------
def eleven_preflight() -> Tuple[bool, str]:
    if not ELEVEN_API_KEY:
        return False, "Add ELEVEN_API_KEY in secrets to enable audio."
    if not ELEVEN_VOICE_ID:
        return False, "Add ELEVEN_VOICE_ID in secrets to choose a voice."
    try:
        r = requests.get("https://api.elevenlabs.io/v1/user",
                         headers={"xi-api-key": ELEVEN_API_KEY, "Accept": "application/json"},
                         timeout=20)
        if r.status_code != 200:
            return False, f"API key check failed ({r.status_code}): {r.text[:200]}"
    except Exception as e:
        return False, f"Could not reach ElevenLabs user endpoint: {e}"
    try:
        r = requests.get(f"https://api.elevenlabs.io/v1/voices/{ELEVEN_VOICE_ID}",
                         headers={"xi-api-key": ELEVEN_API_KEY, "Accept": "application/json"},
                         timeout=20)
        if r.status_code == 200:
            return True, "Audio ready."
        elif r.status_code in (401, 403):
            return False, f"Key unauthorized for this voice ({r.status_code}). Verify workspace or share the voice."
        elif r.status_code == 404:
            return False, "Voice not found (404). Check ELEVEN_VOICE_ID or share the voice to this account."
        else:
            return False, f"Voice check returned {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"Could not reach ElevenLabs voice endpoint: {e}"

# ---------- TTS helpers ----------
RACHEL_ID = "21m00Tcm4TlvDq8ikWAM"

def _http_tts(voice_id: str, text: str, out_path: str) -> Tuple[bool, str]:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_API_KEY}
    payload = {"text": text, "model_id": "eleven_multilingual_v2",
               "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
            if r.status_code >= 400:
                return False, f"{r.status_code} {r.reason}: {r.text[:200]}"
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True, "ok"
    except Exception as e:
        return False, f"request error: {e}"

def synthesize_tts(text: str, out_path: str) -> str:
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID:
        raise RuntimeError("ElevenLabs audio requires ELEVEN_API_KEY and ELEVEN_VOICE_ID in secrets.")

    MAX_CHARS = 1800
    speak_text = text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "…"

    # Try SDK v2 first
    try:
        from elevenlabs.client import ElevenLabs
        el_client = ElevenLabs(api_key=ELEVEN_API_KEY)
        try:
            audio = el_client.text_to_speech.convert(
                voice_id=ELEVEN_VOICE_ID, model_id="eleven_multilingual_v2", text=speak_text
            )
            with open(out_path, "wb") as f:
                if hasattr(audio, "__iter__") and not isinstance(audio, (bytes, bytearray)):
                    for chunk in audio:
                        if chunk:
                            f.write(chunk)
                else:
                    f.write(audio)
            return out_path
        except Exception as e:
            if ELEVEN_VERBOSE_ERRORS: st.caption(f"SDK v2 convert failed: {e}")
    except Exception as e:
        if ELEVEN_VERBOSE_ERRORS: st.caption(f"SDK v2 import/init failed: {e}")

    # Fallback to raw HTTP; then fallback voice
    ok, reason = _http_tts(ELEVEN_VOICE_ID, speak_text, out_path)
    if ok: return out_path
    ok_fb, reason_fb = _http_tts(RACHEL_ID, speak_text, out_path)
    if ok_fb:
        if ELEVEN_VERBOSE_ERRORS: st.caption(f"Fell back to Rachel because primary failed: {reason}")
        return out_path
    raise RuntimeError(f"ElevenLabs TTS failed. Primary: {reason}. Fallback: {reason_fb}")

# ---------- LangChain / Chroma ----------
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Augustine prompt config ---
AUGUSTINE_SYSTEM_PROMPT = (
    "You are Augustine of Hippo (354–430). Speak in the FIRST PERSON, "
    "as a pastor counseling someone in your congregation—warm, candid, concise. "
    "Derive tone and diction from your own writings (Confessions, City of God, On Christian Doctrine, Enchiridion, Letters, Sermons). "
    "Answer in AT MOST TWO PARAGRAPHS. Prefer two, but never more. "
    "Ground your counsel in the retrieved passages. "
    "SCRIPTURE HANDLING: When you bring in the Bible, NEVER imply you authored it. "
    "Explicitly attribute it, e.g., 'as Scripture says,' 'as the Apostle writes,' or 'as the Psalmist says.' "
    "If quoting, keep it brief and put the reference in parentheses, e.g., (Rom 5:5) or (Ps 139). "
    "Do not place Augustine’s words in quotation marks as if they were Scripture, and do not say 'I wrote' about Scripture.\n\n"
    "PARENTHESES: You may include brief citations in parentheses; the app will not speak parentheses aloud."
)

# --- Persisted Chroma config (try both old/new paths) ---
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR_CANDIDATES = [Path("chroma_db/augustine"), Path(".chroma_db/augustine")]
COLLECTION = "augustine"
RETRIEVAL_K = 5
SCORE_THRESHOLD = 0.25  # unused in Option A

def _resolve_db_dir() -> Path:
    for p in DB_DIR_CANDIDATES:
        if p.exists() and any(p.iterdir()):
            return p
    return DB_DIR_CANDIDATES[0]  # default

DB_DIR = _resolve_db_dir()

@st.cache_resource(show_spinner=True)
def load_vectordb():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key=get_openai_api_key())
    return Chroma(
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION,
        embedding_function=emb,
    )

def index_count(vdb) -> int | None:
    try:
        return vdb._collection.count()
    except Exception:
        return None


# ---------- Robust source formatting ----------
def _best_meta_title(meta: dict, fallback: str = "unknown") -> str:
    return (
        meta.get("work_title")
        or meta.get("title")
        or meta.get("source")
        or meta.get("source_path")
        or meta.get("file_path")
        or meta.get("path")
        or meta.get("filename")
        or fallback
    )

def _best_meta_page(meta: dict):
    return meta.get("page") or meta.get("page_number") or meta.get("page_no") or "chunk"

def build_context(hits) -> str:
    """Accepts list[(Document, score)] or list[Document]; formats for prompting."""
    blocks = []
    for h in hits:
        if isinstance(h, tuple):
            doc, score = h
        else:
            doc, score = h, None
        meta = doc.metadata or {}
        title = _best_meta_title(meta, fallback=Path(meta.get("source_path", "unknown")).stem)
        page = _best_meta_page(meta)
        header = f"[SOURCE] ({title} — {page})"
        if score is not None:
            try:
                header += f"  [score: {score:.3f}]"
            except Exception:
                pass
        text = (doc.page_content or "").strip()
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)

def strip_parentheses(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", text)

def limit_to_two_paragraphs(text: str) -> str:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return "\n\n".join(paras[:2])

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Talk to Augustine (Audio-First)", page_icon="📜")
st.markdown(
    """
    <style>
      .block-container { max-width: 900px !important; }
      .audio-wrap { margin: 0.4rem 0 1rem 0; }
      .caption-box { font-size: 0.95rem; line-height: 1.6; color: #444; opacity: 0.9; }
      .stExpander { border: 1px solid #e6e6e6; border-radius: 10px; }
      .stExpander > div[role='button'] { font-weight: 600; }
      .stChatMessage { line-height: 1.55; }
      code, pre { font-size: 0.95em; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Ask St. Augustine Anything")
st.caption("Answers are drawn exclusively from his writings")
st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# NEW: persist last hits for showing Sources even across reruns
if "last_hits" not in st.session_state:
    st.session_state.last_hits = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.spinner("Loading knowledge base…"):
    vectordb = load_vectordb()

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Dataset")
    st.write("Chroma index loaded from ./.chroma_db/augustine.")
    st.caption("Source files (cleaned) live under `data/clean_final`. Use the button below to rebuild the index.")

    # Optional: rebuild button
    try:
       from ingest import rebuild_vectorstore

if st.button("🔁 Rebuild index from data/clean_final"):
    import os
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # <-- bridge secrets → env
    with st.spinner("Rebuilding vector store…"):
        rebuild_vectorstore()
    st.success("Done. Reload the page to use the new index.")


    except Exception:
        st.caption("`ingest.py` not found, rebuild button disabled.")

    st.subheader("Audio")
    ok, reason = eleven_preflight()
    if ok:
        audio_enabled    = st.toggle("🔊 Speak answers (default ON)", value=True)
        autoplay_enabled = st.toggle("▶️ Auto-play audio (default ON)", value=True)
    else:
        audio_enabled = False
        autoplay_enabled = False
        st.info(reason)

# ---------- Helper: inline autoplay audio ----------
def render_autoplay_audio(file_path: str, autoplay: bool = True):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    auto_attr = "autoplay" if autoplay else ""
    st.markdown(
        f"""
        <div class="audio-wrap">
          <audio {auto_attr} controls src="data:audio/mpeg;base64,{b64}"></audio>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Chat (Option A retrieval wired in) ----------
user_q = st.chat_input("Ask your question…")
if user_q:
    if not isinstance(user_q, str) or not user_q.strip():
        st.warning("Empty question; please type something.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    ans = ""
    transcript_text = ""
    spoken_text = ""

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                # OPTION A: simple top-k search (returns list of (Document, score))
                hits = vectordb.similarity_search_with_relevance_scores(
                    user_q, k=RETRIEVAL_K
                )
                st.session_state.last_hits = hits  # <-- persist for the Sources section
                context = build_context(hits)

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "system", "content": AUGUSTINE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{user_q}\n\n"
                            "Context (top passages from Augustine's works):\n"
                            f"{context if context.strip() else '(no strong matches)'}\n\n"
                            "Reminder: Attribute Bible verses as Scripture (e.g., 'as Scripture says …'), "
                            "and never as something you authored. Keep the answer to at most two short paragraphs."
                        ),
                    },
                ]

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                )
                ans = (resp.choices[0].message.content or "").strip()

                # --- POST-PROCESSING FOR AUDIO-FIRST UI ---
                transcript_text = limit_to_two_paragraphs(ans)
                spoken_text = strip_parentheses(transcript_text)

            except Exception as e:
                ans = f"Sorry, I hit an error: `{e}`"
                st.session_state.last_hits = []  # make it explicit on failure

            # 1) AUDIO FIRST
            if audio_enabled and ELEVEN_API_KEY and ans.strip():
                try:
                    out_dir = Path("audio"); out_dir.mkdir(exist_ok=True)
                    audio_path = out_dir / f"reply_{len(st.session_state.messages)}.mp3"
                    synthesize_tts(spoken_text or ans, str(audio_path))
                    render_autoplay_audio(str(audio_path), autoplay=bool(autoplay_enabled))
                except Exception as e:
                    st.warning(f"Audio generation failed: {e}")

            # 2) CAPTIONS
            with st.expander("📝 Transcript (click to show/hide)", expanded=False):
                st.markdown(
                    f"<div class='caption-box'>{(transcript_text or ans)}</div>",
                    unsafe_allow_html=True
                )

# ---------- ALWAYS render Sources + Debug (persists across reruns) ----------
with st.expander("Sources (click to expand)", expanded=False):
    hits_to_show = st.session_state.get("last_hits", [])
    if not hits_to_show:
        st.caption("No sources retrieved. (Index empty or no strong matches.)")
        st.caption("Tip: click “🔁 Rebuild index” in the sidebar if you haven’t ingested yet.")
    else:
        for i, h in enumerate(hits_to_show, 1):
            if isinstance(h, tuple):
                d, sc = h
            else:
                d, sc = h, None
            meta = d.metadata or {}
            work = _best_meta_title(meta, fallback=Path(meta.get("source_path","unknown")).stem)
            page = _best_meta_page(meta)
            line = f"**{i}. {work}** — {page}"
            if sc is not None:
                try:
                    line += f"  _(score: {sc:.3f})_"
                except Exception:
                    pass
            st.markdown(line)

            excerpt = (d.page_content or "").strip().replace("\n", " ")
            if excerpt:
                st.caption(excerpt[:350] + ("…" if len(excerpt) > 350 else ""))

with st.expander("🔧 Debug: raw metadata"):
    try:
        meta_list = [ (h[0].metadata if isinstance(h, tuple) else h.metadata)
                      for h in st.session_state.get("last_hits", []) ]
        st.json(meta_list)
    except Exception:
        st.caption("Could not display raw metadata.")

# Finally, if we generated an answer this run, store it in history
if user_q:
    st.session_state.messages.append({"role": "assistant", "content": (transcript_text or ans)})
