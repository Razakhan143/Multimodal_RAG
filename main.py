

"""
Bayz — Multimodal RAG Chatbot
Streamlit UI  |  PDF · DOCX · Audio · Video · Image

Fixes applied
─────────────
1. Single-click send  → query stored in session_state, cleared after submit
2. Input auto-clears  → key rotated via a counter after each send
3. Loading animation  → st.status block shown while RAG runs, chat renders after
4. 25 MB file cap     → enforced before any processing
5. FFprobe / WinError → soundfile + scipy used for WAV conversion (no FFmpeg needed)
6. Extended audio     → wav, mp3, m4a, ogg, flac, aac, wma, opus all accepted
"""

import os
import sys
import time
import asyncio
import tempfile
import streamlit as st
from pathlib import Path

# On Windows, Python 3.8+ defaults to ProactorEventLoop which logs a spurious
# ConnectionResetError when the remote side (e.g. Groq API) closes a TCP
# connection normally. Switching to SelectorEventLoop silences it.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── RAG backend ───────────────────────────────────────────────────────────────
# The backend pulls in torch + CLIP, which takes ~40s to import. Importing it at
# module top would block Streamlit's very first render and show a blank screen.
# Instead we import it lazily on the first query via _get_caller().
caller = None
CALLER_AVAILABLE = True   # assume available; verified lazily on first use


def _get_caller():
    """Import and cache the heavy RAG backend on first use."""
    global caller, CALLER_AVAILABLE
    if caller is None:
        try:
            from app import main as _caller
            caller = _caller
            CALLER_AVAILABLE = True
        except Exception:
            CALLER_AVAILABLE = False
    return caller

MAX_FILE_MB  = 25
MAX_FILE_B   = MAX_FILE_MB * 1024 * 1024

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bayz · Multimodal RAG",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-base:      #0D0F1A;
    --bg-panel:     #13162A;
    --bg-card:      #1B1F38;
    --bg-input:     #1E2340;
    --accent:       #6C63FF;
    --accent-soft:  rgba(108,99,255,.15);
    --accent-glow:  rgba(108,99,255,.35);
    --success:      #34D399;
    --warning:      #FBBF24;
    --danger:       #F87171;
    --text-primary: #E8EAF6;
    --text-muted:   #8890B0;
    --border:       rgba(108,99,255,.22);
    --radius-lg:    16px;
    --radius-md:    10px;
    --radius-sm:    6px;
    --font-display: 'Syne', sans-serif;
    --font-body:    'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}
.stApp {
    background:
        radial-gradient(900px 500px at 12% -8%, rgba(108,99,255,.14), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(167,139,250,.10), transparent 55%),
        var(--bg-base) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
* { scrollbar-width: thin; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

.bayz-wordmark {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -.02em;
    background: linear-gradient(135deg, #6C63FF 0%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding: 32px 24px 4px;
    display: block;
}
.bayz-tagline {
    font-size: .75rem;
    color: var(--text-muted);
    padding: 0 24px 24px;
    letter-spacing: .06em;
    text-transform: uppercase;
    display: block;
}
.sb-divider {
    height: 1px;
    background: var(--border);
    margin: 0 24px 20px;
}
.sb-label {
    font-size: .65rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0 24px 8px;
    display: block;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-input) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}

/* ── Badges ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: .75rem;
    padding: 4px 10px;
    border-radius: 999px;
    font-weight: 500;
}
.badge-ready   { background:rgba(52,211,153,.12); color:var(--success); border:1px solid rgba(52,211,153,.3); }
.badge-pending { background:rgba(251,191,36,.10);  color:var(--warning); border:1px solid rgba(251,191,36,.3); }
.badge-error   { background:rgba(248,113,113,.12); color:var(--danger);  border:1px solid rgba(248,113,113,.3); }
.badge-info    { background:var(--accent-soft);    color:#A78BFA;        border:1px solid var(--border); }

/* ── Header bar ── */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 36px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, var(--bg-panel), rgba(19,22,42,.6));
    backdrop-filter: blur(8px);
    position: sticky;
    top: 0;
    z-index: 10;
}
.main-header-title {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -.01em;
}
.main-header-sub { font-size:.8rem; color:var(--text-muted); margin-top:2px; }

/* ── Chat ── */
.chat-area {
    padding: 28px 36px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.chat-bubble-user {
    align-self: flex-end;
    max-width: 70%;
    background: linear-gradient(135deg,#6C63FF 0%,#8B83FF 100%);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 13px 18px;
    font-size: .9rem;
    line-height: 1.55;
    box-shadow: 0 6px 20px var(--accent-glow);
    word-break: break-word;
    animation: bubbleIn .32s cubic-bezier(.21,1.02,.73,1) both;
}
.chat-bubble-ai {
    align-self: flex-start;
    max-width: 75%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px 18px 18px 18px;
    padding: 15px 18px;
    font-size: .9rem;
    line-height: 1.6;
    word-break: break-word;
    box-shadow: 0 4px 18px rgba(0,0,0,.22);
    animation: bubbleIn .32s cubic-bezier(.21,1.02,.73,1) both;
}
@keyframes bubbleIn {
    from { opacity: 0; transform: translateY(8px) scale(.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.ai-label {
    font-size:.68rem;
    letter-spacing:.08em;
    text-transform:uppercase;
    color:var(--accent);
    margin-bottom:6px;
    display:block;
    font-weight:600;
}
.chat-meta {
    font-size:.68rem;
    color:var(--text-muted);
    margin-top:5px;
    text-align:right;
}

/* ── Empty state ── */
.empty-state {
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    padding:80px 40px;
    text-align:center;
}
.empty-icon  { font-size:3.5rem; margin-bottom:16px; animation:floaty 3.5s ease-in-out infinite; }
@keyframes floaty { 0%,100%{ transform:translateY(0); } 50%{ transform:translateY(-8px); } }
.empty-title { font-family:var(--font-display); font-size:1.5rem; font-weight:700; margin-bottom:8px; }
.empty-sub   { font-size:.9rem; color:var(--text-muted); max-width:360px; line-height:1.6; }
.chips       { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:24px; }
.chip {
    padding:7px 14px;
    border-radius:999px;
    background:var(--bg-card);
    border:1px solid var(--border);
    font-size:.8rem;
    color:var(--text-muted);
    transition:all .18s ease;
}
.chip:hover { border-color:var(--accent); color:var(--text-primary); transform:translateY(-1px); }

/* ── Input bar ── */
.input-bar-wrap {
    border-top:1px solid var(--border);
    background:var(--bg-panel);
    padding:18px 36px 20px;
    position:sticky;
    bottom:0;
}

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: .9rem !important;
    padding: 12px 16px !important;
    caret-color: var(--accent) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
.stButton > button {
    background: linear-gradient(135deg,#6C63FF,#8B83FF) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: .875rem !important;
    padding: 10px 22px !important;
    transition: all .18s ease !important;
    box-shadow: 0 4px 14px var(--accent-glow) !important;
    width: 100% !important;
}
.stButton > button:hover  { transform:translateY(-1px) !important; }
.stButton > button:active { transform:translateY(0) !important; }

/* ── File info strip ── */
.file-strip {
    display:flex; align-items:center; gap:10px;
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius-md); padding:10px 14px;
    margin-bottom:10px; font-size:.82rem;
}
.file-strip .ficon { font-size:1.2rem; }
.file-strip .fname { font-weight:500; color:var(--text-primary); }
.file-strip .fmeta { color:var(--text-muted); margin-left:auto; font-size:.75rem; }

.conv-notice {
    display:flex; align-items:center; gap:8px;
    background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.25);
    border-radius:var(--radius-sm); padding:8px 12px;
    font-size:.8rem; color:var(--warning); margin-bottom:12px;
}
.size-error {
    display:flex; align-items:center; gap:8px;
    background:rgba(248,113,113,.08); border:1px solid rgba(248,113,113,.25);
    border-radius:var(--radius-sm); padding:10px 14px;
    font-size:.82rem; color:var(--danger); margin-bottom:12px;
}

/* ── Thinking indicator ── */
.thinking-row {
    display:flex; align-items:center; gap:12px;
    padding:14px 18px;
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:4px 18px 18px 18px;
    max-width:75%; font-size:.9rem; color:var(--text-muted);
}
@keyframes pulse {
    0%,100% { opacity:.3; transform:scale(.85); }
    50%      { opacity:1;  transform:scale(1); }
}
.dot { width:7px; height:7px; border-radius:50%; background:var(--accent); display:inline-block; }
.dot:nth-child(1){ animation:pulse 1.2s ease-in-out infinite 0s; }
.dot:nth-child(2){ animation:pulse 1.2s ease-in-out infinite .2s; }
.dot:nth-child(3){ animation:pulse 1.2s ease-in-out infinite .4s; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg-base); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:999px; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
MODE_META = {
    "document": {
        "icon":   "📄",
        "label":  "Document",
        "exts":   ["pdf", "docx", "txt"],
        "accept": {
            "application/pdf": [".pdf"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
            "text/plain": [".txt"],
        },
    },
    "audio": {
        "icon":   "🎙️",
        "label":  "Audio",
        # All formats accepted — WAV passthrough, others converted via soundfile/pydub
        "exts":   ["wav", "mp3", "m4a", "ogg", "flac", "aac", "wma", "opus", "mp4", "mov", "avi", "mkv", "webm"],
        "accept": {
            "audio/*": [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus"],
            "video/*": [".mp4", ".mov", ".avi", ".mkv", ".webm"],
        },
    },
    "image": {
        "icon":   "🖼️",
        "label":  "Image",
        "exts":   ["jpg", "jpeg", "png", "webp", "gif"],
        "accept": {"image/*": [".jpg", ".jpeg", ".png", ".webp", ".gif"]},
    },
    "video": {
        "icon":   "🎬",
        "label":  "Video",
        "exts":   ["mp4", "mov", "avi", "mkv", "webm"],
        "accept": {"video/*": [".mp4", ".mov", ".avi", ".mkv", ".webm"]},
    },
}

SAMPLE_QUESTIONS = {
    "document": ["Summarise the key points", "What are the main conclusions?", "List important dates mentioned"],
    "audio":    ["What is the speaker discussing?", "Summarise the conversation", "What decisions were made?"],
    "image":    ["What is happening in this image?", "Describe the visual elements", "Any text visible?"],
    "video":    ["What events occur in this video?", "Describe the timeline", "What is being demonstrated?"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "messages":       [],
    "rag_type":       None,
    "file_path":      None,
    "file_name":      None,
    "file_ready":     False,
    "converted_wav":  False,
    "error":          None,
    "processing":     False,   # True while RAG is running
    "input_key":      0,       # rotate to clear the text input widget
    "pending_query":  None,    # query waiting to be executed
    "rag_ctx":        None,    # cached ingested RagContext (heavy work done once)
    "ingested_path":  None,    # file_path the cached context was built from
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _reset_file():
    st.session_state.messages      = []
    st.session_state.file_path     = None
    st.session_state.file_name     = None
    st.session_state.file_ready    = False
    st.session_state.converted_wav = False
    st.session_state.error         = None
    st.session_state.processing    = False
    st.session_state.pending_query = None
    st.session_state.rag_ctx       = None
    st.session_state.ingested_path = None


def _save_upload(uploaded_file) -> str:
    """Write UploadedFile bytes to a named temp file, return path."""
    suffix = Path(uploaded_file.name).suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    # Use .read() to get raw bytes; avoids BufferError from memoryview resizing.
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def _ensure_ingested(rag_type: str, file_path: str):
    """Ingest the file once and cache the resulting context in session state.

    The heavy work (frame extraction, transcription, embedding, vector-store
    population) happens only the first time a given file is queried. Every later
    question on the same file reuses the cached context and only retrieves.

    Returns the cached context, or None in demo mode / on failure.
    """
    caller = _get_caller()
    if not CALLER_AVAILABLE or caller is None:
        return None

    # Reuse the cached context if it was built from this exact file.
    if (st.session_state.rag_ctx is not None
            and st.session_state.ingested_path == file_path):
        return st.session_state.rag_ctx

    # New (or changed) file → run ingestion once and cache it.
    ctx = caller.ingest(rag_type, file_path)
    st.session_state.rag_ctx       = ctx
    st.session_state.ingested_path = file_path
    return ctx


def _answer_query(rag_type: str, file_path: str, query: str) -> tuple:
    """Answer a single question, ingesting the file first if needed.

    Returns (answer: str, top_context: dict | None).
    """
    caller = _get_caller()
    if not CALLER_AVAILABLE or caller is None:
        time.sleep(1.0)
        return (
            f"**[Demo mode]** `app.main` could not be imported — connect your backend to get real answers.\n\n"
            f"**Mode:** {rag_type}  |  **File:** `{Path(file_path).name}`\n\n"
            f"**Query:** {query}",
            None,
        )

    ctx = _ensure_ingested(rag_type, file_path)
    return caller.query(ctx, query)


def _needs_ingestion() -> bool:
    """True when the current file hasn't been ingested yet (first question)."""
    return (
        st.session_state.file_ready
        and st.session_state.ingested_path != st.session_state.file_path
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<span class="bayz-wordmark">✦ Bayz</span>', unsafe_allow_html=True)
    st.markdown('<span class="bayz-tagline">Multimodal Intelligence</span>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Mode buttons ──────────────────────────────────────────────────────
    st.markdown('<span class="sb-label">Select Mode</span>', unsafe_allow_html=True)
    for mk, mm in MODE_META.items():
        if st.button(f"{mm['icon']}  {mm['label']}", key=f"mode_{mk}", use_container_width=True):
            if st.session_state.rag_type != mk:
                st.session_state.rag_type = mk
                _reset_file()
                st.rerun()

    st.markdown('<div class="sb-divider" style="margin-top:16px;"></div>', unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────
    mode = st.session_state.rag_type
    if mode:
        meta = MODE_META[mode]
        st.markdown(f'<span class="sb-label">Upload {meta["label"]}</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="padding:0 24px 6px;font-size:.75rem;color:var(--text-muted);">'
            f'Max {MAX_FILE_MB} MB · {", ".join("." + e for e in meta["exts"])}'
            f'</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            label="Upload file",
            type=meta["exts"],
            key=f"upload_{mode}",
            label_visibility="collapsed",
        )

        if uploaded and uploaded.name != st.session_state.file_name:
            _reset_file()
            # Use .read() to obtain a bytes object instead of a memoryview.
            # This prevents BufferError: Existing exports of data: object cannot be re-sized
            # which can occur when Streamlit tries to resize the underlying buffer.
            # Read the uploaded file to determine its size, then reset the pointer
            # so that the subsequent call to `_save_upload` can read the content again.
            file_bytes = uploaded.read()
            file_size  = len(file_bytes)
            # Reset stream position for later use
            uploaded.seek(0)

            # ── 25 MB guard ───────────────────────────────────────────────
            if file_size > MAX_FILE_B:
                st.markdown(
                    f'<div class="size-error">⚠ File is {_human_size(file_size)} — '
                    f'limit is {MAX_FILE_MB} MB. Please upload a smaller file.</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.error = f"File exceeds {MAX_FILE_MB} MB limit."
            else:
                with st.spinner("Preparing file…"):
                    try:
                        raw_path = _save_upload(uploaded)
                        # No conversion: Groq Whisper accepts mp3/m4a/etc natively,
                        # and video must stay intact so frames can be extracted.
                        st.session_state.file_path  = raw_path
                        st.session_state.file_name  = uploaded.name
                        st.session_state.file_ready = True
                        st.session_state.converted_wav = False
                        st.session_state.error      = None

                    except Exception as exc:
                        st.session_state.error      = str(exc)
                        st.session_state.file_ready = False

            st.rerun()

        # ── Status display ────────────────────────────────────────────────
        if st.session_state.file_ready and st.session_state.file_name:
            size_b   = os.path.getsize(st.session_state.file_path)
            ext_disp = Path(st.session_state.file_name).suffix.upper().lstrip(".")

            if st.session_state.converted_wav:
                st.markdown(
                    '<div class="conv-notice">⚡ Converted to WAV automatically</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="file-strip">'
                f'<span class="ficon">{meta["icon"]}</span>'
                f'<span class="fname">{st.session_state.file_name}</span>'
                f'<span class="fmeta">{ext_disp} · {_human_size(size_b)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="text-align:center;margin-bottom:8px;">'
                '<span class="status-badge badge-ready">✓ Ready to chat</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        elif st.session_state.error:
            st.error(st.session_state.error, icon="⚠️")

    else:
        st.markdown(
            '<div style="padding:16px 24px;">'
            '<span class="status-badge badge-pending">← Pick a mode to begin</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Clear chat ────────────────────────────────────────────────────────
    st.markdown('<div class="sb-divider" style="margin-top:8px;"></div>', unsafe_allow_html=True)
    if st.session_state.messages:
        if st.button("🗑  Clear conversation", use_container_width=True, key="clear_chat"):
            st.session_state.messages      = []
            st.session_state.pending_query = None
            st.session_state.processing    = False
            st.rerun()

    st.markdown(
        '<div style="position:absolute;bottom:20px;left:0;right:0;text-align:center;">'
        '<span style="font-size:.7rem;color:var(--text-muted);">Bayz · v1.1 · Multimodal RAG</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTE PENDING QUERY  (runs at top of rerun, before any UI renders)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.pending_query and st.session_state.file_ready:
    _q = st.session_state.pending_query
    st.session_state.pending_query = None   # clear before executing
    st.session_state.processing    = True

    try:
        _result = _answer_query(
            rag_type  = st.session_state.rag_type,
            file_path = st.session_state.file_path,
            query     = _q,
        )
        # _answer_query returns (answer, top_context) or a bare string in demo mode
        if isinstance(_result, tuple):
            _response, _top_ctx = _result
        else:
            _response, _top_ctx = _result, None
    except Exception as _exc:
        _response, _top_ctx = f"⚠️ An error occurred: {_exc}", None

    st.session_state.messages.append({
        "role":    "assistant",
        "content": _response,
        "top_ctx": _top_ctx,
        "ts":      time.strftime("%H:%M"),
    })
    st.session_state.processing = False
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
mode = st.session_state.rag_type

# ── Header ────────────────────────────────────────────────────────────────────
if mode:
    meta = MODE_META[mode]
    badge = (
        '<span class="status-badge badge-ready">✓ File loaded</span>'
        if st.session_state.file_ready
        else '<span class="status-badge badge-pending">⬤ Awaiting file</span>'
    )
    st.markdown(
        f'<div class="main-header"><div>'
        f'<div class="main-header-title">{meta["icon"]} {meta["label"]} Analysis</div>'
        f'<div class="main-header-sub">Ask anything about your {meta["label"].lower()}</div>'
        f'</div>{badge}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="main-header"><div>'
        '<div class="main-header-title">✦ Welcome to Bayz</div>'
        '<div class="main-header-sub">Select a mode from the sidebar to get started</div>'
        '</div><span class="status-badge badge-info">Choose a mode →</span></div>',
        unsafe_allow_html=True,
    )

# ── Chat messages ──────────────────────────────────────────────────────────────
chat_placeholder = st.container()

with chat_placeholder:
    msgs = st.session_state.messages

    if not msgs and not st.session_state.processing:
        # Empty-state screens
        if not mode:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-icon">✦</div>'
                '<div class="empty-title">Multimodal Intelligence</div>'
                '<div class="empty-sub">Understands documents, audio, images, and video.<br>'
                'Pick a mode on the left, upload your file, and ask anything.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif not st.session_state.file_ready:
            st.markdown(
                f'<div class="empty-state">'
                f'<div class="empty-icon">{meta["icon"]}</div>'
                f'<div class="empty-title">Upload a {meta["label"]}</div>'
                f'<div class="empty-sub">Accepted: {", ".join("." + e for e in meta["exts"])}<br>'
                f'Max size: {MAX_FILE_MB} MB</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            chips_html = "".join(
                f'<span class="chip">{q}</span>'
                for q in SAMPLE_QUESTIONS.get(mode, [])
            )
            st.markdown(
                f'<div class="empty-state">'
                f'<div class="empty-icon">💬</div>'
                f'<div class="empty-title">Ready to answer</div>'
                f'<div class="empty-sub">Your {meta["label"].lower()} is loaded. Try a prompt below:</div>'
                f'<div class="chips">{chips_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        # Render chat history
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        for msg in msgs:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin-bottom:16px;">'
                    f'<div class="chat-bubble-user">{msg["content"]}'
                    f'<div class="chat-meta">{msg.get("ts","")}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-start;margin-bottom:16px;">'
                    f'<div class="chat-bubble-ai">'
                    f'<span class="ai-label">✦ Bayz</span>'
                    f'{msg["content"]}'
                    f'<div class="chat-meta">{msg.get("ts","")}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                # ── Top-1 context panel ───────────────────────────────────
                top_ctx = msg.get("top_ctx")
                if top_ctx:
                    ctx_type = top_ctx.get("type")
                    # Constrain panel to ~40% of the chat column width
                    _panel_col, _spacer = st.columns([4, 6])
                    with _panel_col:
                        if ctx_type == "video":
                            ts = top_ctx.get("timestamp")
                            fp = top_ctx.get("file_path")
                            if fp and os.path.exists(fp):
                                ts_label = f" @ {ts:.1f}s" if ts is not None else ""
                                with st.expander(f"▶ Retrieved context{ts_label}", expanded=True):
                                    st.markdown(
                                        '<div style="max-height:220px;overflow:hidden;border-radius:8px;">',
                                        unsafe_allow_html=True,
                                    )
                                    start_time = int(ts) if ts is not None else 0
                                    st.video(fp, start_time=start_time)
                                    st.markdown('</div>', unsafe_allow_html=True)
                        elif ctx_type == "audio":
                            fp  = top_ctx.get("file_path")
                            txt = top_ctx.get("text", "")
                            if fp and os.path.exists(fp):
                                with st.expander("🎙 Retrieved context", expanded=True):
                                    st.audio(fp)
                                    if txt:
                                        st.caption(
                                            f'"{txt[:200]}…"' if len(txt) > 200 else f'"{txt}"'
                                        )
                        elif ctx_type == "document":
                            txt = top_ctx.get("text", "")
                            if txt:
                                with st.expander("📄 Retrieved passage", expanded=True):
                                    st.markdown(
                                        f'<div style="background:var(--bg-input);border:1px solid var(--border);'
                                        f'border-radius:var(--radius-md);padding:10px 14px;'
                                        f'font-size:.82rem;line-height:1.6;color:var(--text-muted);'
                                        f'max-height:160px;overflow-y:auto;">'
                                        f'{txt[:400]}{"…" if len(txt) > 400 else ""}'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                        elif ctx_type == "image":
                            b64 = top_ctx.get("b64", "")
                            if b64:
                                with st.expander("🖼 Source image", expanded=True):
                                    st.markdown(
                                        '<div style="max-height:200px;overflow:hidden;border-radius:8px;">',
                                        unsafe_allow_html=True,
                                    )
                                    st.image(f"data:image/jpeg;base64,{b64}", use_container_width=True)
                                    st.markdown('</div>', unsafe_allow_html=True)

        # ── Animated thinking dots while RAG runs ─────────────────────────
        if st.session_state.processing:
            _status_txt = (
                "Processing your file for the first time — this can take a moment…"
                if _needs_ingestion()
                else "Searching…"
            )
            st.markdown(
                f'<div style="display:flex;justify-content:flex-start;margin-bottom:16px;">'
                f'<div class="thinking-row">'
                f'<span class="ai-label" style="margin-bottom:0;">✦ Bayz</span>'
                f'<span style="color:var(--text-muted);font-size:.85rem;margin-right:4px;">{_status_txt}</span>'
                f'<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT BAR  — single-click, auto-clear
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="input-bar-wrap">', unsafe_allow_html=True)

file_ready = bool(mode and st.session_state.file_ready)
is_busy    = st.session_state.processing

placeholder_txt = (
    f"Ask about your {MODE_META[mode]['label'].lower()}…"
    if file_ready else "Upload a file to start chatting…"
)

def _submit_query(text: str):
    """Shared logic for submitting a query (Enter key or button click)."""
    text = text.strip()
    if not text:
        return
    st.session_state.messages.append({
        "role":    "user",
        "content": text,
        "ts":      time.strftime("%H:%M"),
    })
    st.session_state.pending_query = text
    st.session_state.processing    = True
    st.session_state.input_key    += 1   # rotate key → widget re-creates blank


with st.form(key=f"chat_form_{st.session_state.input_key}", clear_on_submit=True):
    form_col, form_btn = st.columns([9, 1])
    with form_col:
        current_input = st.text_input(
            label="Query",
            placeholder=placeholder_txt,
            label_visibility="collapsed",
            disabled=not file_ready or is_busy,
            key=f"query_input_{st.session_state.input_key}",
        )
    with form_btn:
        send_clicked = st.form_submit_button(
            "⟩ Send" if not is_busy else "…",
            disabled=is_busy,
            use_container_width=True,
        )

st.markdown('</div>', unsafe_allow_html=True)

# ── On Enter or button: submit the query ─────────────────────────────────────
can_send = file_ready and not is_busy and bool(current_input and current_input.strip())
if send_clicked and can_send:
    _submit_query(current_input)
    st.rerun()