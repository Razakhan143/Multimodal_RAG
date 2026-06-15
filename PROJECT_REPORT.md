# Bayz — Multimodal RAG Chatbot
## Project Report

**Author:** raza khan  
**Date:** June 2026  
**Stack:** Python 3.14 · Streamlit · sentence-transformers (MiniLM) · OpenCLIP · ChromaDB · Groq (Llama 4 Scout + Llama 3.1 8B + Whisper) · LangChain · MoviePy

---

## 1. Project Overview

Bayz is a multimodal Retrieval-Augmented Generation (RAG) chatbot that lets users upload a file — video, audio, PDF/DOCX/text document, or image — and ask natural-language questions about its contents. The system retrieves the most semantically relevant chunks or frames from the uploaded file and passes them as grounded context to a large language model, which produces a cited, evidence-backed answer.

Retrieval is made robust with a **Self-RAG-lite** loop: every retrieval is graded by embedding distance, and weak matches trigger an automatic query rewrite and re-retrieval — paying that cost only on hard queries. Small files bypass retrieval entirely and are passed whole to the LLM.

A context panel appears beneath each answer showing the top retrieved source: the video player seeked to the exact moment the answer is **spoken**, the audio player seeked to the matching transcript segment, the top-ranked text passage, or the source image.

---

## 2. Architecture

### 2.1 High-Level Design

The system is split into two strictly separated phases per uploaded file:

```
UPLOAD
  │
  ▼
┌─────────────────────────────────────────────┐
│  INGEST PHASE  (heavy, runs once per file)  │
│                                             │
│  Video  → frames + audio → transcribe       │
│           → timestamped chunks → embed      │
│  Audio  → transcribe → timestamped chunks   │
│  Doc    → load → split → embed chunks       │
│  Image  → resize → base64 encode            │
│                          │                  │
│                          ▼                  │
│           ChromaDB  (text + image stores)   │
└─────────────────────────────────────────────┘
                           │
                           │  RagContext (cached in session,
                           │   incl. full_text + segments)
                           ▼
┌─────────────────────────────────────────────┐
│  QUERY PHASE  (light, runs per question)    │
│                                             │
│  small file?  ──► pass full text (no search)│
│  else: embed query → retrieve top-k         │
│        → grade by distance                  │
│           weak? ─► rewrite query → re-search│
│        → prompt LLM                         │
│                                             │
│  Returns: (answer, top_context)             │
└─────────────────────────────────────────────┘
```

The `RagContext` dataclass is the handoff object: created once by an `ingest_*` function and cached in `st.session_state`, it carries the live ChromaDB collection references, frame-timestamp mappings, the original file path, the **full transcript/document text** (for the small-content bypass), and the **Whisper timestamped segments**. Every follow-up question reuses it without reprocessing the file.

### 2.2 Component Map

```
main.py               — Streamlit UI, session state, form handling, context panel
app/
  main.py             — ingest_*() + query() pipeline, RagContext dataclass
  processors/
    video_processor.py    — MoviePy frame extraction + audio track extraction
    audio_transcribe.py   — Groq Whisper (verbose_json) + segment timestamp capture
    doc_process.py        — PyMuPDF / TextLoader / Docx2txtLoader + splitter
    image_captioner.py    — PIL resize → JPEG → base64
  embeddings/
    models.py             — OpenCLIP ViT-B-32 model loader
    embedding_service.py  — EmbeddingsGenerator (dual: MiniLM text + CLIP image)
  vectordb/
    repository.py         — populate_text() (timestamp-aware chunking), populate_video_images()
    chroma_client.py      — ChromaDB collection factory
  chroma_db/
    db.py                 — get_collections() — single persistent ChromaDB client
  retrieval/
    ranking.py            — text_retrieval() (returns hits+distance), image_retrieval()
    search_service.py     — ask() + rewrite_query() (Self-RAG-lite)
    type_ask.py           — mode-specific prompt builders (image/audio/video/document)
```

### 2.3 Embedding Strategy

The system uses **two embedding spaces**, each chosen for the job it does best:

| Space | Model | Dim | Used for |
|---|---|---|---|
| **Text-to-text** | sentence-transformers `all-MiniLM-L6-v2` | 384 | documents, audio + video transcripts, and the user's query against text |
| **Text-to-image** | OpenCLIP `ViT-B-32` (LAION-2B) | 512 | video frames, and the user's query against frames |

This split is deliberate. CLIP's text encoder truncates at **77 tokens** and is tuned for short image captions, so it embedded only the first ~55 words of each passage — long content was effectively invisible to search. MiniLM is purpose-built for semantic text search, embeds full passages, and dramatically improves recall for the document/audio/video transcripts. CLIP is retained **only** for frames, where the text query and the frame images genuinely must share one contrastive space.

At query time:
- The query is embedded with **MiniLM** and searched against the text collection.
- For video, the same query is *also* embedded with **CLIP** (`embed_text_clip`) and searched against the frame collection.

### 2.4 Vector Store

ChromaDB runs in **persistent mode** on disk (`./chromadb_store`), with two collections:

| Collection | Content | Embedding | Metadata |
|---|---|---|---|
| `text_collection_v2` | Transcript / document text chunks | MiniLM (384-dim) | `timestamp` (audio/video chunks) |
| `image_collection` | Video frame embeddings | CLIP (512-dim) | `image_uri`, `timestamp` |

The text collection is named `text_collection_v2` because ChromaDB pins the embedding dimension per collection — the switch from CLIP (512-dim) to MiniLM (384-dim) required a fresh collection name to avoid a dimension-mismatch error against any pre-existing store.

For audio and video, text chunks now carry a `timestamp` in their metadata, built from Whisper's per-segment timing. This is what lets the answer be traced back to the exact moment it was spoken.

Before each new file is ingested, `_clear_collection()` deletes all existing entries by ID to prevent stale data from a previous upload polluting results.

### 2.5 LLM Layer

Two Groq-hosted models are used, sized to their task:

| Model | Role | Budget |
|---|---|---|
| `meta-llama/llama-4-scout-17b-16e-instruct` | Main answer generation (multimodal) | 1500 tokens |
| `llama-3.1-8b-instant` | Self-RAG-lite query rewrite (cheap, fast) | 120 tokens |

Invocation goes through LangChain's `ChatGroq` wrapper. The answer model receives a structured message pair:

- **SystemMessage**: mode-specific prompt. These were **relaxed** from the original hair-trigger versions — they now instruct the model to answer from partial evidence and synthesise across all snippets, reserving "Insufficient evidence" for genuinely empty retrievals rather than mere wording mismatches.
- **HumanMessage**: numbered context (frames with timestamps or text passages) + the user's query.

For video and image modes the human message includes `image_url` content blocks carrying base64-encoded JPEG frames, making the call natively multimodal.

### 2.6 Self-RAG-lite Retrieval Loop

To make retrieval robust without large latency cost, the query phase runs a lightweight self-reflection loop (`_retrieve_text` in `app/main.py`):

1. **Retrieve** the top-k chunks for the user's query.
2. **Grade** the best hit by its embedding distance — this is *free*, no LLM call. The threshold is `WEAK_DISTANCE = 1.15` (squared-L2 on normalised MiniLM vectors).
3. **Reflect & rewrite** only if the match is weak: one cheap call to `llama-3.1-8b-instant` rewrites the query (expanding synonyms, stripping filler), then re-retrieves. Whichever pass has the closer match wins.

Because step 3 fires only on weak retrievals, the common case stays single-pass and fast, while hard queries get a second chance instead of failing.

A complementary **small-content bypass** short-circuits the whole loop: if the entire transcript/document is ≤ 6000 characters (`SMALL_CONTENT_CHARS`), it is passed to the LLM in full with no retrieval at all — eliminating both retrieval latency and any chance of a relevant passage being missed on small files.

---

## 3. Mode-by-Mode Pipeline

### 3.1 Video Mode

```
MP4/MOV/AVI
  │
  ├─► MoviePy: extract frames at 0.9 FPS (adaptive cap: max 40 frames)
  │     → PNG files + timestamps.json mapping filename → seconds
  │
  ├─► MoviePy: extract audio track → audio.mp3
  │
  ├─► Groq Whisper (whisper-large-v3, verbose_json): transcribe audio
  │     → transcript.txt + segments.json (per-segment start/end timestamps)
  │
  ├─► segment-aware chunking (~600 chars/chunk, keeps start timestamp)
  │     → MiniLM text embeddings → text_collection_v2 (timestamp metadata)
  │
  └─► CLIP image embeddings per frame → image_collection (with timestamp metadata)

Query:
  small transcript?  → pass full text; else _retrieve_text() (MiniLM + Self-RAG-lite)
  + image_retrieval(top_k=3) via CLIP
  → video_query() prompt → Groq Llama 4 Scout
  → returns (answer, {type:"video", timestamp:<spoken-answer ts>, file_path:str})

  Timestamp priority: the transcript chunk that answers the query (where it is
  SPOKEN), falling back to the visually-matched frame only if no text timestamp.
```

### 3.2 Audio Mode

```
MP3/WAV/M4A/OGG/FLAC/AAC/WMA/OPUS
  │
  ├─► Groq Whisper → transcript.txt + segments.json (timestamps)
  │
  └─► segment-aware chunking → MiniLM embeddings → text_collection_v2 (timestamps)

Query:
  small transcript?  → pass full text; else _retrieve_text() (MiniLM + Self-RAG-lite)
  → Audio_query() prompt → Groq Llama 4 Scout
  → returns (answer, {type:"audio", text:str, timestamp:float, file_path:str})
```

### 3.3 Document Mode

```
PDF / DOCX / TXT
  │
  ├─► PyMuPDFLoader (PDF) / Docx2txtLoader (DOCX) / TextLoader (TXT)
  │
  └─► RecursiveCharacterTextSplitter → chunks → MiniLM embeddings → text_collection_v2

Query:
  small document?  → pass full text; else _retrieve_text() (MiniLM + Self-RAG-lite, top_k=4)
  → document_query() prompt → Groq Llama 4 Scout
  → returns (answer, {type:"document", text:<top chunk>})
```

### 3.4 Image Mode

```
JPG/PNG/WEBP/GIF
  │
  └─► PIL resize to 512×512 → JPEG → base64 string (no vector store)

Query:
  base64 payload sent directly to LLM as image_url block
  → image_query() prompt → Groq LLM
  → returns (answer, {type:"image", b64:str})
```

---

## 4. UI Design

The frontend is a single-file Streamlit app with a fully custom dark theme injected via `st.markdown(<style>)`. Key design decisions:

- **Sticky header** with mode label and file-ready status badge.
- **Sidebar** for mode selection (button group) and file upload.
- **Chat bubble layout**: user messages right-aligned (gradient fill), AI responses left-aligned with the Bayz label.
- **Top-1 context panel**: an `st.expander` rendered immediately below each AI bubble, constrained to ~40% width via `st.columns([4, 6])` so it never fills the screen. It shows the most relevant retrieved source:
  - Video → `st.video(file, start_time=int(timestamp))` seeked to the second the answer is **spoken**, capped at 220px height.
  - Audio → `st.audio(file, start_time=int(timestamp))` seeked to the matching segment, plus a caption with the transcript excerpt.
  - Document → styled div with the top retrieved passage (max 160px, scrollable; truncated at 400 chars).
  - Image → `st.image()` capped at 200px height.
- **Sample question chips** shown on the ready-state screen (before first message).
- **Animated thinking dots** (CSS keyframe animation) displayed while RAG is running.
- **25 MB file size cap** enforced client-side before any processing begins.

---

## 5. Technical Challenges & Solutions

### 5.1 Cold-Start Latency (CLIP Model Load)

**Problem:** The OpenCLIP ViT-B-32 model is several hundred MB and takes ~30–40 seconds to load on CPU. Importing it at module level would block Streamlit's first render, showing a blank screen.

**Solution:** Two-layer lazy loading. The `app.main` backend module itself is imported lazily via `_get_caller()` on the first query. Within the backend, `EmbeddingsGenerator` caches **both** models at class level — the MiniLM text model (`_text_shared`) and the CLIP image model (`_clip_shared`) — each materialised on its first use and reused across all instances and queries for the session lifetime.

### 5.2 Enter Key Not Triggering Search

**Problem:** The original implementation used a standalone `st.text_input` paired with a `st.button`. In Streamlit, pressing Enter in a text input triggers a script rerun but does not click buttons — the user had to click the Send button with the mouse.

**Solution:** Wrapped the input and submit button inside a `st.form(clear_on_submit=True)` with a `st.form_submit_button`. Streamlit forms intercept the Enter keypress as a form submission, firing `send_clicked=True` identically to a mouse click. The form key rotates with `input_key` so the widget is recreated blank after each send.

### 5.3 ChromaDB Duplicate-ID Errors on Re-Upload

**Problem:** ChromaDB collections persist to disk. On a second query with a different file, calling `.add()` with the same IDs (`text_0`, `text_1`, …) raised a duplicate-ID exception.

**Solution:** `_clear_collection()` in `repository.py` fetches all existing IDs via `collection.get()` and calls `collection.delete(ids=...)` before every ingestion. This keeps each query scoped to the freshly uploaded file.

### 5.4 Weak Retrieval & "Insufficient Evidence" Answers

**Problem:** The model frequently replied *"Insufficient evidence to answer this query"* even when the answer was plainly in the file. Root cause was threefold:
1. **CLIP text encoder truncates at 77 tokens** — long chunks were embedded from only their first ~55 words, so most content was invisible to search.
2. **CLIP text-to-text matching is weak** — CLIP is tuned for image captions, not question→passage retrieval.
3. **Hair-trigger prompts** — the system prompts treated "Insufficient evidence" as the safe default.

**Solution:** A three-part fix.
- **Proper text embedder:** swapped the text path to `sentence-transformers/all-MiniLM-L6-v2` (purpose-built for semantic search, embeds full passages). CLIP is kept only for frame images. This alone fixed the bulk of the failures.
- **Self-RAG-lite:** grade every retrieval by embedding distance (free); on a weak match, do one cheap `llama-3.1-8b-instant` query rewrite and re-retrieve. Small files bypass retrieval entirely (full text passed to the LLM).
- **Relaxed prompts:** rewrote the audio/video/document system prompts to answer from partial evidence and reserve "Insufficient evidence" for genuinely empty retrievals.

A functional test confirmed the fix end-to-end: queries that previously failed now return strong matches (distance 0.85–0.98), and a deliberately vague query correctly crossed the 1.15 threshold and triggered the rewrite fallback.

### 5.5 Wrong Video Timestamp

**Problem:** The video context panel showed the timestamp of `retrieved_images[0]` — the frame that *visually* matched the query via CLIP. But the *answer* comes from the *transcript*, so the displayed timestamp pointed at an arbitrary-looking frame, not where the answer was actually spoken. Worse, Whisper's per-segment timestamps (returned in `verbose_json`) were being discarded entirely.

**Solution:** `audio_transcribe.py` now captures Whisper's segments and persists them to `segments.json`. `repository.py` builds text chunks *from those segments* (`_chunks_from_segments`), attaching each chunk's start time as ChromaDB metadata. At query time, `text_retrieval` returns these timestamps, and `query()` uses the matched transcript chunk's timestamp — the moment the answer is **spoken** — falling back to the visual frame only when no text timestamp exists. The same mechanism now lets audio mode seek its player to the relevant segment.

### 5.6 Frame Count Explosion for Long Videos

**Problem:** At 0.9 FPS, a 90-minute video would produce ~4860 frames — far too many to embed or pass to an LLM.

**Solution:** An adaptive sampling cap in `video_processor.py`: if `int(duration × fps) + 1 > 40`, the extractor falls back to sampling exactly 40 frames evenly spaced across the full duration. The cap is configurable (`max_frames=40`).

### 5.7 Videos With No Audio Track

**Problem:** Attempting to extract audio from a video with no audio stream raised an exception and left a stale `.mp3` file from a previous video.

**Solution:** `clip.audio is not None` guard before extraction. If no audio is present, any existing stale `audio.mp3` is explicitly deleted so Whisper doesn't pick it up. The transcript is treated as an empty string and the text collection is left unpopulated.

### 5.8 Two-Phase Pipeline / Avoiding Re-Ingestion

**Problem:** Users ask multiple follow-up questions about the same file. Re-running the full ingest pipeline (frame extraction, Whisper transcription, embedding) on every message would make the app unusable.

**Solution:** The `RagContext` dataclass holds live references to the ChromaDB collections, frame timestamp map, image data, the original file path, the full text, and the Whisper segments. It is cached in `st.session_state.rag_ctx` keyed to `ingested_path`. The `_ensure_ingested()` helper compares the current `file_path` against `ingested_path` and only runs ingestion when they differ. Follow-up questions skip straight to the fast query phase.

### 5.9 Returning Context Alongside the Answer

**Problem:** The original `query()` returned only a `str`. The UI had no way to know which timestamp or passage was most relevant to show alongside the answer.

**Solution:** `query()` was changed to return `(answer: str, top_context: dict | None)`. The `top_context` dict carries type-specific fields (`timestamp` + `file_path` for video, `text` + `timestamp` + `file_path` for audio, `text` for document, `b64` for image). The assistant message stored in `st.session_state.messages` now carries a `top_ctx` key, and the chat render loop checks it to display the appropriate Streamlit widget below each AI bubble.

### 5.10 Windows / FFmpeg Compatibility

**Problem:** Several audio-processing libraries on Windows require FFmpeg to be installed system-wide. This is fragile on developer machines and unavailable on Streamlit Cloud.

**Solution:** Groq Whisper API accepts raw `.mp3`, `.m4a`, `.wav`, `.ogg`, and other formats natively — no local FFmpeg transcoding needed. MoviePy uses `imageio-ffmpeg` (a bundled FFmpeg binary shipped as a Python wheel) for frame and audio extraction, which works without a system FFmpeg installation.

### 5.11 DOCX Upload Crash

**Problem:** The UI accepted `.docx` uploads, but `get_document_loader()` only mapped `.pdf` and `.txt`, so every DOCX raised `ValueError: Unsupported file extension: '.docx'`.

**Solution:** Added `docx2txt` and a `docx_loader()` using LangChain's `Docx2txtLoader`, registered against `.docx` in the loader map.

### 5.12 Windows asyncio ConnectionResetError Noise

**Problem:** On Windows, Python's default `ProactorEventLoop` logs a spurious `ConnectionResetError [WinError 10054]` traceback when a remote host (e.g. the Groq API) closes a TCP connection normally — alarming but harmless console noise.

**Solution:** At the top of `main.py`, behind a `sys.platform == "win32"` guard, the loop policy is switched to `WindowsSelectorEventLoopPolicy`, which handles remote closes silently.

### 5.13 Context Panel Covering the Screen

**Problem:** The retrieval context panel (video/audio/image/text) rendered at full container width, dominating the chat.

**Solution:** Each panel is rendered inside `st.columns([4, 6])` (≈40% width) with per-widget `max-height` + `overflow` caps (220px video, 200px image, 160px scrollable text).

---

## 6. Dependency Stack

| Package | Version | Role |
|---|---|---|
| streamlit | 1.58.0 | Web UI framework |
| torch (CPU) | 2.12.0+cpu | Tensor ops for CLIP + MiniLM |
| sentence-transformers | 5.5.1 | MiniLM text embeddings (semantic retrieval) |
| open-clip-torch | 3.3.0 | ViT-B-32 CLIP embeddings (frames) |
| pillow | 11.3.0 | Image resize + base64 encode |
| chromadb | 1.5.9 | Local persistent vector store |
| langchain | 1.3.7 | RAG orchestration |
| langchain-groq | 1.1.3 | Groq LLM wrapper |
| groq | 0.37.1 | Groq API client (Whisper + LLM) |
| pymupdf | 1.27.2.3 | PDF text extraction |
| docx2txt | 0.9 | DOCX text extraction |
| moviepy | 2.2.1 | Video frame + audio extraction |
| imageio-ffmpeg | 0.6.0 | Bundled FFmpeg for MoviePy |
| python-dotenv | 1.2.2 | `.env` secrets loading |
| numpy | 2.4.6 | Embedding array ops |

---

## 7. Data Flow Diagram

```
User uploads file
      │
      ▼
 _save_upload() ──► temp file on disk
      │
      ▼
 ingest(rag_type, file_path)
      │
      ├── video ──► process_video() ──► frames (PNG) + timestamps.json
      │                │                      │
      │                │             embed_image() ──► image_collection (ChromaDB)
      │                │
      │             transcribe_audio() ──► transcript.txt
      │                │
      │             get_document_loader() + text_splitter()
      │                │
      │             embed_text() ──► text_collection (ChromaDB)
      │
      ├── audio ──► transcribe_audio() ──► text_splitter() ──► text_collection
      │
      ├── document ──► pdf_loader / document_loader ──► text_splitter() ──► text_collection
      │
      └── image ──► encode_image_to_base64() ──► RagContext.image_data
                              │
                         (no ChromaDB)

                    RagContext cached in session_state
                       (collections + full_text + segments)
                              │
User asks question            │
      │                       ▼
      └──► query(ctx, query_str)
                │
                ├── small file? ──► pass full_text to LLM (no search)
                │
                ├── else: embed_text (MiniLM) ──► ChromaDB cosine search
                │         │
                │     grade best distance
                │         │
                │     weak? ──► rewrite_query (Llama 3.1 8B) ──► re-search
                │         │
                │    retrieved_texts (+timestamps) + frames (CLIP, video only)
                │
                └── type_ask.*_query() ──► SystemMessage + HumanMessage
                              │
                         ChatGroq.invoke()  (Llama 4 Scout)
                              │
                    (answer, top_context)   top_context.timestamp =
                              │              spoken-answer moment
                 stored in session_state.messages
                              │
                    rendered in chat + context panel (seeked player)
```

---

## 8. Known Limitations

1. **Single-user / local only.** ChromaDB is a local persistent store. All uploads share the same two collections, so concurrent users would corrupt each other's data. Production would need per-session namespacing or a remote vector database.

2. **CPU-only inference.** The CPU PyTorch build keeps the deployment size manageable, but CLIP frame embedding on CPU is slow (~1–3 seconds per frame). A video with 40 frames takes ~40–120 seconds to ingest. (MiniLM text embedding is comparatively fast.)

3. **Image mode has no RAG.** The image pipeline sends the raw image directly to the LLM without any vector retrieval step. For multi-image scenarios, this would not scale.

4. **Single rewrite pass.** Self-RAG-lite performs at most one query rewrite. A genuinely hard query that the rewrite doesn't fix still falls through to a best-effort answer; there is no multi-round retrieval refinement.

5. **First-run model download.** `all-MiniLM-L6-v2` (~80 MB) downloads on first use and adds a one-time ~10–15 s load. Subsequent sessions reuse the cached weights.

6. **No persistence across server restarts of `RagContext`.** The cached context lives in Streamlit session state; restarting the server forces re-ingestion of the active file.

---

## 9. Possible Improvements

- **Per-session ChromaDB namespacing:** Prefix collection names with a session UUID to support concurrent users.
- **GPU acceleration:** Swap `torch==2.12.0+cpu` for a CUDA wheel when running on a GPU host to cut frame-embedding time from minutes to seconds.
- **Streaming LLM responses:** Use LangChain's streaming callbacks to stream tokens into the chat bubble, reducing perceived latency.
- **Cross-encoder re-ranking:** Add a re-ranker between retrieval and prompting to further sharpen the top-k context — a natural next step now that base retrieval is strong.
- **Multi-round Self-RAG:** Allow more than one rewrite/retrieve cycle for hard queries, bounded by a budget, with an LLM relevance grader in addition to the distance grader.
- **Word-level audio seeking:** Use Whisper's word-level timestamps (not just segments) for even finer audio/video seek precision.
- **HTML / PPTX support:** Add `BSHTMLLoader` and a PPTX loader to `get_document_loader()`.
