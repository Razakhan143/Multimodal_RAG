# Bayz — Multimodal RAG Chatbot
## Project Report

**Author:** raza khan  
**Date:** June 2026  
**Stack:** Python 3.14 · Streamlit · OpenCLIP · ChromaDB · Groq (Llama 4 Scout + Whisper) · LangChain · MoviePy

---

## 1. Project Overview

Bayz is a multimodal Retrieval-Augmented Generation (RAG) chatbot that lets users upload a file — video, audio, PDF/text document, or image — and ask natural-language questions about its contents. The system retrieves the most semantically relevant chunks or frames from the uploaded file and passes them as grounded context to a large language model, which produces a cited, evidence-backed answer.

A context panel appears beneath each answer showing the top retrieved source: the video player seeked to the exact relevant timestamp, the audio file with the matching transcript snippet, the top-ranked text passage, or the source image.

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
│  Video  → extract frames + audio → embed   │
│  Audio  → transcribe → embed chunks        │
│  Doc    → load → split → embed chunks      │
│  Image  → resize → base64 encode           │
│                          │                 │
│                          ▼                 │
│                    ChromaDB store          │
└─────────────────────────────────────────────┘
                           │
                           │  RagContext (cached in session)
                           ▼
┌─────────────────────────────────────────────┐
│  QUERY PHASE  (light, runs per question)    │
│                                             │
│  Embed query → retrieve top-k → prompt LLM │
│                                             │
│  Returns: (answer, top_context)             │
└─────────────────────────────────────────────┘
```

The `RagContext` dataclass is the handoff object: created once by an `ingest_*` function and cached in `st.session_state`, it carries the live ChromaDB collection references, frame-timestamp mappings, and the original file path. Every follow-up question reuses it without reprocessing the file.

### 2.2 Component Map

```
main.py               — Streamlit UI, session state, form handling, context panel
app/
  main.py             — ingest_*() + query() pipeline, RagContext dataclass
  processors/
    video_processor.py    — MoviePy frame extraction + audio track extraction
    audio_transcribe.py   — Groq Whisper transcription (verbose_json)
    doc_process.py        — PyMuPDF / TextLoader + RecursiveCharacterTextSplitter
    image_captioner.py    — PIL resize → JPEG → base64
  embeddings/
    models.py             — OpenCLIP ViT-B-32 model loader
    embedding_service.py  — EmbeddingsGenerator (lazy-loaded, class-level shared)
  vectordb/
    repository.py         — populate_text(), populate_video_images(), _clear_collection()
    chroma_client.py      — ChromaDB collection factory
  chroma_db/
    db.py                 — get_collections() — single persistent ChromaDB client
  retrieval/
    ranking.py            — text_retrieval(), image_retrieval(), video_retrival()
    search_service.py     — ask() — LLM invocation via LangChain-Groq
    type_ask.py           — mode-specific prompt builders (image/audio/video/document)
```

### 2.3 Embedding Strategy

All modalities share a single vector space via **OpenCLIP ViT-B-32** (pretrained on LAION-2B). This is the architectural centrepiece:

- Text chunks and transcript segments are embedded with `encode_text`.
- Video frames are embedded with `encode_image`.
- At query time, the user's text question is embedded with `encode_text` and compared against **both** the text collection and the image collection using cosine similarity in ChromaDB.

Because CLIP is trained with a contrastive text–image objective, a text query such as *"the part where they discuss pricing"* can retrieve a matching video frame even with no explicit caption, and vice versa.

### 2.4 Vector Store

ChromaDB runs in **persistent mode** on disk (`./chromadb_store`), with two fixed collections:

| Collection | Content | Metadata |
|---|---|---|
| `text_collection` | Transcript / document text chunks | — |
| `image_collection` | Video frame embeddings | `image_uri`, `timestamp` |

Before each new file is ingested, `_clear_collection()` deletes all existing entries by ID to prevent stale data from a previous upload polluting results.

### 2.5 LLM Layer

The LLM is **meta-llama/llama-4-scout-17b-16e-instruct** served through the Groq API (max 1500 output tokens). Invocation goes through LangChain's `ChatGroq` wrapper. The model receives a structured message pair:

- **SystemMessage**: mode-specific epistemic prompt (strict grounding rules, citation requirements, hallucination-resistance instructions).
- **HumanMessage**: numbered context (frames with timestamps or text passages) + the user's query.

For video and image modes the human message includes `image_url` content blocks carrying base64-encoded JPEG frames, making the call natively multimodal.

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
  │     → transcript.txt
  │
  ├─► PyMuPDF TextLoader + RecursiveCharacterTextSplitter (1000 chars / 200 overlap)
  │     → text chunks → CLIP text embeddings → ChromaDB text_collection
  │
  └─► CLIP image embeddings per frame → ChromaDB image_collection (with timestamp metadata)

Query:
  text_retrieval(top_k=5) + image_retrieval(top_k=3)
  → video_query() prompt → Groq LLM
  → returns (answer, {type:"video", timestamp:float, file_path:str})
```

### 3.2 Audio Mode

```
MP3/WAV/M4A/OGG/FLAC/AAC/WMA/OPUS
  │
  ├─► Groq Whisper → transcript.txt
  │
  └─► text chunks → CLIP text embeddings → ChromaDB text_collection

Query:
  text_retrieval(top_k=3)
  → Audio_query() prompt → Groq LLM
  → returns (answer, {type:"audio", text:str, file_path:str})
```

### 3.3 Document Mode

```
PDF / TXT
  │
  ├─► PyMuPDFLoader (PDF) or TextLoader (TXT)
  │
  └─► RecursiveCharacterTextSplitter → chunks → CLIP text embeddings → ChromaDB

Query:
  text_retrieval(top_k=3)
  → document_query() prompt → Groq LLM
  → returns (answer, {type:"document", text:str})
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
- **Top-1 context panel**: an `st.expander` rendered immediately below each AI bubble, showing the most relevant retrieved source:
  - Video → `st.video(file, start_time=int(timestamp))` seeked to exact second.
  - Audio → `st.audio(file)` + caption with the matching transcript excerpt.
  - Document → styled div with the top text passage (truncated at 600 chars).
  - Image → `st.image()` rendered inline.
- **Sample question chips** shown on the ready-state screen (before first message).
- **Animated thinking dots** (CSS keyframe animation) displayed while RAG is running.
- **25 MB file size cap** enforced client-side before any processing begins.

---

## 5. Technical Challenges & Solutions

### 5.1 Cold-Start Latency (CLIP Model Load)

**Problem:** The OpenCLIP ViT-B-32 model is several hundred MB and takes ~30–40 seconds to load on CPU. Importing it at module level would block Streamlit's first render, showing a blank screen.

**Solution:** Two-layer lazy loading. The `app.main` backend module itself is imported lazily via `_get_caller()` on the first query. Within the backend, `EmbeddingsGenerator` uses a class-level `_shared` tuple — the model is loaded once on first embed call and reused across all instances and all queries for the session lifetime.

### 5.2 Enter Key Not Triggering Search

**Problem:** The original implementation used a standalone `st.text_input` paired with a `st.button`. In Streamlit, pressing Enter in a text input triggers a script rerun but does not click buttons — the user had to click the Send button with the mouse.

**Solution:** Wrapped the input and submit button inside a `st.form(clear_on_submit=True)` with a `st.form_submit_button`. Streamlit forms intercept the Enter keypress as a form submission, firing `send_clicked=True` identically to a mouse click. The form key rotates with `input_key` so the widget is recreated blank after each send.

### 5.3 ChromaDB Duplicate-ID Errors on Re-Upload

**Problem:** ChromaDB collections persist to disk. On a second query with a different file, calling `.add()` with the same IDs (`text_0`, `text_1`, …) raised a duplicate-ID exception.

**Solution:** `_clear_collection()` in `repository.py` fetches all existing IDs via `collection.get()` and calls `collection.delete(ids=...)` before every ingestion. This keeps each query scoped to the freshly uploaded file.

### 5.4 Shared Embedding Space for Cross-Modal Retrieval

**Problem:** A text query needs to match both transcript text chunks and video frames in the same similarity search.

**Solution:** CLIP's joint text–image embedding space is used for all modalities. Both `embed_text()` and `embed_image()` produce L2-normalised 512-dimensional vectors in the same space. The same query embedding can therefore be sent to both `text_collection.query()` and `image_collection.query()` without any adaptation layer.

### 5.5 Frame Count Explosion for Long Videos

**Problem:** At 0.9 FPS, a 90-minute video would produce ~4860 frames — far too many to embed or pass to an LLM.

**Solution:** An adaptive sampling cap in `video_processor.py`: if `int(duration × fps) + 1 > 40`, the extractor falls back to sampling exactly 40 frames evenly spaced across the full duration. The cap is configurable (`max_frames=40`).

### 5.6 Videos With No Audio Track

**Problem:** Attempting to extract audio from a video with no audio stream raised an exception and left a stale `.mp3` file from a previous video.

**Solution:** `clip.audio is not None` guard before extraction. If no audio is present, any existing stale `audio.mp3` is explicitly deleted so Whisper doesn't pick it up. The transcript is treated as an empty string and the text collection is left unpopulated.

### 5.7 Two-Phase Pipeline / Avoiding Re-Ingestion

**Problem:** Users ask multiple follow-up questions about the same file. Re-running the full ingest pipeline (frame extraction, Whisper transcription, CLIP embedding) on every message would make the app unusable.

**Solution:** The `RagContext` dataclass holds live references to the ChromaDB collections, frame timestamp map, image data, and the original file path. It is cached in `st.session_state.rag_ctx` keyed to `ingested_path`. The `_ensure_ingested()` helper compares the current `file_path` against `ingested_path` and only runs ingestion when they differ. Follow-up questions skip straight to the fast query phase.

### 5.8 Returning Context Alongside the Answer

**Problem:** The original `query()` returned only a `str`. The UI had no way to know which timestamp or passage was most relevant to show alongside the answer.

**Solution:** `query()` was changed to return `(answer: str, top_context: dict | None)`. The `top_context` dict carries type-specific fields (`timestamp` + `file_path` for video, `text` + `file_path` for audio, `text` for document, `b64` for image). The assistant message stored in `st.session_state.messages` now carries a `top_ctx` key, and the chat render loop checks it to display the appropriate Streamlit widget below each AI bubble.

### 5.9 Windows / FFmpeg Compatibility

**Problem:** Several audio-processing libraries on Windows require FFmpeg to be installed system-wide. This is fragile on developer machines and unavailable on Streamlit Cloud.

**Solution:** Groq Whisper API accepts raw `.mp3`, `.m4a`, `.wav`, `.ogg`, and other formats natively — no local FFmpeg transcoding needed. MoviePy uses `imageio-ffmpeg` (a bundled FFmpeg binary shipped as a Python wheel) for frame and audio extraction, which works without a system FFmpeg installation.

---

## 6. Dependency Stack

| Package | Version | Role |
|---|---|---|
| streamlit | 1.58.0 | Web UI framework |
| torch (CPU) | 2.12.0+cpu | Tensor ops for CLIP |
| open-clip-torch | 3.3.0 | ViT-B-32 CLIP embeddings |
| pillow | 11.3.0 | Image resize + base64 encode |
| chromadb | 1.5.9 | Local persistent vector store |
| langchain | 1.3.7 | RAG orchestration |
| langchain-groq | 1.1.3 | Groq LLM wrapper |
| groq | 0.37.1 | Groq API client (Whisper + LLM) |
| pymupdf | 1.27.2.3 | PDF text extraction |
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
                              │
User asks question            │
      │                       ▼
      └──► query(ctx, query_str)
                │
                ├── embed_text(query_str) ──► ChromaDB cosine search
                │         │
                │    retrieved_texts + retrieved_images (with timestamps)
                │
                └── type_ask.*_query() ──► SystemMessage + HumanMessage
                              │
                         ChatGroq.invoke()
                              │
                    (answer, top_context)
                              │
                 stored in session_state.messages
                              │
                    rendered in chat + context panel
```

---

## 8. Known Limitations

1. **Single-user / local only.** ChromaDB is a local persistent store. All uploads share the same two collections, so concurrent users would corrupt each other's data. Production would need per-session namespacing or a remote vector database.

2. **Text-only ChromaDB metadata for audio.** Audio mode stores transcript chunks as plain text without segment timestamps. The top-1 context panel plays the full audio file from the start rather than seeking to a specific second. Whisper's `verbose_json` response includes word-level timestamps that could be used to fix this.

3. **No DOCX support.** The document loader only handles `.pdf` and `.txt`. DOCX, PPTX, and HTML are unsupported.

4. **CLIP text token limit.** OpenCLIP tokenises at most 77 tokens. Text chunks up to 1000 characters are split by `RecursiveCharacterTextSplitter`, but long chunks are silently truncated during embedding, which can degrade retrieval quality for dense technical text.

5. **CPU-only inference.** The CPU PyTorch build keeps the deployment size manageable, but CLIP embedding on CPU is slow (~1–3 seconds per frame). A video with 40 frames takes ~40–120 seconds to ingest.

6. **Image mode has no RAG.** The image pipeline sends the raw image directly to the LLM without any vector retrieval step. For multi-image scenarios, this would not scale.

---

## 9. Possible Improvements

- **Whisper timestamp alignment:** Parse `verbose_json` segment timestamps from the Whisper response and store them as metadata on text chunks, enabling audio-mode context panels to seek to the correct second.
- **Per-session ChromaDB namespacing:** Prefix collection names with a session UUID to support concurrent users.
- **DOCX / HTML support:** Add `Docx2txtLoader` and `BSHTMLLoader` to `get_document_loader()`.
- **GPU acceleration:** Swap `torch==2.12.0+cpu` for a CUDA wheel when running on a GPU host to cut embedding time from minutes to seconds.
- **Streaming LLM responses:** Use LangChain's streaming callbacks to stream tokens into the chat bubble, reducing perceived latency.
- **Re-ranking:** Add a cross-encoder re-ranker between retrieval and prompting to improve precision of the top-k context sent to the LLM.
