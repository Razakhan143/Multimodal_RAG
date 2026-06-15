"""Multimodal RAG backend.

The pipeline is split into two phases so the expensive work runs only once per
uploaded file:

* ``ingest_*``  — heavy: process video/audio, transcribe, embed, populate the
  vector store. Returns a ``RagContext`` describing what was ingested.
* ``query_*``   — light: retrieve from the already-populated store and ask the
  LLM. Safe to call repeatedly for follow-up questions on the same file.

The original ``*_rag_system`` functions are kept as thin wrappers (ingest +
query in one call) for backwards compatibility and one-off scripts.
"""

from dataclasses import dataclass, field
from typing import Any

from app.processors import audio_transcribe
from app.retrieval.search_service import ask, rewrite_query
from app.retrieval.ranking import text_retrieval, image_retrieval, WEAK_DISTANCE
from app.processors.video_processor import process_video
from app.vectordb.repository import populate_text, populate_video_images
from app.chroma_db.db import get_collections
from app.processors.doc_process import get_document_loader
from app.processors.image_captioner import encode_image_to_base64

# Paths produced by the video processor (frames + extracted audio + transcript).
AUDIO_OUT = "app/ingestion/audio_ingess/audio.mp3"

# If the entire transcript/document fits comfortably in the LLM context, skip
# retrieval and pass everything. This eliminates "insufficient evidence" misses
# on small files and removes retrieval latency entirely for them.
SMALL_CONTENT_CHARS = 6000


def _document_text(document) -> str:
    """Flatten a LangChain document (or list) to a single plain string."""
    if document is None:
        return ""
    if isinstance(document, (list, tuple)):
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in document)
    return getattr(document, "page_content", str(document))


@dataclass
class RagContext:
    """Everything needed to answer follow-up queries without re-ingesting.

    Created once by an ``ingest_*`` function and reused by ``query_*`` for every
    subsequent question about the same file.
    """
    rag_type: str
    text_collection: Any = None
    image_collection: Any = None
    frame_timestamps: dict = field(default_factory=dict)
    image_data: list = field(default_factory=list)   # for image mode (b64 payloads)
    file_path: str = None                             # original uploaded file path
    full_text: str = ""                               # whole transcript/doc (bypass)
    segments: list = field(default_factory=list)      # Whisper timestamped segments


# ──────────────────────────────────────────────────────────────────────────────
# INGESTION  — heavy, runs once per uploaded file
# ──────────────────────────────────────────────────────────────────────────────

def ingest_video(video_path: str) -> RagContext:
    """Process a video once: extract frames, transcribe audio, embed + store."""
    print(f"Ingesting video: {video_path}")
    frame_timestamps = process_video(video_path)

    transcript = audio_transcribe.transcribe_audio(AUDIO_OUT)
    segments = audio_transcribe.load_segments()

    text_collection, image_collection = get_collections()
    text_collection = populate_text(transcript, text_collection, segments=segments) or text_collection
    image_collection = populate_video_images(image_collection, frame_timestamps)

    return RagContext(
        rag_type="video",
        text_collection=text_collection,
        image_collection=image_collection,
        frame_timestamps=frame_timestamps,
        file_path=video_path,
        full_text=transcript or "",
        segments=segments,
    )


def ingest_audio(audio_path: str) -> RagContext:
    """Transcribe an audio file once and embed + store the transcript."""
    print(f"Ingesting audio: {audio_path}")
    transcript = audio_transcribe.transcribe_audio(audio_path)
    segments = audio_transcribe.load_segments()

    text_collection, _ = get_collections()
    text_collection = populate_text(transcript, text_collection, segments=segments) or text_collection

    return RagContext(
        rag_type="audio",
        text_collection=text_collection,
        file_path=audio_path,
        full_text=transcript or "",
        segments=segments,
    )


def ingest_document(doc_path: str) -> RagContext:
    """Load a document once and embed + store its chunks."""
    print(f"Ingesting document: {doc_path}")
    document = get_document_loader(doc_path)
    text_collection, _ = get_collections()
    text_collection = populate_text(document, text_collection) or text_collection

    return RagContext(
        rag_type="document",
        text_collection=text_collection,
        file_path=doc_path,
        full_text=_document_text(document),
    )


def ingest_image(img_path: str) -> RagContext:
    """Encode an image once; no vector store needed (sent inline to the LLM)."""
    print(f"Ingesting image: {img_path}")
    img_data = {"b64": encode_image_to_base64(img_path)}
    return RagContext(rag_type="image", image_data=[img_data], file_path=img_path)


def ingest(rag_type: str, file_path: str) -> RagContext:
    """Dispatch to the correct ``ingest_*`` function by RAG type."""
    dispatch = {
        "video": ingest_video,
        "audio": ingest_audio,
        "document": ingest_document,
        "image": ingest_image,
    }
    if rag_type not in dispatch:
        raise ValueError(f"Unknown RAG type: {rag_type!r}")
    return dispatch[rag_type](file_path)


# ──────────────────────────────────────────────────────────────────────────────
# QUERYING  — light, runs for every follow-up question
# ──────────────────────────────────────────────────────────────────────────────

def _retrieve_text(ctx: RagContext, query_str: str, top_k: int):
    """Retrieve text with a Self-RAG-lite rewrite fallback.

    1. Retrieve once with the user's query.
    2. Grade the best hit by embedding distance (free — no LLM call).
    3. Only if the match is weak, spend ONE cheap LLM call to rewrite the query
       and retrieve again, then keep whichever pass had the closer match.

    This bounds the extra latency to a single small-model call, and only on the
    hard queries that actually need it. Returns (texts, hits).
    """
    texts, hits, best = text_retrieval(query_str, ctx.text_collection, top_k=top_k)

    if best <= WEAK_DISTANCE:
        return texts, hits   # strong match — answer immediately, no rewrite

    # Weak retrieval → one rewrite + re-retrieve.
    print(f"🔎 Weak retrieval (dist={best:.3f}); rewriting query…")
    rq = rewrite_query(query_str)
    if rq and rq.lower() != query_str.lower():
        texts2, hits2, best2 = text_retrieval(rq, ctx.text_collection, top_k=top_k)
        if best2 < best:
            return texts2, hits2
    return texts, hits


def _top_timestamp(hits: list):
    """First non-None timestamp among ranked hits (where the answer is spoken)."""
    for h in hits:
        ts = h.get("timestamp")
        if ts is not None:
            return ts
    return None


def query(ctx: RagContext, query_str: str) -> tuple:
    """Answer a question using an already-ingested ``RagContext``.

    Returns
    -------
    (answer: str, top_context: dict | None)
        top_context keys vary by type:
          video    → {"type": "video",    "timestamp": float, "file_path": str}
          audio    → {"type": "audio",    "text": str, "file_path": str}
          document → {"type": "document", "text": str}
          image    → {"type": "image",    "b64": str}
    """
    if ctx.rag_type == "video":
        small = 0 < len(ctx.full_text) <= SMALL_CONTENT_CHARS
        if small:
            # Whole transcript fits — pass it all; still pull frames for vision.
            retrieved_texts = [ctx.full_text]
            hits = []
        else:
            retrieved_texts, hits = _retrieve_text(ctx, query_str, top_k=5)

        retrieved_images = image_retrieval(query_str, ctx.image_collection, top_k=3)
        answer = ask(query_str, retrieved_texts, retrieved_images, type="video")

        # Timestamp priority: the transcript chunk that answers the query (where
        # it is *spoken*), falling back to the visually-matched frame.
        ts = _top_timestamp(hits)
        if ts is None and retrieved_images:
            ts = retrieved_images[0].get("timestamp")
        top_ctx = {"type": "video", "timestamp": ts, "file_path": ctx.file_path}
        return answer, top_ctx

    if ctx.rag_type == "audio":
        small = 0 < len(ctx.full_text) <= SMALL_CONTENT_CHARS
        if small:
            retrieved_texts, hits = [ctx.full_text], []
        else:
            retrieved_texts, hits = _retrieve_text(ctx, query_str, top_k=3)

        answer = ask(query_str, retrieved_texts, [], type="audio")
        top_ctx = {
            "type": "audio",
            "text": retrieved_texts[0] if retrieved_texts else "",
            "timestamp": _top_timestamp(hits),
            "file_path": ctx.file_path,
        }
        return answer, top_ctx

    if ctx.rag_type == "document":
        small = 0 < len(ctx.full_text) <= SMALL_CONTENT_CHARS
        if small:
            retrieved_texts, hits = [ctx.full_text], []
        else:
            retrieved_texts, hits = _retrieve_text(ctx, query_str, top_k=4)

        answer = ask(query_str, retrieved_texts, type="document")
        top_ctx = None
        if retrieved_texts:
            # Show the top retrieved chunk, not the whole bypassed transcript.
            snippet = hits[0]["text"] if hits else retrieved_texts[0]
            top_ctx = {"type": "document", "text": snippet}
        return answer, top_ctx

    if ctx.rag_type == "image":
        answer = ask(query_str, [], ctx.image_data, type="image")
        top_ctx = None
        if ctx.image_data:
            top_ctx = {"type": "image", "b64": ctx.image_data[0].get("b64", "")}
        return answer, top_ctx

    raise ValueError(f"Unknown RAG type: {ctx.rag_type!r}")


# ──────────────────────────────────────────────────────────────────────────────
# BACKWARDS-COMPAT WRAPPERS  — ingest + query in a single call
# ──────────────────────────────────────────────────────────────────────────────

def video_rag_system(video_url: str, query_str: str, rag: str = "video") -> str:
    answer, _ = query(ingest_video(video_url), query_str)
    return answer


def audio_rag_system(audio_path: str, query_str: str, rag: str = "audio") -> str:
    answer, _ = query(ingest_audio(audio_path), query_str)
    return answer


def image_rag_system(img_url: str, query_str: str, rag: str = "image") -> str:
    answer, _ = query(ingest_image(img_url), query_str)
    return answer


def document_rag_system(doc_url: str, query_str: str, rag: str = "document") -> str:
    answer, _ = query(ingest_document(doc_url), query_str)
    return answer
