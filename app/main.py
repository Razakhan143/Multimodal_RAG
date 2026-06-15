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
from app.retrieval.search_service import ask
from app.retrieval.ranking import video_retrival, text_retrieval
from app.processors.video_processor import process_video
from app.vectordb.repository import populate_text, populate_video_images
from app.chroma_db.db import get_collections
from app.processors.doc_process import get_document_loader
from app.processors.image_captioner import encode_image_to_base64

# Paths produced by the video processor (frames + extracted audio + transcript).
AUDIO_OUT = "app/ingestion/audio_ingess/audio.mp3"
TRANSCRIPT_OUT = "app/ingestion/audio_ingess/transcript.txt"


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


# ──────────────────────────────────────────────────────────────────────────────
# INGESTION  — heavy, runs once per uploaded file
# ──────────────────────────────────────────────────────────────────────────────

def ingest_video(video_path: str) -> RagContext:
    """Process a video once: extract frames, transcribe audio, embed + store."""
    print(f"Ingesting video: {video_path}")
    frame_timestamps = process_video(video_path)

    transcript = audio_transcribe.transcribe_audio(AUDIO_OUT)
    document = get_document_loader(file_path=TRANSCRIPT_OUT) if transcript else None

    text_collection, image_collection = get_collections()
    text_collection = populate_text(document, text_collection) or text_collection
    image_collection = populate_video_images(image_collection, frame_timestamps)

    return RagContext(
        rag_type="video",
        text_collection=text_collection,
        image_collection=image_collection,
        frame_timestamps=frame_timestamps,
    )


def ingest_audio(audio_path: str) -> RagContext:
    """Transcribe an audio file once and embed + store the transcript."""
    print(f"Ingesting audio: {audio_path}")
    transcript = audio_transcribe.transcribe_audio(audio_path)
    document = get_document_loader(file_path=TRANSCRIPT_OUT) if transcript else None

    text_collection, _ = get_collections()
    text_collection = populate_text(document, text_collection) or text_collection

    return RagContext(rag_type="audio", text_collection=text_collection)


def ingest_document(doc_path: str) -> RagContext:
    """Load a document once and embed + store its chunks."""
    print(f"Ingesting document: {doc_path}")
    document = get_document_loader(doc_path)
    text_collection, _ = get_collections()
    text_collection = populate_text(document, text_collection) or text_collection

    return RagContext(rag_type="document", text_collection=text_collection)


def ingest_image(img_path: str) -> RagContext:
    """Encode an image once; no vector store needed (sent inline to the LLM)."""
    print(f"Ingesting image: {img_path}")
    img_data = {"b64": encode_image_to_base64(img_path)}
    return RagContext(rag_type="image", image_data=[img_data])


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

def query(ctx: RagContext, query_str: str) -> str:
    """Answer a question using an already-ingested ``RagContext``."""
    if ctx.rag_type == "video":
        retrieved_texts, retrieved_images = video_retrival(
            query_str, ctx.text_collection, ctx.image_collection,
            ctx.frame_timestamps, top_k=5,
        )
        return ask(query_str, retrieved_texts, retrieved_images, type="video")

    if ctx.rag_type == "audio":
        retrieved_texts = text_retrieval(query_str, ctx.text_collection, top_k=3)
        return ask(query_str, retrieved_texts, [], type="audio")

    if ctx.rag_type == "document":
        retrieved_texts = text_retrieval(query_str, ctx.text_collection, top_k=3)
        return ask(query_str, retrieved_texts, type="document")

    if ctx.rag_type == "image":
        return ask(query_str, [], ctx.image_data, type="image")

    raise ValueError(f"Unknown RAG type: {ctx.rag_type!r}")


# ──────────────────────────────────────────────────────────────────────────────
# BACKWARDS-COMPAT WRAPPERS  — ingest + query in a single call
# ──────────────────────────────────────────────────────────────────────────────

def video_rag_system(video_url: str, query_str: str, rag: str = "video") -> str:
    return query(ingest_video(video_url), query_str)


def audio_rag_system(audio_path: str, query_str: str, rag: str = "audio") -> str:
    return query(ingest_audio(audio_path), query_str)


def image_rag_system(img_url: str, query_str: str, rag: str = "image") -> str:
    return query(ingest_image(img_url), query_str)


def document_rag_system(doc_url: str, query_str: str, rag: str = "document") -> str:
    return query(ingest_document(doc_url), query_str)
