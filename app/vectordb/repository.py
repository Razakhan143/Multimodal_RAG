
import os
from app.embeddings.embedding_service import EmbeddingsGenerator
from app.processors.doc_process import text_splitter
embeddings_generator = EmbeddingsGenerator()
embed_text = embeddings_generator.embed_text
embed_texts = embeddings_generator.embed_texts
embed_image = embeddings_generator.embed_image


def _clear_collection(collection):
    """Remove all existing entries from a persistent collection.

    Collections persist across queries on disk, so reusing fixed ids (``text_0``…)
    with ``add`` raises a duplicate-id error on the second query. Clearing first
    keeps each query scoped to the freshly uploaded file.
    """
    try:
        existing = collection.get()
        ids = existing.get("ids", []) if existing else []
        if ids:
            collection.delete(ids=ids)
    except Exception as e:
        print(f"⚠️ Could not clear collection: {e}")


def _chunks_from_segments(segments, target_chars=600):
    """Group Whisper segments into ~target_chars chunks, keeping a start time.

    Returns a list of {"text": str, "start": float} so every text chunk carries
    the moment it was spoken. This is what makes the video/audio timestamp precise
    — the answer's chunk knows exactly where in the media it came from.
    """
    chunks = []
    buf, buf_start = [], None
    for seg in segments:
        if buf_start is None:
            buf_start = float(seg.get("start", 0.0) or 0.0)
        buf.append(seg.get("text", "").strip())
        if sum(len(t) for t in buf) >= target_chars:
            chunks.append({"text": " ".join(buf).strip(), "start": buf_start})
            buf, buf_start = [], None
    if buf:
        chunks.append({"text": " ".join(buf).strip(), "start": buf_start or 0.0})
    return [c for c in chunks if c["text"]]


def populate_text(transcript_text, text_collection, segments=None):
    """Embed text into the text collection.

    If ``segments`` (Whisper timestamped segments) are given, chunks are built
    from them and each carries a ``timestamp`` in its metadata. Otherwise the
    plain transcript/document text is split by the recursive splitter.
    """
    # Path A: timestamp-aware chunking from Whisper segments (audio / video).
    if segments:
        seg_chunks = _chunks_from_segments(segments)
        if seg_chunks:
            _clear_collection(text_collection)
            documents = [c["text"] for c in seg_chunks]
            metadatas = [{"timestamp": float(c["start"])} for c in seg_chunks]
            ids = [f"text_{i}" for i in range(len(seg_chunks))]
            text_collection.upsert(
                ids=ids,
                embeddings=embed_texts(documents),
                documents=documents,
                metadatas=metadatas,
            )
            print(f"✅ Indexed {len(seg_chunks)} timestamped transcript chunks.")
            return text_collection
        # Segments present but empty → fall through to plain-text handling.

    # Path B: plain document/transcript text (no timestamps).
    if not transcript_text or not str(transcript_text).strip():
        return text_collection

    _clear_collection(text_collection)
    text_chunks = text_splitter(transcript_text)

    if text_chunks:
        print(f"📄 Splitting into {len(text_chunks)} text chunks.")
        documents = [getattr(c, "page_content", str(c)) for c in text_chunks]
        ids = [f"text_{i}" for i in range(len(documents))]
        text_collection.upsert(
            ids=ids,
            embeddings=embed_texts(documents),
            documents=documents,
        )
        print(f"✅ Indexed {len(documents)} text segments.")
    else:
        # Fallback: index the whole thing as one document.
        try:
            if isinstance(transcript_text, (list, tuple)):
                contents = [getattr(d, "page_content", str(d)) for d in transcript_text]
                full_text = "\n\n".join(contents)
            else:
                full_text = getattr(transcript_text, "page_content", str(transcript_text))
            text_collection.upsert(
                ids=["text_full_0"],
                embeddings=[embed_text(full_text)],
                documents=[full_text],
            )
            print("✅ Indexed full transcript as a single segment.")
        except Exception as e:
            print(f"⚠️ Failed to index transcript: {e}")
    return text_collection


def populate_video_images(image_collection, frame_timestamps=None, images_folder="app/ingestion/video_ingess"):
    # 2. IMAGE INGESTION (ChromaDB) — CLIP space, unchanged.
    image_embeddings = []
    image_ids = []
    metadatas = []

    if not frame_timestamps:
        return image_collection

    _clear_collection(image_collection)

    for filename, ts in frame_timestamps.items():
        path = os.path.join(images_folder, filename)
        if os.path.exists(path):
            image_ids.append(f"img_{filename}")
            image_embeddings.append(embed_image(path))
            metadatas.append({"image_uri": path, "timestamp": float(ts)})

    if image_ids:
        image_collection.upsert(
            ids=image_ids,
            embeddings=image_embeddings,
            metadatas=metadatas,
        )
        print(f"✅ Indexed {len(image_ids)} video frames with timestamps.")
    else:
        print(f"⚠️ No frame images found to index in '{images_folder}'.")
    return image_collection
