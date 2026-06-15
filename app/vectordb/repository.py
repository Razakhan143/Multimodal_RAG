
import re
import os
from app.embeddings.embedding_service import EmbeddingsGenerator
from app.processors.doc_process import text_splitter
embeddings_generator = EmbeddingsGenerator()
embed_text = embeddings_generator.embed_text
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


def populate_text(transcript_text,text_collection):
    # 1. TEXT INGESTION (ChromaDB)

    # Guard against None or empty transcripts
    if not transcript_text or not str(transcript_text).strip():
        # Nothing to ingest; return the existing collection unchanged
        return text_collection

    _clear_collection(text_collection)
    text_chunks = text_splitter(transcript_text)

    if text_chunks:
        print(f"📄 Splitting transcript into {len(text_chunks)} text chunks.")

        text_embeddings = []
        ids = []
        documents = []

        for i, chunk in enumerate(text_chunks):
            # LangChain splitters return Document objects; embed and store their page_content
            content = getattr(chunk, "page_content", str(chunk))
            ids.append(f"text_{i}")
            documents.append(content)
            text_embeddings.append(embed_text(content))

        text_collection.upsert(
            ids=ids,
            embeddings=text_embeddings,
            documents=documents
        )

        print(f"✅ Indexed {len(text_chunks)} text segments.")
    elif str(transcript_text).strip():
        # Fallback: index the entire transcript as a single document
        try:
            # Handle both raw strings and Document lists gracefully
            if isinstance(transcript_text, (list, tuple)):
                try:
                    # Join page_content if a list of Document objects was provided
                    contents = [getattr(d, "page_content", str(d)) for d in transcript_text]
                    full_text = "\n\n".join(contents)
                except Exception:
                    full_text = str(transcript_text)
            else:
                full_text = getattr(transcript_text, "page_content", str(transcript_text))

            text_embedding = embed_text(full_text)
            text_collection.upsert(
                ids=["text_full_0"],
                embeddings=[text_embedding],
                documents=[full_text]
            )
            print("✅ Indexed full transcript as a single segment.")
        except Exception as e:
            print(f"⚠️ Failed to index transcript: {e}")
    # Always return the collection (even if there were no chunks)
    return text_collection

def populate_video_images(image_collection, frame_timestamps=None, images_folder="app/ingestion/video_ingess"):
    # 2. IMAGE INGESTION (ChromaDB)
    image_embeddings = []
    image_ids = []
    metadatas = []

    # Guard against missing timestamps input
    if not frame_timestamps:
        return image_collection

    _clear_collection(image_collection)

    for filename, ts in frame_timestamps.items():
        path = os.path.join(images_folder, filename)

        if os.path.exists(path):
            image_ids.append(f"img_{filename}")
            image_embeddings.append(embed_image(path))

            metadatas.append({
                "image_uri": path,
                "timestamp": float(ts)
            })

    if image_ids:
        image_collection.upsert(
            ids=image_ids,
            embeddings=image_embeddings,
            metadatas=metadatas
        )

        print(f"✅ Indexed {len(image_ids)} video frames with timestamps.")
    else:
        print(f"⚠️ No frame images found to index in '{images_folder}'.")
    return image_collection