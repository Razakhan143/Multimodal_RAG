
from app.processors.image_captioner import encode_image_to_base64
from app.embeddings.embedding_service import EmbeddingsGenerator

embed = EmbeddingsGenerator()

# ChromaDB returns squared-L2 distance on normalised vectors. For MiniLM
# (cosine) the usable range is ~0 (identical) to ~2 (opposite). Anything beyond
# this is treated as a weak match that should trigger a Self-RAG query rewrite.
WEAK_DISTANCE = 1.15


def text_retrieval(queryy, text_collection, top_k=3):
    """Retrieve text chunks with their timestamps and distances.

    Returns
    -------
    (texts: list[str], hits: list[dict], best_distance: float)
        hits carry {"text", "timestamp", "distance"} so callers can pick the
        most relevant chunk's timestamp and grade retrieval quality.
    """
    results = text_collection.query(
        query_embeddings=[embed.embed_text(queryy)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs  = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    dists = results["distances"][0] if results.get("distances") else []

    hits = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        dist = dists[i] if i < len(dists) else None
        hits.append({
            "text":      doc,
            "timestamp": meta.get("timestamp"),
            "distance":  dist,
        })

    best = dists[0] if dists else float("inf")
    return docs, hits, best


def image_retrieval(queryy, image_collection, top_k=3):
    # Frame search must use CLIP text embeddings to match the CLIP image space.
    image_results = image_collection.query(
        query_embeddings=[embed.embed_text_clip(queryy)],
        n_results=top_k,
        include=["metadatas"],
    )

    retrieved_images_data = []
    if image_results.get("metadatas"):
        for meta in image_results["metadatas"][0]:
            try:
                img_path = meta["image_uri"]
                ts = meta.get("timestamp", None)
                retrieved_images_data.append({
                    "b64": encode_image_to_base64(img_path),
                    "timestamp": ts,
                })
            except Exception as e:
                print(f"❌ Error loading image: {e}")
                continue
    return retrieved_images_data


def video_retrival(query, text_collection, image_collection, timestamps, top_k=3):
    retrieved_texts, hits, best = text_retrieval(query, text_collection, top_k)
    retrieved_images = image_retrieval(query, image_collection, top_k)
    return retrieved_texts, retrieved_images, hits, best
