
from app.processors.image_captioner import encode_image_to_base64
from app.embeddings.embedding_service import EmbeddingsGenerator

embed=EmbeddingsGenerator()

def text_retrieval(queryy,text_collection,top_k=3):
    """
    ChromaDB multimodal retrieval:
    - text_collection
    - image_collection
    """

    # -------------------------
    # 1. TEXT SEARCH (ChromaDB)
    # -------------------------
    text_results = text_collection.query(
        query_embeddings=[embed.embed_text(queryy)],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    retrieved_texts = text_results["documents"][0] if text_results["documents"] else []

    return retrieved_texts
def image_retrieval(queryy,image_collection ,top_k=3):
        # -------------------------
    # 2. IMAGE SEARCH (ChromaDB)
    # -------------------------
    image_results = image_collection.query(
        query_embeddings=[embed.embed_text(queryy)],  # using same embedding space (CLIP)
        n_results=top_k,
        include=["metadatas"]
    )

    retrieved_images_data = []

    if image_results["metadatas"]:
        for meta in image_results["metadatas"][0]:
            try:
                img_path = meta["image_uri"]
                ts = meta.get("timestamp", None)

                b64_str = encode_image_to_base64(img_path)

                retrieved_images_data.append({
                    "b64": b64_str,
                    "timestamp": ts
                })

            except Exception as e:
                print(f"❌ Error loading image: {e}")
                # Continue collecting other images even if one fails
                continue
    # print(f"Retrieved Images Metadata: {retrieved_images_data}")
    return retrieved_images_data
def video_retrival(query,text_collection, image_collection, timestamps, top_k=3):
    retrieved_texts=text_retrieval(query, text_collection, top_k)
    
    retrieved_images=image_retrieval(query, image_collection, top_k)
    
    return retrieved_texts, retrieved_images