import chromadb

client = chromadb.PersistentClient(path="./chromadb_store")

def get_collections():
    # text_collection_v2: 384-dim sentence-transformer space (was 512-dim CLIP).
    # A new name is required because ChromaDB fixes the embedding dimension per
    # collection — reusing the old name would raise a dimension-mismatch error.
    text = client.get_or_create_collection("text_collection_v2")
    image = client.get_or_create_collection("image_collection")
    return text, image