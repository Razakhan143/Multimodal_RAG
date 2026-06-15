import chromadb

client = chromadb.PersistentClient(path="./chromadb_store")

def get_collections():
    text = client.get_or_create_collection("text_collection")
    image = client.get_or_create_collection("image_collection")
    return text, image