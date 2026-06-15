import chromadb

class ChromaDB:

    
    
    def create_text_collections():
        client = chromadb.PersistentClient(path="./chromadb_store")
        # Create collections
        text_collection = client.get_or_create_collection(
            name="text_collection"
        )
        return text_collection
    def create_image_collections():
        client = chromadb.PersistentClient(path="./chromadb_store")
        image_collection = client.get_or_create_collection(
            name="image_collection"
        )
        return image_collection


