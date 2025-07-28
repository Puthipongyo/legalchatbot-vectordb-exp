from qdrant_client import QdrantClient

class Database:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)

    def get_collections(self):
        return self.client.get_collections()

    def recreate_collection(self, collection_name, vectors_config): # Recreate collection 
        return self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )

    def upsert(self, collection_name, points):  # Update + Insert new data
        return self.client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    def search(self, collection_name, query_vector, top_k=5):
        result = self.client.search(
            collection_name = collection_name,
            query_vector = query_vector,
            limit = top_k
        )
        return result
    