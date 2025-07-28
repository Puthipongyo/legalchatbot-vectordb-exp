from VectorDB import Database
from mlflow_config import setup_mlflow
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from pathlib import Path
import fitz
import mlflow
from typing import List


def read_pdf_text(path: str) -> List[str]:
    doc = fitz.open(path)
    chunks = []

    for page in doc:
        text = page.get_text().strip()
        for paragraph in text.split("\n\n"):
            if len(paragraph.strip()) > 50:
                chunks.append(paragraph.strip())

    doc.close()
    return chunks


def create_collection(collection_name: str, distance: Distance, vectors: List[List[float]]):
    db.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=len(vectors[0]),
            distance=distance
        )
    )


def upsert(collection_name: str, vectors: List[List[float]], documents: List[str]):
    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={"text": documents[i], "page": i + 1}
        )
        for i in range(len(documents))
    ]
    db.upsert(collection_name=collection_name, points=points)

def search(collection_name: str, model: SentenceTransformer, query: str, top_k: int):
    queryVector = model.encode([query])[0]
    result = db.search(
        collection_name = collection_name,
        query_vector = queryVector,
        top_k = top_k
    )
    
    print(f"Search : {query}\n")
    for i, result in enumerate(result, start=1):
        score = result.score
        text = result.payload.get("text", "")
        page = result.payload.get("page", "?")
        print(f"{i}. [Score: {score:.4f}] (Page {page})\n{text[:300]}...\n")
    

if __name__ == "__main__":

    
    db = Database()
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    pdf_path = str(Path("Documents/legal_2pages.pdf"))

    # Read and embed
    documents = read_pdf_text(pdf_path)
    vectors = model.encode(documents).tolist()

    # Create collection & insert
    collection_name = "legal_pdf_demo"
    create_collection(collection_name, Distance.COSINE, vectors)
    upsert(collection_name, vectors, documents)

    print("Inserted PDF into Qdrant.")
    print(db.get_collections())
    
    # Search
    sampleQuery = "This site may link you to other sites on the Internet"
    top_k = 5
    search(collection_name, model, sampleQuery, top_k)
    
    # mlFlow
    setup_mlflow("TestLaw_Qdrant")
    with mlflow.start_run():
        mlflow.log_param("model", "MiniLM")
        mlflow.log_param("collection", collection_name)
        mlflow.log_metric(f"recall_at{top_k}", 0.85)

    print("Logged to MLflow")
    
    print("############ End Process ############")