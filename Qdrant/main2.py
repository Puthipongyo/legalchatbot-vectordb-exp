from VectorDB import Database
from qdrant_client import QdrantClient
from mlflow_config import setup_mlflow
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from pathlib import Path
from llm import LLM 
import fitz
import mlflow
import uuid, math
import pandas as pd
from typing import List


def l2_normalize(v):
    import math
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]
class Main:
    @staticmethod
    def read_pdf_text(path: str, min_len: int = 50, join_paragraphs: bool = True, save_path: str = "output.txt") -> List[dict]:
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์: {pdf_path}")

        chunks: List[dict] = []

        with fitz.open(str(pdf_path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text() or ""
                text = text.strip()
                if not text:
                    continue

                if join_paragraphs:
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                else:
                    paragraphs = [text]

                for idx, para in enumerate(paragraphs):
                    if len(para) >= min_len:
                        chunks.append({
                            "page": page_num,
                            "chunk_index": idx,
                            "text": para
                        })
                        
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(f"[Page {c['page']} | Chunk {c['chunk_index']}]\n{c['text']}\n\n")

        print(f"Read PDF Finish")

        return chunks

    def create_collection(self, collection_name: str, distance: Distance, vector_size: int):
            db.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )

    def upsert(self, collection_name: str, vectors: List[List[float]], documents: List[dict]):
        points = []
        for vec, doc in zip(vectors, documents):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "text": doc["text"],
                        "page": doc["page"],
                        "chunk_index": doc["chunk_index"]
                    }
                )
            )
        db.upsert(collection_name=collection_name, points=points)

    
    # def mlflow_log(self, experiment_name: str, embedding_model: str, collection: str, recall_at_k: float, k: int):
    #     setup_mlflow(experiment_name)
    #     with mlflow.start_run():
    #         mlflow.log_param("embedding_model", embedding_model)
    #         mlflow.log_param("collection", collection)
    #         mlflow.log_param("distance", "cosine")
    #         mlflow.log_metric(f"recall_at{k}", recall_at_k)
    #     print("Logged to MLflow")
        

if __name__ == "__main__":

    qdrant_path = "./qdrant_data"  
    qdrant_client = QdrantClient(path=qdrant_path)

    db = Database(client=qdrant_client)

    main = Main()
    embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(embed_model_name)

    pdf_path = str(Path("Documents/ประมวลกฎหมายแพ่งและพาณิชย์.pdf"))
    output_path = str(Path("Documents/Output"))

    documents = main.read_pdf_text(pdf_path, min_len=50, join_paragraphs=True,
                                   save_path=f"{output_path}/output.txt")
    print(f"{len(documents)} chunks")

    texts = [d["text"] for d in documents]
    vectors = [l2_normalize(v) for v in model.encode(texts).tolist()]

    collection_name = "legal_pdf_demo"
    main.create_collection(collection_name, Distance.COSINE,
                           vector_size=len(vectors[0]))
    main.upsert(collection_name, vectors, documents)

    print("Inserted PDF into Qdrant.")
    print(db.get_collections())

    # LLM model
    csv_path = str(Path("csv/data_case_100.csv"))
    csv_output = str(Path("csv/answers_rag.csv"))
    model_llm = "scb10x/llama3.1-typhoon2-70b-instruct"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    llm_model = LLM(model_llm, db=db, use_4bit=True)
    llm_model.llmSetup()

    question_prompt = "ข้อความคดีนี้มีมาตราที่เกี่ยวข้องมากที่สุด 5 อันดับแรกมีอะไรบ้าง"
    out_path = llm_model.ask_law_questions_to_csv_rag(
        df=df,
        out_path=csv_output,
        collection_name=collection_name,
        embed_model=model,
        question_prompt=question_prompt,
        top_k=5,
        max_ctx_chars=4000,
        max_new_tokens=256
    )

    print("############ End Process ############")