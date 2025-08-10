# llm.py
import os, re, torch, pandas as pd
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfFolder
from sentence_transformers import SentenceTransformer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

def l2_normalize(v):
    import math
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def qdrant_retrieve_context(db, collection_name: str, embed_model: SentenceTransformer,
                            query: str, top_k: int = 5, max_ctx_chars: int = 4000) -> str:
    q_vec = embed_model.encode([query])[0]
    q_vec = l2_normalize(q_vec)
    hits = db.search(collection_name=collection_name, query_vector=q_vec, top_k=top_k)

    parts, total = [], 0
    for h in hits:
        t = (h.payload.get("text") or "").strip()
        if not t:
            continue
        if total + len(t) > max_ctx_chars:
            t = t[: max(0, max_ctx_chars - total)]
        if t:
            parts.append(f"[Page {h.payload.get('page','?')} | Chunk {h.payload.get('chunk_index','?')}]\n{t}")
            total += len(t)
        if total >= max_ctx_chars:
            break
    return "\n\n---\n\n".join(parts)

class LLM:
    def __init__(self, model_name: str, db=None, use_4bit: bool = True):
        self.model_name = model_name
        self.db = db
        self.use_4bit = use_4bit
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    def llmSetup(self):
        token = HfFolder.get_token()
        has_cuda = torch.cuda.is_available()
        has_mps = torch.backends.mps.is_available()

        quant_config = None
        dtype = torch.float32
        device_map = "auto"

        if self.use_4bit and has_cuda:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            dtype = torch.float16
        else:
            
            dtype = torch.float16 if has_mps else torch.float32
            device_map = {"": "mps" if has_mps else "cpu"}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map=device_map,
            torch_dtype=dtype,
            token=token
        ).eval()

    def ask_law_questions_to_csv_rag(self,
                                 df: pd.DataFrame,
                                 out_path: str,
                                 collection_name: str,
                                 embed_model: SentenceTransformer,
                                 question_prompt: str = "ข้อความคดีนี้มีมาตราที่เกี่ยวข้องมากที่สุด 5 อันดับแรกมีอะไรบ้าง",
                                 top_k: int = 5,
                                 max_ctx_chars: int = 4000,
                                 max_new_tokens: int = 256) -> str:

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        terminators = [tid for tid in [self.tokenizer.eos_token_id, eot_id] if tid is not None]

        # สร้างคอลัมน์ predicted_law ว่างๆ ถ้ายังไม่มี
        df_2 = df 
        if "predicted_law" not in df.columns:
            df_2["predicted_law"] = ""

        for idx, query_text in enumerate(df["text"].astype(str)):
            context = qdrant_retrieve_context(
                db=self.db,
                collection_name=collection_name,
                embed_model=embed_model,
                query=query_text,
                top_k=top_k,
                max_ctx_chars=max_ctx_chars
            )

            messages = [
                {"role": "system", "content":
                "คุณคือนักกฎหมายผู้เชี่ยวชาญกฎหมายแพ่งและพาณิชย์ "
                "อ่านข้อความคดี แล้วตอบเฉพาะเลขมาตรากฎหมายแพ่งและพาณิชย์ "
                "ที่เกี่ยวข้องมากที่สุด 5 มาตราแรก (ทั้งโดยตรงและโดยอ้อม) "
                "เรียงจากเกี่ยวข้องมากไปน้อย ห้ามเกิน 5 มาตรา และไม่ต้องมีคำว่า 'มาตรา'. "
                "รูปแบบคำตอบ: 1336, 1299, 1520, 1500, 1337"},
                {"role": "user", "content":
                f"บริบทจากฐานความรู้:\n{context or '(ไม่พบบริบท)'}\n\n"
                f"ข้อความคดี/คำถาม:\n{query_text}\n\n"
                f"โปรดตอบ: {question_prompt}"}
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators if terminators else None,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                )
            response = outputs[0][input_ids.shape[-1]:]
            decoded = self.tokenizer.decode(response, skip_special_tokens=True).strip()

            df_2.at[idx, "predicted_law"] = decoded

        # เซฟ CSV
        df_2.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    @staticmethod
    def extract_law_numbers(text: str) -> List[str]:
        return re.findall(r'\d+', text or "")

    @staticmethod
    def evaluate_mrr_at_k(df: pd.DataFrame, k: int = 5) -> Tuple[float, float]:
        df['recall_at_k'] = 0.0
        df['mrr_at_k'] = 0.0
        mrrs, recalls = [], []
        for i in range(df.shape[0]):
            gold = set(LLM.extract_law_numbers(str(df.at[i, 'answers'])))
            pred = LLM.extract_law_numbers(str(df.at[i, 'predicted_law']))[:k]
            # MRR
            rank = None
            for idx, pid in enumerate(pred):
                if pid in gold:
                    rank = idx + 1
                    break
            rr = 1.0 / rank if rank else 0.0
            rec = (len(gold & set(pred)) / len(gold)) if gold else 0.0
            df.at[i, 'mrr_at_k'] = rr
            df.at[i, 'recall_at_k'] = rec
            mrrs.append(rr); recalls.append(rec)
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        return avg_recall, avg_mrr