from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


BASE_MODEL_PATH = resolve_repo_path(
    os.getenv("BASE_MODEL_PATH", str(BASE_DIR / "models" / "base_llm"))
)
ADAPTER_PATH = resolve_repo_path(
    os.getenv("LORA_ADAPTER_PATH", str(BASE_DIR / "outputs" / "lora_adapter"))
)
COUNSEL_FULL_PATH = resolve_repo_path(
    os.getenv("COUNSEL_FULL_PATH", str(BASE_DIR / "data" / "gold" / "counsel_full.parquet"))
)


# =========================
# RAG / EMBEDDING CONFIG
# =========================
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Eğer kendi collection adın farklıysa bunu değiştir
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "counsel_rag")

TOP_K = int(os.getenv("TOP_K", "4"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "120"))


# =========================
# EMOTION / TONE
# =========================
EMOTION_KEYWORDS = {
    "anxiety": [
        "anxious", "anxiety", "overwhelmed", "panic", "worried",
        "worry", "nervous", "stress", "stressed", "fear", "afraid"
    ],
    "distress": [
        "sad", "hopeless", "empty", "down", "lonely", "crying",
        "worthless", "depressed", "hurt", "upset"
    ],
    "anger": [
        "angry", "furious", "annoyed", "mad", "resentful", "irritated"
    ],
    "self-esteem": [
        "not good enough", "worthless", "insecure", "not enough",
        "self-doubt", "hate myself"
    ],
}

EMOTION_TO_TONE = {
    "anxiety": "calm, reassuring, grounded",
    "distress": "warm, validating, gentle",
    "anger": "calm, respectful, de-escalating",
    "self-esteem": "affirming, non-judgmental, supportive",
    "mixed": "warm, validating, supportive",
}

TOPIC_TO_TONE = {
    "anxiety": "calm, reassuring, grounded",
    "depression": "warm, validating, gentle",
    "self-esteem": "affirming, non-judgmental, supportive",
    "workplace-relationships": "respectful, balanced, supportive",
    "stress": "steady, practical, supportive",
    "behavioral-change": "encouraging, practical, supportive",
}

TARGET_TOPICS = {
    "anxiety",
    "depression",
    "self-esteem",
    "workplace-relationships",
    "stress",
    "behavioral-change",
}


# =========================
# OUTPUT CLEANING
# =========================
STOP_PATTERNS = [
    r"\nHuman:",
    r"\nUser:",
    r"\nContext:",
    r"\nTitle:",
    r"\nTopic:",
    r"\nQuestion:",
    r"\nAnswer:",
    r"\n###",
    r"Human:",
    r"User:",
    r"Context:",
    r"Title:",
    r"Topic:",
    r"Question:",
    r"Answer:",
    r"###",
    r"\[Document\]",
    r"\n\[Document\]",
]


def clean_model_output(decoded_text: str) -> str:
    text = decoded_text.strip()

    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1].strip()

    cut_positions = []
    for pattern in STOP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            cut_positions.append(match.start())

    if cut_positions:
        text = text[:min(cut_positions)].strip()

    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# EMOTION / TONE
# =========================
def predict_emotion(user_query: str) -> str:
    q = user_query.lower()

    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in q:
                return emotion

    return "mixed"


def choose_tone(predicted_emotion: str, retrieved_topics: List[str]) -> str:
    if predicted_emotion in EMOTION_TO_TONE:
        return EMOTION_TO_TONE[predicted_emotion]

    for topic in retrieved_topics:
        if topic in TOPIC_TO_TONE:
            return TOPIC_TO_TONE[topic]

    return "warm, validating, supportive"


# =========================
# PROMPT BUILDER
# =========================
def build_prompt(
    user_query: str,
    context_text: str,
    predicted_emotion: str,
    tone: str,
    topics_used: List[str],
    document_count: int,
) -> str:
    return (
        "Use only the context below to answer the user.\n"
        "Be empathetic, grounded, concise, and practical.\n"
        "Do not continue with another example.\n"
        "Do not write titles, labels, or extra sections.\n"
        "Do not claim to be a therapist or give diagnosis.\n\n"
        "Answer as a direct supportive assistant, not as a forum commenter.\n"
        "Do not say you are not a professional, do not say you have heard things, and do not use casual openings like 'yeah' or 'look'.\n\n"
        f"Retrieved Context Count: {document_count}\n"
        f"Retrieved Topics: {topics_used}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User: {user_query}\n"
        f"Emotion: {predicted_emotion}\n"
        f"Tone: {tone}\n\n"
        "Answer:"
    )


# =========================
# MODEL LOADER
# =========================
class LoraInferenceEngine:
    def __init__(self, base_model_path: Path, adapter_path: Path):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None

    def load(self):
        logger.info("Tokenizer yükleniyor...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            local_files_only=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Base model yükleniyor...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            local_files_only=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
            device_map="auto" if torch.cuda.is_available() else None,
        )

        logger.info("LoRA adapter yükleniyor...")
        self.model = PeftModel.from_pretrained(
            self.model,
            self.adapter_path,
            is_trainable=False,
        )
        self.model.eval()

        logger.info("LLM hazır.")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_model_output(decoded)


# =========================
# RAG RETRIEVER
# =========================
class HybridRetriever:
    def __init__(
        self,
        collection_name: str,
        parquet_path: Path,
        embed_model_name: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        self.collection_name = collection_name
        self.parquet_path = parquet_path
        self.embed_model_name = embed_model_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.local_files_only = HF_LOCAL_FILES_ONLY

        self.embedder = None
        self.qdrant_client = None
        self.local_df = None
        self.local_embeddings = None
        self.qdrant_available = False

    def load(self):
        logger.info("Embedder yükleniyor...")
        self.embedder = SentenceTransformer(
            self.embed_model_name,
            local_files_only=self.local_files_only,
        )

        logger.info("Local counsel parquet yükleniyor...")
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Counsel parquet bulunamadı: {self.parquet_path}")

        self.local_df = pd.read_parquet(self.parquet_path)

        required_cols = {"rag_document", "topic", "quality_score"}
        missing = required_cols - set(self.local_df.columns)
        if missing:
            raise ValueError(f"Counsel parquet içinde eksik kolonlar var: {missing}")

        logger.info("Local semantic retrieval için embedding hazırlanıyor...")
        docs = self.local_df["rag_document"].fillna("").tolist()
        self.local_embeddings = self.embedder.encode(
            docs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        logger.info("Qdrant bağlantısı deneniyor...")
        try:
            self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            collections = self.qdrant_client.get_collections()
            names = [c.name for c in collections.collections]
            self.qdrant_available = self.collection_name in names
            logger.info("Qdrant available: %s", self.qdrant_available)
        except Exception as e:
            logger.warning("Qdrant bağlantısı yok, local retrieval kullanılacak. Detay: %s", e)
            self.qdrant_available = False

    def _infer_topic_hints(self, query: str) -> List[str]:
        q = query.lower()
        hints = []

        if any(x in q for x in ["work", "manager", "office", "job", "coworker", "boss"]):
            hints.append("workplace-relationships")
        if any(x in q for x in ["anxiety", "anxious", "panic", "worried", "overwhelmed"]):
            hints.append("anxiety")
        if any(x in q for x in ["stress", "stressed", "pressure"]):
            hints.append("stress")
        if any(x in q for x in ["not good enough", "self-doubt", "insecure", "worthless"]):
            hints.append("self-esteem")
        if any(x in q for x in ["depressed", "hopeless", "empty", "sad"]):
            hints.append("depression")

        return list(dict.fromkeys(hints))

    def _build_context_result(self, rows: List[Dict]) -> Dict:
        context_chunks = []
        topics_used = []

        for item in rows:
            topic = item.get("topic", "unknown")
            score = item.get("score", 0.0)
            quality = item.get("quality_score", 0.0)
            doc = item.get("rag_document", "")

            topics_used.append(topic)

            context_chunks.append(
                f"[Document]\n"
                f"Topic: {topic}\n"
                f"Score: {score:.4f}\n"
                f"Quality: {quality:.4f}\n"
                f"{doc}"
            )

        return {
            "context_text": "\n\n".join(context_chunks),
            "document_count": len(rows),
            "topics_used": list(dict.fromkeys(topics_used)),
            "documents": rows,
        }

    def _retrieve_from_qdrant(self, query: str, top_k: int) -> Dict:
        query_vec = self.embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        topic_hints = self._infer_topic_hints(query)

        query_filter = None
        if topic_hints:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="topic",
                        match=MatchAny(any=topic_hints)
                    )
                ]
            )

        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k,
            query_filter=query_filter,
        )

        rows = []
        for r in results:
            payload = r.payload or {}
            rows.append({
                "topic": payload.get("topic", "unknown"),
                "quality_score": float(payload.get("quality_score", 0.0)),
                "rag_document": payload.get("rag_document", ""),
                "score": float(r.score),
            })

        return self._build_context_result(rows)

    def _retrieve_from_local(self, query: str, top_k: int) -> Dict:
        query_vec = self.embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        sims = np.dot(self.local_embeddings, query_vec)

        temp_df = self.local_df.copy()
        temp_df["semantic_score"] = sims

        topic_hints = self._infer_topic_hints(query)
        if topic_hints:
            filtered = temp_df[temp_df["topic"].isin(topic_hints)].copy()
            if not filtered.empty:
                temp_df = filtered

        temp_df["final_score"] = (
            0.80 * temp_df["semantic_score"] +
            0.20 * temp_df["quality_score"]
        )

        temp_df = temp_df.sort_values("final_score", ascending=False).head(top_k)

        rows = []
        for _, row in temp_df.iterrows():
            rows.append({
                "topic": row.get("topic", "unknown"),
                "quality_score": float(row.get("quality_score", 0.0)),
                "rag_document": row.get("rag_document", ""),
                "score": float(row.get("final_score", 0.0)),
            })

        return self._build_context_result(rows)

    def retrieve(self, query: str, top_k: int = 4) -> Dict:
        if self.qdrant_available:
            try:
                logger.info("Qdrant retrieval kullanılıyor...")
                return self._retrieve_from_qdrant(query, top_k)
            except Exception as e:
                logger.warning("Qdrant retrieval başarısız, local fallback kullanılacak. Detay: %s", e)

        logger.info("Local semantic retrieval kullanılıyor...")
        return self._retrieve_from_local(query, top_k)


# =========================
# FULL PIPELINE
# =========================
def run_pipeline(user_query: str, retriever: HybridRetriever, llm: LoraInferenceEngine) -> Dict:
    predicted_emotion = predict_emotion(user_query)

    context_result = retriever.retrieve(user_query, top_k=TOP_K)
    topics_used = context_result.get("topics_used", [])
    context_text = context_result.get("context_text", "")
    document_count = context_result.get("document_count", 0)

    tone = choose_tone(predicted_emotion, topics_used)

    prompt = build_prompt(
        user_query=user_query,
        context_text=context_text,
        predicted_emotion=predicted_emotion,
        tone=tone,
        topics_used=topics_used,
        document_count=document_count,
    )

    final_answer = llm.generate(prompt)

    return {
        "user_query": user_query,
        "predicted_emotion": predicted_emotion,
        "tone": tone,
        "document_count": document_count,
        "topics_used": topics_used,
        "context_text": context_text,
        "final_answer": final_answer,
    }


# =========================
# MAIN
# =========================
def main():
    logger.info("Full RAG + LoRA inference started...")
    logger.info("Base model path: %s", BASE_MODEL_PATH)
    logger.info("LoRA adapter path: %s", ADAPTER_PATH)

    llm = LoraInferenceEngine(
        base_model_path=BASE_MODEL_PATH,
        adapter_path=ADAPTER_PATH,
    )
    llm.load()

    retriever = HybridRetriever(
        collection_name=QDRANT_COLLECTION_NAME,
        parquet_path=COUNSEL_FULL_PATH,
        embed_model_name=EMBED_MODEL_NAME,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
    )
    retriever.load()

    user_query = os.getenv(
        "TEST_USER_QUERY",
        "I feel overwhelmed at work and I keep thinking that I'm not good enough. "
        "I want to handle this in a healthier way.",
    )

    result = run_pipeline(user_query=user_query, retriever=retriever, llm=llm)

    print("\n" + "=" * 100)
    print("USER QUERY")
    print("=" * 100)
    print(result["user_query"])

    print("\n" + "=" * 100)
    print("PREDICTED EMOTION")
    print("=" * 100)
    print(result["predicted_emotion"])

    print("\n" + "=" * 100)
    print("SELECTED TONE")
    print("=" * 100)
    print(result["tone"])

    print("\n" + "=" * 100)
    print("RETRIEVED TOPICS")
    print("=" * 100)
    print(result["topics_used"])

    print("\n" + "=" * 100)
    print("RETRIEVED DOCUMENT COUNT")
    print("=" * 100)
    print(result["document_count"])

    print("\n" + "=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)
    print(result["final_answer"])


if __name__ == "__main__":
    main()
