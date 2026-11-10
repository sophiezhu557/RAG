import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent
JSON_DIR = DATA_DIR / "databases_ilya" / "chunked_reports"
FAISS_DIR = DATA_DIR / "databases_qwen" / "vector_dbs"
FAISS_DIR.mkdir(parents=True, exist_ok=True)


def build_databases():
    # 1. Check JSON files
    json_files = list(JSON_DIR.glob("*.json"))

    # 2. Load Qwen only after confirming data exists
    print("Loading Qwen model...", flush=True)
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    print("Device:", model.device)
    print("Qwen model loaded.", flush=True)

    # 3. Build indexes
    for json_path in json_files:
        file_sha1 = json_path.stem
        faiss_path = FAISS_DIR / f"{file_sha1}.faiss"

        with open(json_path, encoding="utf-8") as f:
            document = json.load(f)

        chunks = document["content"]["chunks"]
        chunk_texts = [chunk["text"] for chunk in chunks]
        # chunk_texts = [chunk["text"] for chunk in chunks[:50]]

        print(
            f"Embedding {file_sha1}: {len(chunk_texts)} chunks",
            flush=True
        )

        index = None
        for start in range(0, len(chunk_texts), 20):
            chunk_text = chunk_texts[start: start + 20]
            print(f"Embedding chunks {start} to {start + 19}", flush=True)

            chunk_embeddings = model.encode_document(
                chunk_text,
                batch_size=1,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True
            )

            print("Shape:", chunk_embeddings.shape, flush=True)
            print("Dtype:", chunk_embeddings.dtype, flush=True)

            if not index:
                index = faiss.IndexFlatL2(chunk_embeddings.shape[1])

            index.add(chunk_embeddings)

        faiss.write_index(index, str(faiss_path))
        print("Saved:", faiss_path, flush=True)

if __name__ == "__main__":
    build_databases()
