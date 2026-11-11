from hashlib import sha1
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

DATA_DIR = Path(__file__).resolve().parent / "databases_qwen"
JSON_DIR = DATA_DIR / "chunked_reports"
FAISS_DIR = DATA_DIR / "vector_dbs"


def load_databases():
    databases = {}

    for json_path in JSON_DIR.glob("*.json"):
        file_sha1 = json_path.stem
        faiss_path = FAISS_DIR / f"{file_sha1}.faiss"

        with open(json_path, encoding="utf-8") as f:
            document = json.load(f)

        index = faiss.read_index(str(faiss_path))
        chunks = document["content"]["chunks"]

        databases[file_sha1] = {
            "index": index,
            "chunks": chunks
        }
    return databases


# 2. Generate query embedding
def embed_query(query):
    return model.encode_query(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )


# 3. Vector search and parent page retrieval
def retrieve(
    query,
    company_query_list,
    top_k=5,
    databases=None
    ):
    if databases is None:
        databases = load_databases()

    query_vector = embed_query(query)
    selected_chunks = {}

    for company in company_query_list:
        company_name = company["company_name"]
        file_sha1 = company["sha1"]

        index = databases[file_sha1]["index"]
        chunks = databases[file_sha1]["chunks"]

        scores, indices = index.search(query_vector, k=top_k)

        selected_chunks[company_name] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            selected_chunks[company_name].append({
                "page": chunks[int(idx)]["page"],
                "text": chunks[int(idx)]["text"],
                "id": chunks[int(idx)]["id"],
                "score": float(score)
            })

    return selected_chunks       


if __name__ == "__main__":
    from company_router import (
        load_company_mapping,
        identify_companies,
    )

    # Load once
    company_mapping = load_company_mapping()

    # Test query
    query = "What was ENRG Elements' revenue in 2022?"

    # Identify company
    company_list = identify_companies(
        query,
        company_mapping
    )

    context = retrieve(
        query=query,
        company_query_list=company_list,
        top_k=5,
        databases=load_databases()
    )

    for chunks in context:
        print(f"Company: {chunks['company_name']}")
        print("-" * 50)
