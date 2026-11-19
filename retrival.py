from hashlib import sha1
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from schemas import DocumentChunk, RetrievedChunk
from query_rewrite import rewrite_query

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

        if document["metainfo"]["sha1"] != file_sha1:
                    raise ValueError(f"SHA1 mismatch: {json_path.name}")
        
        index = faiss.read_index(str(faiss_path))
        # chunks = document["content"]["chunks"]
        chunks = [DocumentChunk.model_validate(chunk) 
                  for chunk in document["content"]["chunks"]]

        # check
        if index.ntotal != len(chunks):
                    raise ValueError(
                        f"Index mismatch for {file_sha1}: "
                        f"{index.ntotal} vectors, {len(chunks)} chunks"
                    )
        
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
            chunk = chunks[int(idx)]
            selected_chunk = RetrievedChunk(
                            sha1 = file_sha1,
                            page = chunk.page,
                            text = chunk.text,
                            id = chunk.id,
                            score = float(score)
                        )
            
            selected_chunks[company_name].append(selected_chunk)

    return selected_chunks       


if __name__ == "__main__":
    from company_router import (
        load_company_mapping,
        identify_companies,
    )

    databases = load_databases()

    # Load once
    company_mapping = load_company_mapping()

    # Test query
    queries = ["What was Compagnie Financière Tradition's reported consolidated revenue in 2022?",
             "reported consolidated revenue in 2022."]
    
    company_list = identify_companies(
                queries[0],
                company_mapping
            )
    
    for query in queries:
        context = retrieve(
            query=query,
            company_query_list=company_list,
            top_k=5,
            databases=databases
        )

        for company_name, chunks in context.items():
            print(f"Company: {company_name}")

            for chunk in chunks:
                print(
                    f"Page: {chunk.page}, "
                    f"Score: {chunk.score:.4f}"
                )
                print(chunk.text[:300])
