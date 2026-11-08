# Dify RAG → LangGraph (Python)

Converted from your uploaded `RAG.ifpkg` file. This reproduces the **connected graph structure and node intentions**, not the Dify-hosted vector indices or API credentials.

## Graph

```text
START → LLM 2: identify company/year and rewrite query
  ├── query not empty → Knowledge Retrieval → LLM answer → Code formatter → Output
  └── query empty → Output 2: ambiguous_query → END
```

Six executable LangGraph nodes correspond to Dify's two LLM nodes, IF/ELSE routing, retrieval, formatting code, and the early exit. The Dify Start and End nodes become `START` / `END`; decorative notes aren't executable nodes.

## Run (offline demo)

```bash
pip install -r requirements.txt
python main.py "What was ENRG Elements' main focus in 2022?"
```

Offline mode uses a **mock query classifier**, deliberately returns empty retrieval evidence, and produces an explicit no-evidence answer. It demonstrates routing but **does not validate retrieval hit rate**.

## Connect real models

Set the following before running:

```bash
export USE_LIVE_LLM=1
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_BASE_URL='https://api.deepseek.com'
export LLM_MODEL='deepseek-chat'
python main.py "What was the revenue in 2022?"
```

These settings are generic examples, **not identical to the exported Dify Cloud model identifiers** (`deepseek-flash` for query processing and `deepseek-v4-pro` for generation). The sample uses one configurable model for both nodes. Adapt provider-specific model IDs and JSON output support as necessary; the LLM call uses JSON prompting rather than strict server-side schema enforcement.

## Still needed for actual RAG

- `retrieve()` must be connected to a real vector store or Dify Knowledge API. The `.ifpkg` does not contain your three annual reports, indexed chunks, metadata values, or embedding vectors.
- The Dify workflow has **two** knowledge base IDs, although the app description mentions **three companies**. This is not necessarily an error if one KB contains multiple company PDFs.
- Dify retrieval settings: **multiple KB retrieval**, hybrid weight **vector 0.7 / keyword 0.3**, **Top K = 5**, **reranking disabled**, **metadata filtering automatic** (`company`).
- **Existing configuration detail:** Dify `Knowledge Retrieval.query_variable_selector` points to the original `Start.Prompt`, not `LLM 2.structured_output.query`. `retrieve()` intentionally preserves this. To make rephrasing effective, change `retrieval_query = state['Prompt']` to `state['query']`.
- The Dify LLM answer node uses `deepseek-v4-pro`, temp **0.7**, and a 4-field structured response. The Dify LLM 2 uses `deepseek-flash`, temp **0.7**, and a 2-field structured response.
- This is the **actual exported Dify workflow**, not the full Ilya parent-page/LLM-reranking architecture discussed earlier. Those stages are not connected nodes in this export.

The extracted original YAML is included as `app.yaml` for comparison.
