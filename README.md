## Run (offline demo)

```bash
uv run python main.py "Your question or blank(default question is about TSX_Y' revenue in 202)"
```

Offline mode uses a **mock query classifier**, deliberately returns empty retrieval evidence, and produces an explicit no-evidence answer. It demonstrates routing but **does not validate retrieval hit rate**.

## Connect real models

Set the following before running:

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_BASE_URL='https://api.deepseek.com'
export LLM_MODEL='deepseek-chat'
python main.py "What was the revenue in 2022?"
```

## Still needed for actual RAG
- Build a knowledge base by myself.
- This is the **actual exported Dify workflow**, not the full Ilya parent-page/LLM-reranking architecture discussed earlier. Those stages are not connected nodes in this export.