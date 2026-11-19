import json
import os
from pathlib import Path
import pandas as pd
from langgraph.graph import StateGraph
from ragas import EvaluationDataset, evaluate
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from main import RAGState
from main import RAGState
from langchain_openai import ChatOpenAI
from main import run_rag
from langgraph.graph import StateGraph, START, END
from retrival import load_databases

EVAL_DIR = Path(__file__).resolve().parent / "evaluation"

if __name__ == "__main__":
    load_databases()

    samples = []
    chunk_rows = []

    with open(EVAL_DIR / "test_questions.json", encoding="utf-8") as f:
        test_questions = json.load(f)

    for question_id, item in enumerate(test_questions, start=1):
        completion = run_rag(item["question"])

        samples.append({
            "user_input": item["question"],
            "rewritten_query": completion.rewritten_query,
            "reference": item["reference"],
            "response": completion.final_answer or "",
            "retrieved_contexts": [
                f"Company: {company}\n"
                f"Page: {chunk.page}\n"
                f"Content: {chunk.text}"
                for company, chunks in completion.selected_chunks.items()
                for chunk in chunks
            ]
        })

        for company, chunks in completion.selected_chunks.items():
            for chunk in chunks:
                chunk_rows.append({
                    "question_id": question_id,
                    "user_input": item["question"],
                    "company": company,
                    "page": chunk.page,
                    "chunk_id": chunk.id,
                    "score": chunk.score,
                    "text": chunk.text
                })

    dataset = EvaluationDataset.from_list(samples)

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            temperature=0,
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            FactualCorrectness(mode="precision"), 
            Faithfulness(),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ],
        llm=evaluator_llm,
    )
# f1：既检查答案有没有编造事实，也检查有没有遗漏 Reference 中的事实。
# precision：主要检查 Response 提出的事实是否与 Reference 一致，不因遗漏额外事实而直接扣分。
# atomicity="low"：尽量减少过度拆分事实陈述。
# coverage="low"：不要求拆分时保留所有细枝末节。

    print(result)

    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    df = result.to_pandas()

    df.insert(0, "question_id", range(1, len(df) + 1))

    # 删除冗长的 retrieved_contexts 列
    df = df.drop(columns=["retrieved_contexts"])

    df.to_csv(results_dir / "ragas_results.csv", index=False)

    # 单独保存 chunks
    chunks_df = pd.DataFrame(chunk_rows)
    chunks_df.to_csv(results_dir / "retrieved_chunks.csv", index=False)