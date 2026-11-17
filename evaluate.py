import json
import os
from pathlib import Path
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

    with open(EVAL_DIR / "test_questions.json", encoding="utf-8") as f:
        test_questions = json.load(f)

    for item in test_questions:
        completion = run_rag(item["question"])

        samples.append({
            "user_input": item["question"],
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

    (EVAL_DIR / "results").mkdir(exist_ok=True)
    df = result.to_pandas()
    df.to_csv(EVAL_DIR / "results" / "ragas_results.csv", index=False)
