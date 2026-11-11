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
from main import RAGState, retrieval_node, generation_node
from langgraph.graph import StateGraph, START, END
from retrival import load_databases

EVAL_DIR = Path(__file__).resolve().parent / "evaluation"

def run_rag(question):
    """
    返回:
        response: str
        retrieved_contexts: list[str]
    """
    builder = StateGraph(RAGState)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("generation", generation_node)
    builder.add_edge(START, "retrieval")
    builder.add_edge("retrieval", "generation")
    builder.add_edge("generation", END)
    app = builder.compile()

    result = app.invoke({
            "question": question
        })
    
    return {
        "response": result["final_answer"], 
        "company_query_list": result["company_query_list"],
        "retrieved_contexts":
        [
            f"company: {company}:,Page: {chunk['page']}, 'chunks': {chunk['text']}"
            for company, chunks in result["selected_chunks"].items()
            for chunk in chunks
            ]
    }
            

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
            "response": completion["response"],
            "retrieved_contexts": completion["retrieved_contexts"],
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
    # evaluator_llm = LangchainLLMWrapper(
    #     ChatOpenAI(
    #         model="gpt-4.1-mini",
    #         temperature=0,
    #     )
    # )

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
    # with open(EVAL_DIR / "results" / "evaluation_results.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)
    
    df = result.to_pandas()
    df.to_csv(EVAL_DIR / "results" / "ragas_results.csv", index=False)
