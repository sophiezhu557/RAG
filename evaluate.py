import json
import os
from pathlib import Path
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from main import databases, run_rag


EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
# QUESTION_FILE = EVAL_DIR / "test_questions.json"
QUESTION_FILE = EVAL_DIR / "questions_with_ground_truth.json"
results_dir = EVAL_DIR / "results"
results_dir.mkdir(exist_ok=True)
results_path = results_dir / "ragas_results_with_ground_truth.csv"
retrieved_chunks_path = results_dir / "retrieved_chunks.csv"

def add_tokens(first, second):
    if first is None or second is None:
        return None
    return first + second

if __name__ == "__main__":
    questions = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    runs = []
    chunk_rows = []
    samples = []
    scored_ids = []

    for question_id, item in enumerate(questions, start=1):
        question = item["question"]
        expected_pdf = item["pdf"]
        expected_sha1 = Path(expected_pdf).stem

        row = {
            "question_id": question_id,
            "user_input": question,
            "reference": item["reference"],
            "reference_source": item.get("reference_source"),
            "expected_pdf": expected_pdf,
            "status": "MissingIndex",
            "message": None,
            "route_matches_expected": None,
            "matched_companies": None,
            "response": None,
            "rewritten_query": None,
            "rewrite_total_tokens": 0,
            "generation_total_tokens": 0,
            "rag_input_tokens": 0,
            "rag_output_tokens": 0,
            "rag_total_tokens": 0,
        }
        runs.append(row)

        if expected_sha1 not in databases:
            row["message"] = f"Expected report is not indexed: {expected_pdf}"
            print(f"[{question_id}/{len(questions)}] {row['message']}", flush=True)
            continue

        completion = run_rag(question)
        rewrite_usage = completion.rewrite_usage
        generation_usage = completion.generation_usage
        matched_sha1s = {entry["sha1"] for entry in completion.company_query_list}

        row.update(
            status=completion.status,
            message=completion.message,
            route_matches_expected=matched_sha1s == {expected_sha1},
            matched_companies="; ".join(
                entry["company_name"] for entry in completion.company_query_list
            ),
            response=completion.final_answer,
            rewritten_query=completion.rewritten_query,
            rewrite_total_tokens=rewrite_usage.total_tokens,
            generation_total_tokens=generation_usage.total_tokens,
            
            rag_input_tokens=add_tokens(
                rewrite_usage.input_tokens, generation_usage.input_tokens
            ),
            rag_output_tokens=add_tokens(
                rewrite_usage.output_tokens, generation_usage.output_tokens
            ),
            rag_total_tokens=add_tokens(
                rewrite_usage.total_tokens, generation_usage.total_tokens
            ),
        )
        print(f"[{question_id}/{len(questions)}] {completion.status}", flush=True)

        contexts = []
        for company, chunks in completion.selected_chunks.items():
            for chunk in chunks:
                contexts.append(
                    f"Company: {company}\nPage: {chunk.page}\nContent: {chunk.text}"
                )
                chunk_rows.append({
                    "question_id": question_id,
                    "user_input": question,
                    "company": company,
                    "page": chunk.page,
                    "chunk_id": chunk.id,
                    "score": chunk.score,
                    "text": chunk.text,
                })

        if completion.final_answer and contexts:
            scored_ids.append(question_id)
            samples.append({
                "user_input": question,
                "reference": item["reference"],
                "response": completion.final_answer,
                "retrieved_contexts": contexts,
            })

    metrics = [
        FactualCorrectness(mode="f1"),
        Faithfulness(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]

    results = pd.DataFrame(runs).set_index("question_id")
    for metric in metrics:
        results[metric.name] = float("nan")
    results.to_csv(results_path)

    chunk_columns = [
        "question_id", "user_input", "company", "page", "chunk_id", "score", "text"
    ]
    pd.DataFrame(chunk_rows, columns=chunk_columns).to_csv(
        retrieved_chunks_path, index=False
    )

    print(f"Scoring {len(samples)} of {len(questions)} questions.")
    if samples:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
                api_key=os.environ["DEEPSEEK_API_KEY"],
                temperature=0,
            )
        )
        result = evaluate(
            dataset=EvaluationDataset.from_list(samples),
            metrics=metrics,
            llm=evaluator_llm,
        )

        scores = result.to_pandas().rename(columns={
            "factual_correctness(mode=f1)": "factual_correctness",
        })
        for metric in metrics:
            results.loc[scored_ids, metric.name] = scores[metric.name].to_numpy()
        results.to_csv(results_path)
        print(result)