
import os
import sys
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from company_router import load_company_mapping, identify_companies
from retrival import load_databases, retrieve
from schemas import RAGResponse
from prompts import SYSTEM_prompt, build_user_question
from query_rewrite import rewrite_query


load_dotenv()

llm = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

company_mapping = load_company_mapping()
databases = load_databases()


# 1. State
class RAGState(TypedDict, total=False):
    question: str
    rewritten_query: str
    company_query_list: list[dict]
    selected_chunks: dict
    final_answer: str
    status: str
    message: str | None


def query_rewritting_node(state: RAGState):
    question = state["question"]
    company_query_list = identify_companies(
            question,
            company_mapping
        )
    
    rewritten_query = rewrite_query(question, company_query_list, llm)
    
    return {
        "company_query_list": company_query_list,
        "rewritten_query": rewritten_query
    }


# 2. Retrieval Node
def retrieval_node(state: RAGState):
    if state["company_query_list"]: # if companies are found, proceed to retrieve relevant chunks
        selected_chunks = retrieve(
            query=state["rewritten_query"],
            company_query_list=state["company_query_list"],
            databases=databases,
            top_k=5
        )
    else: # if no companies are found, return an empty dictionary for selected_chunks
        selected_chunks = {}

    return {
        "selected_chunks": selected_chunks
    }


# 3. Generation Node
def generation_node(state: RAGState):
    question = state["question"]
    selected_chunks = state["selected_chunks"]

    if not state["company_query_list"]:  # Check if company_query_list is empty
        return {
            "final_answer": None,
            "status": "CompanyNotFound",
            "message": "No matching company was identified."
        }

    if not any(selected_chunks.values()):  # Check if all values in selected_chunks are empty
        return {
            "final_answer": None,
            "status": "NoRelevantChunks",
            "message": "No relevant chunks were found for the identified companies."
        }


    # Call DeepSeek
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_prompt
            },
            {
                "role": "user",
                "content": build_user_question(question, selected_chunks)
            }
        ],
        temperature=0
    )

    final_answer = response.choices[0].message.content
    if (final_answer is None 
        or final_answer.strip().lower() in {"none", "not found", ""}
        ):
        return {
            "status": "NoAnswerFound",
            "message": "Answer not found in retrieved context.",
            "final_answer": None
        }

    return {
        "status": "Success",
        "message": None,
        "final_answer": final_answer
    }


# 4. Build LangGraph
# ——————————————————————————————————————————————————————————————
builder = StateGraph(RAGState)

builder.add_node("query_rewrite", query_rewritting_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("generation", generation_node)

builder.add_edge(START, "query_rewrite")
builder.add_edge("query_rewrite", "retrieval")
builder.add_edge("retrieval", "generation")
builder.add_edge("generation", END)

app = builder.compile()


# ————————————————————————————————————————————————————————————————
def run_rag(question):
    result = app.invoke({
        "question": question
    })

    return RAGResponse(
        final_answer=result.get("final_answer"),
        rewritten_query=result.get("rewritten_query"),
        company_query_list=result.get("company_query_list", []),
        selected_chunks=result.get("selected_chunks", {}),
        status=result["status"],
        message=result.get("message")
    )


#  Test
if __name__ == "__main__":
    question = " ".join(sys.argv[1:])

    if not question:
        question = "What was TSX_Y' revenue in 2022?"

    result = run_rag(question)

    print("\nMatched companies:")
    print(result.company_query_list) # result 是一个 Pydantic Model 对象

    print("\nRetrieved chunks:")
    for company, chunks in result.selected_chunks.items():
        for chunk in chunks:
            print(
                f"{company} | Page {chunk.page} | "
                f"Score {chunk.score:.4f}"
            )
            print(chunk.text[:300])
            print("-" * 50)

    print("\nFinal Answer:")
    print(result.final_answer)
