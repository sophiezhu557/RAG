
import os
import sys
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from company_router import load_company_mapping, identify_companies
from retrival import load_databases, retrieve


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
    company_query_list: list[dict]
    selected_chunks: dict
    final_answer: str


# 2. Retrieval Node
def retrieval_node(state: RAGState):
    query = state["question"]

    company_query_list = identify_companies(
        query,
        company_mapping
    )

    if company_query_list:
        selected_chunks = retrieve(
            query=query,
            company_query_list=company_query_list,
            databases=databases,
            top_k=5
        )
    else:
        selected_chunks = {}

    return {
        "company_query_list": company_query_list,
        "selected_chunks": selected_chunks
    }


# 3. Generation Node
def generation_node(state: RAGState):
    question = state["question"]
    selected_chunks = state["selected_chunks"]

    if not selected_chunks:
        return {
            "final_answer": "Can't find relevant information."
        }

    # Build context from retrieved chunks
    context_parts = []

    for company, chunks in selected_chunks.items():
        for chunk in chunks:
            context_parts.append(
                f"Company: {company}\n"
                f"Page: {chunk['page']}\n"
                f"Content: {chunk['text']}"
            )

    context = "\n\n---\n\n".join(context_parts)

    # Call DeepSeek
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial report analysis assistant. "
                    "Answer the user's question using only the "
                    "provided context from annual reports. "
                    "Do not invent financial figures. "
                    "Pay attention to companies, years, "
                    "currencies and units. "
                    "If the context is insufficient, say "
                    "'Can't find relevant information.' "
                    "Cite the company name and page number "
                    "for supporting evidence."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}"
                )
            }
        ],
        temperature=0
    )

    final_answer = response.choices[0].message.content

    return {
        "final_answer": final_answer
    }


# 4. Build LangGraph
builder = StateGraph(RAGState)

builder.add_node("retrieval", retrieval_node)
builder.add_node("generation", generation_node)

builder.add_edge(START, "retrieval")
builder.add_edge("retrieval", "generation")
builder.add_edge("generation", END)

app = builder.compile()


# 5. Test
if __name__ == "__main__":
    question = " ".join(sys.argv[1:])

    if not question:
        question = "What was TSX_Y' revenue in 2022?"

    result = app.invoke({
        "question": question
    })

    print("\nMatched companies:")
    print(result["company_query_list"])

    print("\nRetrieved chunks:")
    for company, chunks in result["selected_chunks"].items():
        for chunk in chunks:
            print(
                f"{company} | Page {chunk['page']} | "
                f"Score {chunk['score']:.4f}"
            )
            print(chunk["text"][:300])
            print("-" * 50)

    print("\nFinal Answer:")
    print(result["final_answer"])
