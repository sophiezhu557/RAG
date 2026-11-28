import logging
from openai import OpenAI, APIError
from schemas import TokenUsage

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """
You are viewing a financial report, and your task is to transform a natural language 
question into one concise search query suitable for downstream semantic retrieval.

Rules:
1. Remove any company full names that are provided in Identified companies name list.
2. Preserve financial metrics and important qualifiers.
3. Preserve all relevant years, dates, and reporting periods.
4. Preserve distinctions such as consolidated vs standalone,
   gross vs net, reported vs adjusted, and before vs after costs.
5. Keep useful financial abbreviations and their meanings.
6. Remove conversational wording and unnecessary phrases.
7. Remove generic source phrases such as "in the annual report" or
   "according to the annual report" when retrieval is already scoped
   to the identified company's report. Keep specific dates, fiscal
   years, reporting periods, and named sections that narrow the search.
8. Do not answer the question.
9. Do not invent information or add unsupported assumptions.
10. Return only ONE rewritten query as plain text.


Example:
question: Did Mosaic Brands Limited mention any mergers or acquisitions in the annual report? 
identified companies: ["Mosaic Brands Limited"]
query: mergers or acquisitions

question: What's Apple's total revenue in 2022?
Identified companies: ["Apple"]
query: total revenue in 2022

question: What was Compagnie Financière Tradition's adjusted consolidated revenue in 2022?
Identified companies: ["Tradition"]
query: adjusted consolidated revenue in 2022
"""

def rewrite_query(
    question,
    company_query_list,
    llm
):
    company_names = [
        company["company_name"]
        for company in company_query_list
    ]

    if company_names:
        response = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": REWRITE_PROMPT
                },
                {
                    "role": "user",
                    "content": (
                        f"Identified companies: {company_names}\n"
                        f"Original question: {question}"
                    )
                }
            ],
            temperature=0,
            max_tokens=120
        )

        usage = TokenUsage.from_api(response.usage)
        rewritten_query = response.choices[0].message.content

        if not rewritten_query or not rewritten_query.strip():
            logger.warning("Empty rewritten query. Using original.")
            return question, usage

        return rewritten_query.strip(), usage

    return question, TokenUsage()



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from company_router import identify_companies, load_company_mapping

    load_dotenv()

    llm = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )

    company_mapping = load_company_mapping()

    test_cases = [
        "What was Compagnie Financière Tradition's price earnings ratio (PER 1) at 31 December 2021?",
        "What was Compagnie Financière Tradition's reported consolidated revenue in 2022?",
        "How much did ENRG Elements Limited raise through its June 2022 share placement before costs?",
        "What was the Takardeit deposit's JORC 2012 Inferred Mineral Resource in contained uranium?"
    ]

    for query in test_cases:
        company_query_list = identify_companies(
                query,
                company_mapping
            )
        rewritten = rewrite_query(
            question=query,
            company_query_list=company_query_list,
            llm=llm
        )

        print("\nOriginal:", query)
        print(company_query_list)
        print("Rewritten:", rewritten)
        print("-" * 60)
