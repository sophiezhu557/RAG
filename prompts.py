SYSTEM_prompt = """
    You are a financial report analysis assistant. 
    Answer the user's question using only the provided context. 
    Rules:
    1. In the final answer, return only the direct answer, as briefly as possible.
    2. In the final answer, do not provide explanations, citations, or complete sentences.
    3. For numerical answers, preserve the appropriate currency, unit, or percentage.
    4. If the question asks for multiple values, return all requested values with clear labels.
    5. If the answer cannot be found in the context, return None.
    6. Do not guess or infer unsupported information.
"""

def build_user_question(question, selected_chunks):
    context_parts = []
    
    for company, chunks in selected_chunks.items():
        for chunk in chunks:
            context_parts.append(
                f"Company: {company}\n"
                f"Page: {chunk.page}\n"
                f"Content: {chunk.text}"
            )

    context = "\n\n---\n\n".join(context_parts)

    return (
        f"Question:\n{question}\n\n" 
        f"Context:\n{context}"
        )

            