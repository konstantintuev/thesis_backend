from typing import List

from langchain_core.prompts import ChatPromptTemplate

from file_processing.llm_chat_support import LLMTemp, get_llm

"""
Thank you https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
"""

"""
Idea:
1. Check if question is generic -> route query to higher level RAPTOR summary
                           else -> route query with no restrictions
2. Rewrite latest query based on message history
3. Decompose question
4. Rerank
5. Combine
"""

# Unique thing for file retriever chat
def rewrite_search_query_based_on_history(latest_question: str, previous_questions: List[str]) -> str:
    all_questions = previous_questions + [f"{latest_question} (rewrite this)"]

    formatted_questions = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(all_questions)])

    template = (
        "Rewrite the latest question in a series so that it includes all relevant information from previous questions, "
        "making the latest question fully self-contained and understandable on its own.\n"
        "Preserve the context, but avoid unnecessary repetition.\n"
        "Questions:\n"
        "{questions}\n"
        "Output:")
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    rewrite_query = (prompt_decomposition | get_llm(LLMTemp.NO_IMAGINATION))

    return rewrite_query.invoke({"questions": formatted_questions}).content