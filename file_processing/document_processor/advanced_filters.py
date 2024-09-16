import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from file_processing.llm_chat_support import llama_8b_llm_concrete
from langchain_core.pydantic_v1 import BaseModel, Field


class AdvancedFilterSchema(BaseModel):
    score: bool = Field(description="Binary true or false response")
    explanation: str = Field(description="One sentence justifying your decision")

def ask_file_llm(query_text: str, file_sections: List[str]):
    system_message_template = SystemMessagePromptTemplate.from_template(
        "You are a grader assessing the truthfulness of a user's question based on a retrieved document. "
        "Your task is to determine whether the document provides sufficient information to verify the question as true or false. "
        "If the document confirms the truthfulness of the question, respond with 'true'. If it does not provide enough support or contradicts the question, respond with 'false'. "
        "In addition to the boolean response, provide a single-sentence explanation justifying your decision."
    )

    human_message_template = HumanMessagePromptTemplate.from_template(
        "Here is the retrieved document:\n\n{document}\n\nHere is the user question: {question}"
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

    structured_llm = llama_8b_llm_concrete.with_structured_output(AdvancedFilterSchema)

    rewrite_query = (chat_prompt | structured_llm)

    res = rewrite_query.invoke({"document": "\n".join(file_sections), "question":query_text})
    return res