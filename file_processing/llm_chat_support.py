import json
import logging
import os
import sys
import time
from enum import Enum
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.utils.json import parse_json_markdown, parse_partial_json
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from langchain_together import ChatTogether
from together import Together

from raptor.raptor import BaseSummarizationModel

MODELS = {
    "llama3-8b-8192": {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": 30000},
    "gemma-7b-it": {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": 15000},
    "mixtral-8x7b-32768": {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": 5000},
    "llama3-70b-8192": {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": 6000}
}


class SummarizerGroq(BaseSummarizationModel):
    def __init__(self, model_name):
        self.summarizer_chat = ChatGroq(model_name=model_name, max_tokens=int(os.environ.get(
            "TARGET_SUMMARY_LENGTH"
        )))
        self.model_limits = MODELS[model_name]
        self.requests_made = 0
        self.tokens_generated = 0
        self.last_request_time = time.time()

    def summarize(self, context, stop_sequence=None, retry_delay=5):
        while True:
            # Check if we've reached the daily request limit
            if self.requests_made >= self.model_limits["requests_per_day"]:
                print("Daily request limit reached. Waiting for 24 hours.")
                time.sleep(86400)  # wait for 24 hours
                self.requests_made = 0
                self.tokens_generated = 0
                self.last_request_time = time.time()

            # Check if we've reached the minute request limit
            current_time = time.time()
            if current_time - self.last_request_time < 60:
                if self.requests_made >= self.model_limits["requests_per_minute"]:
                    print("Minute request limit reached. Waiting for", 60 - (current_time - self.last_request_time),
                          "seconds.")
                    time.sleep(60 - (current_time - self.last_request_time))
                    self.requests_made = 0
                    self.tokens_generated = 0
                    self.last_request_time = time.time()

            # Check if we've reached the minute token limit
            if self.tokens_generated >= self.model_limits["tokens_per_minute"]:
                print("Minute token limit reached. Waiting for", 60, "seconds.")
                time.sleep(60)
                self.tokens_generated = 0
                self.last_request_time = time.time()

            try:
                system = ("You are an expert summarizer who condenses complex texts into concise summaries that "
                          "accurately capture the main themes, key points, and essential arguments, while maintaining "
                          "the integrity and intent of the original material.")
                human = (f"When preparing a summary, ensure that all content is derived directly from the specified "
                         f"batch of text sections, with no additional information, interpretation, "
                         f"or inference.Adhere strictly to the facts presented in the text.Focus on capturing the "
                         f"main themes and pivotal arguments relevant to the broader narrative or argument of the "
                         f"work.Exclude any details not explicitly mentioned in the text to maintain relevance and "
                         f"clarity.If any part of the text is ambiguous, note the uncertainty without making "
                         f"assumptions or adding your own knowledge of the work.Do not imply or infer any information "
                         f"that is not directly stated.Your goal is to create a concise summary that provides a "
                         f"contextual backdrop, aiding in the reader’s understanding of any part of the batch that "
                         f"may later be selected for detailed analysis.The summary should be coherent and focused, "
                         f"emphasizing relevance and avoiding any extraneous details, to effectively prepare the "
                         f"stage for a deeper examination of any segment within these texts.This approach ensures "
                         f"that the summary is comprehensive and flexible, ready to support the analysis of any "
                         f"specific part chosen subsequently. Here is the batch"
                         f"with the text sections to be summarized:\n\n{context}")

                resp = self.summarizer_chat.invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=human)
                ])
                """ resp.response_metadata
                {'token_usage': 
                    {'completion_tokens': 290,
                     'prompt_tokens': 260,
                     'total_tokens': 550,
                     'completion_time': 0.462656485,
                     'prompt_time': 0.102705737,
                     'queue_time': None,
                     'total_time': 0.565362222
                     },
                 'model_name': 'mixtral-8x7b-32768',
                 'system_fingerprint': 'fp_c5f20b5bb1',
                 'finish_reason': 'stop',
                 'logprobs': None
                }
                """

                token_usage = resp.response_metadata.get("token_usage", {}).get("total_tokens", 0)
                self.tokens_generated += token_usage
                self.requests_made += 1

                return resp.content

            except Exception as e:
                print(e, file=sys.stderr)
                return e


# Very imaginative
chat_model_crazy = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_GPT_4o_API_VERSION"),
    azure_deployment=os.environ.get("AZURE_GPT_4o_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_GPT_4o_ENDPOINT"),
    openai_api_key=os.environ.get("AZURE_GPT_4o_API_KEY"),
    temperature=1
)

# Make it keep to the rules - https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions#temperature-and-top_p-parameters
model_concrete = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_GPT_4o_API_VERSION"),
    azure_deployment=os.environ.get("AZURE_GPT_4o_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_GPT_4o_ENDPOINT"),
    openai_api_key=os.environ.get("AZURE_GPT_4o_API_KEY"),
    temperature=0.2
)

# RankGPT - just reorder
model_no_imagination = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_GPT_4o_API_VERSION"),
    azure_deployment=os.environ.get("AZURE_GPT_4o_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_GPT_4o_ENDPOINT"),
    openai_api_key=os.environ.get("AZURE_GPT_4o_API_KEY"),
    temperature=0
)

# Make it think - traditional temp - https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions#temperature-and-top_p-parameters
model_abstract = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_GPT_4o_API_VERSION"),
    azure_deployment=os.environ.get("AZURE_GPT_4o_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_GPT_4o_ENDPOINT"),
    openai_api_key=os.environ.get("AZURE_GPT_4o_API_KEY"),
    temperature=0.7
)

# RankGPT - just reorder
llama_8b_llm_no_imagination = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llama_8b_llm_concrete = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llama_8b_llm_abstract = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llama_8b_llm_crazy = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llama_8b_llm_no_imagination = ChatTogether(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Very imaginative
big_llama_chat_model_crazy = ChatTogether(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    max_retries=2,
    temperature=1
)

# Make it keep to the rules - https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions#temperature-and-top_p-parameters
big_llama_model_concrete = ChatTogether(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    temperature=0.2
)

# RankGPT - just reorder
big_llama_model_no_imagination = ChatTogether(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    temperature=0
)

# Make it think - traditional temp - https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions#temperature-and-top_p-parameters
big_llama_model_abstract = ChatTogether(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    temperature=0.7
)

together = Together()

class LLMTemp(Enum):
    NO_IMAGINATION = 0
    CONCRETE = 1
    ABSTRACT = 2
    CRAZY = 3

def small_llm_json_response(messages: List[dict]):
    # TODO: JSON model for Llama 3.1 8b on together not yet supported
    # json_llm = get_llm(LLMTemp.ABSTRACT, LLMTypes.SMALL_JSON_MODEL).bind(response_format={"type": "json_object"})

    extract = together.chat.completions.create(
        messages=messages,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        response_format={
            "type": "json_object",
        },
        temperature=0.4,
    )

    try:
        js = parse_json_markdown(extract.choices[0].message.content)
    except BaseException as e:
        logging.error(f"Non fatal for markdown json:\n{extract.choices[0].message.content}", exc_info=e)
        try:
            js = parse_partial_json(extract.choices[0].message.content)
        except BaseException as e:
            logging.error(f"Non fatal for partial json:\n{extract.choices[0].message.content}", exc_info=e)
            js = {}
    return js

class LLMTypes(Enum):
    BIG_VISUAL_MODEL = 0
    SMALL_JSON_MODEL = 1

use_azure_gpt4o = False

def get_llm(temp: LLMTemp, type: LLMTypes = LLMTypes.BIG_VISUAL_MODEL):
    if temp == LLMTemp.NO_IMAGINATION:
        if type == LLMTypes.BIG_VISUAL_MODEL:
            return model_no_imagination if use_azure_gpt4o else big_llama_model_no_imagination
        elif type == LLMTypes.SMALL_JSON_MODEL:
            return llama_8b_llm_no_imagination
    elif temp == LLMTemp.CONCRETE:
        if type == LLMTypes.BIG_VISUAL_MODEL:
            return model_concrete if use_azure_gpt4o else big_llama_model_concrete
        elif type == LLMTypes.SMALL_JSON_MODEL:
            return llama_8b_llm_concrete
    elif temp == LLMTemp.ABSTRACT:
        if type == LLMTypes.BIG_VISUAL_MODEL:
            return model_abstract if use_azure_gpt4o else big_llama_model_abstract
        elif type == LLMTypes.SMALL_JSON_MODEL:
            return llama_8b_llm_abstract
    elif temp == LLMTemp.CRAZY:
        if type == LLMTypes.BIG_VISUAL_MODEL:
            return chat_model_crazy if use_azure_gpt4o else big_llama_chat_model_crazy
        elif type == LLMTypes.SMALL_JSON_MODEL:
            return llama_8b_llm_crazy


class SummarizerAzureGPT(BaseSummarizationModel):
    def summarize(self, context, stop_sequence=None, retry_delay=5):
        while True:

            try:
                system = ("You are an expert summarizer who condenses complex texts into concise summaries that "
                          "accurately capture the main themes, key points, and essential arguments, while maintaining "
                          "the integrity and intent of the original material.")
                human = (f"When preparing a summary, ensure that all content is derived directly from the specified "
                         f"batch of text sections, with no additional information, interpretation, "
                         f"or inference.Adhere strictly to the facts presented in the text.Focus on capturing the "
                         f"main themes and pivotal arguments relevant to the broader narrative or argument of the "
                         f"work.Exclude any details not explicitly mentioned in the text to maintain relevance and "
                         f"clarity.If any part of the text is ambiguous, note the uncertainty without making "
                         f"assumptions or adding your own knowledge of the work.Do not imply or infer any information "
                         f"that is not directly stated.Your goal is to create a concise summary that provides a "
                         f"contextual backdrop, aiding in the reader’s understanding of any part of the batch that "
                         f"may later be selected for detailed analysis.The summary should be coherent and focused, "
                         f"emphasizing relevance and avoiding any extraneous details, to effectively prepare the "
                         f"stage for a deeper examination of any segment within these texts.This approach ensures "
                         f"that the summary is comprehensive and flexible, ready to support the analysis of any "
                         f"specific part chosen subsequently. Here is the batch"
                         f"with the text sections to be summarized:\n\n{context}")

                resp = get_llm(LLMTemp.ABSTRACT).invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=human)
                ])

                return resp.content

            except Exception as e:
                print("Summarisation error: ",e, file=sys.stderr)
                return e
