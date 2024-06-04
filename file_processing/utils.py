import logging
import time

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

from langchain_together.embeddings import TogetherEmbeddings
from tenacity import wait_random_exponential, stop_after_attempt, retry

from raptor.raptor import RetrievalAugmentation, RetrievalAugmentationConfig, SBertEmbeddingModel

from raptor.raptor import BaseSummarizationModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        # Initialize your model here
        self.summarizer_chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", max_tokens=500)
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            system = "You are a helpful assistant."
            human = f"Write a summary of the following, including as many key details as possible: {context}:"

            resp = self.summarizer_chat.invoke([
                SystemMessage(content=system),
                HumanMessage(content=human)
            ])
            resp.response_metadata

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

            return resp.content

        except Exception as e:
            print(e)
            return e


MODELS = {
    "llama3-8b-8192": {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": 30000},
    "gemma-7b-it": {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": 15000},
    "mixtral-8x7b-32768": {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": 5000},
    "llama3-70b-8192": {"requests_per_minute": 30, "requests_per_day": 14400, "tokens_per_minute": 6000}
}


class Summarizer(BaseSummarizationModel):
    def __init__(self, model_name):
        self.summarizer_chat = ChatGroq(temperature=0, model_name=model_name, max_tokens=500)
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
                system = "You are a helpful assistant."
                human = f"Write a summary of the following, including as many key details as possible: {context}:"

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
                print(e)
                return e


from raptor.raptor import BaseQAModel


class CustomQAModel(BaseQAModel):
    def __init__(self):
        # Initialize your model here
        pass

    def answer_question(self, context, question):
        # Implement your QA logic here
        # Return the answer as a string
        answer = "Your answer here"
        return answer


from raptor.raptor import BaseEmbeddingModel

embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        # Initialize your model here
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        # Implement your embedding logic here
        # Return the embedding as a numpy array or a list of floats
        #embedding = [0.0] * embedding_dim  # Replace with actual embedding logic

        #return np.random.rand(768)

        try:
            embedding = embeddings.embed_documents([text])
            # convert list of floats (embedding[0]) to numpy array
            return np.array(embedding[0])
        except Exception as e:
            print(e)
            return e


# Assuming the tree structure is already available as `tree`

def node_to_dict(node, layer):
    """Convert a tree node to a dictionary format for JSON."""
    node_dict = {
        "embedding": node.embeddings['EMB'].tolist(),
        "text": node.text,
        "children": [child for child in node.children],
        "layer": layer
    }
    return node_dict


def tree_to_dict(tree):
    """Convert the entire tree to a dictionary format for JSON using layer_to_nodes."""
    all_nodes_dict = {}
    for layer, nodes in tree.layer_to_nodes.items():
        for node in nodes:
            all_nodes_dict[node.index] = node_to_dict(node, layer)
    return all_nodes_dict

# Initialize your custom models
custom_summarizer = Summarizer("mixtral-8x7b-32768")
custom_qa = CustomQAModel()
custom_embedding = SBertEmbeddingModel()#CustomEmbeddingModel()

# Create a config with your custom models
custom_config = RetrievalAugmentationConfig(
    summarization_model=custom_summarizer,
    qa_model=custom_qa,
    embedding_model=custom_embedding,

)

import json

SAVE_PATH = "../raptor/demo/cinderella"
# Initialize RAPTOR with your custom config
RA = RetrievalAugmentation(config=custom_config)#,tree=SAVE_PATH)

print("NE")
def main():
    with open('../raptor/demo/sample.txt', 'r') as file:
        text = file.read()
    #print(text)
    # Convert the tree to dictionary
    tree_dict = tree_to_dict(RA.tree)

    # Convert the dictionary to JSON
    tree_json = json.dumps(tree_dict, indent=4)

    # Save the JSON to a file
    with open('../raptor/demo/tree_structure.json', 'w') as json_file:
        json_file.write(tree_json)

    # Output the JSON for inspection
    print(tree_json)
    #return
    #RA.add_documents(text)
    #RA.save(SAVE_PATH)
    #RA.answer_question("What is Cinderella?")


if __name__ == "__main__":
    print("OK")
    main()
