import logging
import os
import time
import uuid
from typing import List

import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from file_processing.document_processor.llm_chat_support import SummarizerGroq, SummarizerAzureGPT
from file_processing.embeddings import embeddings_model, pending_embeddings_singleton
from file_processing.document_processor.summarisation_utils import chunk_into_semantic_chapters

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

from langchain_together.embeddings import TogetherEmbeddings
from tenacity import wait_random_exponential, stop_after_attempt, retry

from raptor.raptor import RetrievalAugmentation, RetrievalAugmentationConfig, SBertEmbeddingModel

from raptor.raptor import BaseSummarizationModel
from langchain_core.prompts import ChatPromptTemplate

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


class TogetherEmbeddingModel(BaseEmbeddingModel):
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

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embeddings(self, text: List[str]) -> List[List[float]]:
        # Implement your embedding logic here
        # Return the embeddings as a list of numpy arrays or a list of lists of floats
        #embeddings = [[0.0] * embedding_dim] * len(text)  # Replace with actual embedding logic
        while True:
            try:
                embedding_list = embeddings.embed_documents(text)
                return embedding_list
            except Exception as e:
                print(e)


class BGE3EmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self.model = embeddings_model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        return self.model.embed_documents([text])[0]

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)


class LangchainPendingEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        return pending_embeddings_singleton.embed_documents([text])[0]

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        return pending_embeddings_singleton.embed_documents(texts)


def node_to_dict(node, layer):
    """Convert a tree node to a dictionary format for JSON."""
    node_dict = {
        "id": f"{uuid.uuid4()}",
        "embedding": node.embeddings['EMB'],
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
    # Map children int keys to child UUIDs
    for node in all_nodes_dict.values():
        node["children"] = [all_nodes_dict[child_index]["id"] for child_index in node["children"]]
    return all_nodes_dict


# Initialize your custom models
custom_summarizer = SummarizerAzureGPT()  #SummarizerGroq("mixtral-8x7b-32768")
custom_qa = CustomQAModel()
custom_embedding = LangchainPendingEmbeddingModel()  #BGE3EmbeddingModel() #SBertEmbeddingModel(device="mps") #TogetherEmbeddingModel()

# Create a config with your custom models
custom_config = RetrievalAugmentationConfig(
    summarization_model=custom_summarizer,
    qa_model=custom_qa,
    embedding_model=custom_embedding,

)

import json

SAVE_PATH = "../raptor/demo/cinderella"


# Initialize RAPTOR with your custom config


def read_file():
    RA = RetrievalAugmentation(config=custom_config, tree=SAVE_PATH)
    tree_dict = tree_to_dict(RA.tree)

    # Convert the dictionary to JSON
    tree_json = json.dumps(tree_dict, indent=4)

    # Save the JSON to a file
    with open('../raptor/demo/tree_structure.json', 'w') as json_file:
        json_file.write(tree_json)

    # Output the JSON for inspection
    print(tree_json)


print("NE")


def main():
    RA = RetrievalAugmentation(config=custom_config)
    with open('../raptor/demo/sample.txt', 'r') as file:
        text = file.read()
    RA.add_documents(text)
    RA.save(SAVE_PATH)
    RA.answer_question("What is Cinderella?")

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
    return


def main_semantic_chapters():
    RA = RetrievalAugmentation(config=custom_config, tree=SAVE_PATH)
    with open('../raptor/demo/sample.txt', 'r') as file:
        text = file.read()
    out = chunk_into_semantic_chapters(text)
    RA.add_semantic_chapters(out)
    RA.save(SAVE_PATH)
    #RA.answer_question("What is Cinderella?")
    RA.answer_question("What moral lessons can be learned from the story of Cinderella?")

    # print(text)
    # Convert the tree to dictionary
    tree_dict = tree_to_dict(RA.tree)

    # Convert the dictionary to JSON
    tree_json = json.dumps(tree_dict, indent=4)

    # Save the JSON to a file
    with open('../raptor/demo/tree_structure.json', 'w') as json_file:
        json_file.write(tree_json)

    # Output the JSON for inspection
    print(tree_json)
    return


if __name__ == "__main__":
    print("OK")
    #main()
    main_semantic_chapters()
    #read_file()
