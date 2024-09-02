import os
from typing import List
import re

from ragatouille import RAGPretrainedModel
from FlagEmbedding import LayerWiseFlagLLMReranker
from operator import itemgetter

from file_processing.document_processor.semantic_text_splitter import uuid_pattern
from file_processing.document_processor.types_local import UUIDExtractedItemDict
from file_processing.file_queue_management.file_queue_db import get_file_from_queue, get_all_files_queue


def test_colbert():
    # 2312.05934v3.pdf
    file_data = get_all_files_queue()["files"]
    file_data["tree"] = file_data["result"]["tree"]
    file_data["result"] = None
    add_documents_to_index(file_data)


index_path = ".ragatouille/colbert/indexes/default"
colbert_model: RAGPretrainedModel = (
    RAGPretrainedModel.from_index(index_path)
    if os.path.exists(index_path)
    else RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v1-en")
)


def initialise_search_component():
    search_colbert_index("Initialise the search component")
    pass


def add_uuid_object_to_string(match, uuid_items: UUIDExtractedItemDict):
    # Get the matched UUID
    uuid = match.group(0)
    # Check if the UUID exists in the uuid_items and return its string representation
    if uuid in uuid_items:
        item = uuid_items[uuid]
        if item['type'] == 'ul':
            list_marker = '*'
            list_string = '\n'.join([f'{list_marker} {child}' for child in item['children']])
            return '\n' + list_string
        elif item['type'] == 'li':
            list_string = '\n'.join([f'{i + 1}. {child}' for i, child in enumerate(item['children'])])
            return '\n' + list_string
        elif item['type'] == 'table':
            return '\n' + item["content"]
        elif item['type'] == 'math':
            return '\n' + item["content"]
        else:
            return uuid
    return uuid


"""
Expected input structure:
files = {
        "tree": {
            "id": f"{uuid.uuid4()}",
            "embedding": node.embeddings['EMB'],
            "text": node.text,
            "children": [child for child in node.children],
            "layer": layer
        },
        "metadata": pdf_metadata.to_dict(),
        "uuid_items": UUIDExtractedItemDict
    }
"""


def add_documents_to_index(files: List[dict]):
    document_collection = [re.sub(uuid_pattern,
                                  lambda match: add_uuid_object_to_string(match, file_data["uuid_items"]),
                                  node["text"])  #node["text"]
                           for file_data in files
                           # Values of dict in python are ORDERED
                           for node in file_data["tree"].values()]
    document_ids = [node['id']
                    for file_data in files
                    # Values of dict in python are ORDERED
                    for node in file_data["tree"].values()]
    document_metadatas = [{"doc_id": file_data['file_uuid'],
                           "chunk_id": node["id"],
                           "children": node["children"],
                           "layer": node["layer"]}
                          for file_data in files
                          for node in file_data["tree"].values()]
    index_path = ".ragatouille/colbert/indexes/default"

    if not os.path.exists(index_path):
        colbert_model.index(
            collection=document_collection,
            document_ids=document_ids,
            document_metadatas=document_metadatas,
            max_document_length=8190,
            split_documents=False,
            index_name="default"
        )
        initialise_search_component()
    else:
        colbert_model.add_to_index(
            new_collection=document_collection,
            new_document_ids=document_ids,
            new_document_metadatas=document_metadatas,
            index_name="default",
            split_documents=False,
        )
        # TODO: the index is not updated after adding


"""
Reranker proved to have marginal improvement at best at the cost of 2 mins per query
Keeping the code here if better research comes out, but disabled for now
------------------------------------------------------------------------------------
reranker = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise',
                        use_fp16=True,
                        device='mps')
"""


def search_colbert_index(query: str, high_level_summary: str = None) -> dict or None:
    if not os.path.exists(index_path):
        return {"error": "No files indexed!"}
    # 'colbert_model.search' returns a list of dictionaries with the following structure:
    # {
    # "score": 22.4420108795166,
    # "content": "text of the relevant passage",
    # "rank": 1,
    # "passage_id": 9,
    # "document_metadata": {
    #     "doc_id": "c8978a17-7d35-4d5a-977e-c295ab5e16b1",
    #     "chunk_id": "4e33e0a3-7622-4840-b195-75fa9339373f",
    #     "children": [],
    #     "layer": 0
    #  }
    # }

    res = colbert_model.search(
        query=query,
        index_name="default",
        k=100
    )
    """
    Tested with:
    Eval raw Colbert (jinaai/jina-colbert-v1-en; Passage) 
     vs Colbert + Rerank (BAAI/bge-reranker-v2-minicpm-layerwise, 28 layers; Passage V2; Standard Prompt) 
     vs Colbert + Rerank(BAAI/bge-reranker-v2-minicpm-layerwise, 28 layers; Passage V3 prompt with context)
     vs lastest below Colbert + Rerank(BAAI/bge-reranker-v2-minicpm-layerwise, 28 layers; Passage V4 prompt with better integration of context)
     
    Evalueted retrieval manually + https://chatgpt.com/c/66b32b82-08b1-4e63-bbb9-918e43521dc2
    
    File: 2312.05934v3.pdf
    Query: What are some innovative uses for large language models (LLMs)?
    high_level_summary: "The text evaluates three models (Llama2-7B, Mistral-7B, and Orca2-7B) and an embedding model (bge-large-en) for a question-answering system, using unsupervised learning and special tokens to preserve document structure. RAG outperforms fine-tuning in a knowledge injection study for large language models (LLMs), measuring their ability to answer factual questions accurately. The study found RAG more effective in learning new information, with fine-tuning focusing on overall response quality rather than knowledge breadth. The text also discusses various strategies for fine-tuning LLMs, including reinforcement learning and unsupervised fine-tuning methods. Paraphrasing tasks and scenarios demonstrate models' abilities to infer correct answers based on reasoning and prior knowledge. The knowledge injection framework has limitations, primarily evaluating factual information and not accounting for other quality metrics."
    
    The https://huggingface.co/jinaai/jina-colbert-v1-en model is also noted by the original research
      to be almost as good as the rerankers BGE and MiniLM-L-6-v2 (BAAI/bge-reranker-v2-minicpm-layerwise is based on that model too).
    
    ---------------------------------------------------------------------------------------------------------------------
    
    res = [chunk for chunk in res if chunk["document_metadata"]["doc_id"] == "c8978a17-7d35-4d5a-977e-c295ab5e16b1"]

    rerank_score = reranker.compute_score([(query, chunk["content"]) for chunk in res],
                                          max_length=8190,
                                          batch_size=25,
                                          cutoff_layers=28,
                                          prompt="Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'.\n" +
                                                 "This is a high-level summary of the document, use it only as additional context if there is any uncertainty about the content of any of the passages, don't use it to evaluate relevance:\n" +
                                                 f"{high_level_summary}")
    """

    reranked = [{
        # "content": chunk["content"],
        "score": chunk["score"] / 100,
        "rank": chunk["rank"],
        # "rerank_score": rerank_score[index],
        "passage_id": chunk["passage_id"],
        "document_metadata": chunk["document_metadata"]
    } for index, chunk in enumerate(res)]

    # reranked_order = sorted(reranked, key=itemgetter('rerank_score'), reverse=True)

    return reranked
