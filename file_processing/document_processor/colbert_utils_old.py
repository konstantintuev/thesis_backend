import os
from typing import List
import re

import psutil
from ragatouille import RAGPretrainedModel
from FlagEmbedding import LayerWiseFlagLLMReranker
from operator import itemgetter, attrgetter
from FlagEmbedding import FlagLLMReranker
from together import Together

from file_processing.document_processor.rank_gpt import RankGPTRanker
from file_processing.document_processor.semantic_text_splitter import uuid_pattern
from file_processing.document_processor.types_local import UUIDExtractedItemDict
from file_processing.file_queue_management.file_queue_db import get_file_from_queue, get_all_files_queue
from file_processing.query_processor.process_search_query import rewrite_search_query_based_on_history
from file_processing.document_processor.rerankers_local import do_llama_rerank, normalize


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
Sadly colbert is far from ready:
- adding to index reencodes all passages TWICE
- the index is lost if the indexing fails
- colbert eats a lot of VRAM no matter the batch size (I hit 8GB of NVIDIA GTX 1080 VRAM with batch size 1 and ~ 220 passages)
- on cpu indexing ~220 passages takes 1 hour (encoding the passages twice)

I tried to make colbert to use the GPU for encoding but to no avail - the tensor operations are too complex.
"""

class ColbertLocal():
    def __init__(self):
        self.index_path = ".ragatouille/colbert/indexes/default"
        self.colbert_model: RAGPretrainedModel = (
            RAGPretrainedModel.from_index(self.index_path)
            if os.path.exists(self.index_path)
            else RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v1-en")
        )

    def initialise_search_component(self):
        self.search_colbert_index("Initialise the search component")
        pass

    def get_batch_size(self):
        return 5


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

    def add_documents_to_index(self, files: List[dict]):
        document_collection = [re.sub(uuid_pattern,
                                      lambda match: add_uuid_object_to_string(match, file_data["uuid_items"]),
                                      node["text"])  # node["text"]
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
            self.colbert_model.index(
                collection=document_collection,
                document_ids=document_ids,
                document_metadatas=document_metadatas,
                max_document_length=8190,
                split_documents=False,
                index_name="default",
                bsize=self.get_batch_size(),
            )
            self.initialise_search_component()
        else:
            self.colbert_model.add_to_index(
                new_collection=document_collection,
                new_document_ids=document_ids,
                new_document_metadatas=document_metadatas,
                index_name="default",
                split_documents=False,
                bsize=self.get_batch_size()
            )
            self.colbert_model: RAGPretrainedModel = RAGPretrainedModel.from_index(self.index_path)



    def search_colbert_index(self, query: str,
                             high_level_summary: str = None,
                             unique_file_ids: List[str] = None,
                             source_count: int = None,
                             no_reranking: bool = False) -> dict or None:
        if not os.path.exists(self.index_path):
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

        if unique_file_ids and len(unique_file_ids) == 0:
            return []

        internal_source_count = (source_count if source_count else 100)

        res = self.colbert_model.search(
            query=query,
            index_name="default",
            k=internal_source_count,
            doc_ids=unique_file_ids
        )

        scores = normalize([chunk["score"] for chunk in res])

        reranked = [{
            # "content": chunk["content"],
            "score": scores[index],
            "unnormal_score": chunk["score"],
            "rank": chunk["rank"],
            # "rerank_score": rerank_score[index],
            "passage_id": chunk["passage_id"],
            "document_metadata": chunk["document_metadata"]
        } for index, chunk in enumerate(res)]

        return reranked

    def test_colbert(self):
        # 2312.05934v3.pdf
        file_data = get_all_files_queue()["files"]
        file_data["tree"] = file_data["result"]["tree"]
        file_data["result"] = None
        self.add_documents_to_index(file_data)


# TODO: enable when colbert get's better
colber_local = None # ColbertLocal()