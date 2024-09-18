import re
from typing import List
import numpy as np

import torch
from pylate import indexes, models, retrieve
from transformers import is_torch_npu_available

from file_processing.document_processor.semantic_text_splitter import uuid_pattern
from file_processing.document_processor.types_local import UUIDExtractedItemDict
from file_processing.file_queue_management.file_queue_db import get_all_files_queue

def normalize(values):
    min_val = min(values)
    max_val = max(values)

    # Avoid division by zero if all values are the same
    if min_val == max_val:
        return [0.5] * len(values)

    return [(x - min_val) / (max_val - min_val) for x in values]


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

class ColbertLocal():
    def __init__(self):
        pass

    def init_model(self):
        use_fp16 = True
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        elif is_torch_npu_available():
            device = "npu"
        else:
            device = "cpu"
            use_fp16 = False

        self.model = models.ColBERT(
            model_name_or_path="jinaai/jina-colbert-v1-en",
            attend_to_expansion_tokens=True,
            trust_remote_code=True,
            device=device,
            document_length=8192,
            query_length=256,
            model_kwargs={
                "torch_dtype": torch.float16 if use_fp16 else torch.float32
            }
        )

        try:
            self.index = indexes.Voyager(
                index_folder=".pylate-index",
                index_name="index",
                override=False,
            )

            self.retriever = retrieve.ColBERT(index=self.index)

            return False
        except BaseException as e:
            print('An exception occurred: {}'.format(e))

            self.index = indexes.Voyager(
                index_folder=".pylate-index",
                index_name="index",
                override=True,
            )

            self.retriever = retrieve.ColBERT(index=self.index)

            return True

    def initialise_search_component(self):
        self.search_colbert_index("Initialise the search component")
        pass

    def get_batch_size(self):
        # Fit into 8GB of VRAM
        return 3


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

        documents_embeddings = self.model.encode(
            document_collection,
            batch_size=self.get_batch_size(),
            is_query=False,  # Encoding documents
            show_progress_bar=True,
            precision="float32",
            convert_to_numpy=True
        )

        # We use quantised float16 for the model as the gpu is a bit old - GTX 1080,
        #   but pylate likes float32 and we don't want to quantise to int
        embeddings = [embedding.astype(np.float32) for embedding in documents_embeddings]

        # Add the documents ids and embeddings to the Voyager index
        self.index.add_documents(
            documents_ids=document_ids,
            documents_embeddings=embeddings
        )



    def search_colbert_index(self, query: str,
                             high_level_summary: str = None,
                             unique_file_ids: List[str] = None,
                             source_count: int = None) -> dict or None:
        try:
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

            # It's a list of numpy arrays
            queries_embeddings = self.model.encode(
                [query],
                batch_size=self.get_batch_size(),
                is_query=True,  # Encoding queries
                show_progress_bar=True,
                precision="float32",
                convert_to_numpy=True
            )

            # We use quantised float16 for the model as the gpu is a bit old - GTX 1080,
            #   but pylate likes float32
            embeddings = [embedding.astype(np.float32) for embedding in queries_embeddings]

            res = self.retriever.retrieve(
                queries_embeddings=embeddings,
                k=internal_source_count,
                batch_size=100
            )

            res = res[0]  # [0] as we have a single query

            scores_normal = normalize([chunk["score"] for chunk in res])

            reranked = [{
                "unnormal_score": chunk["score"],
                "score": scores_normal[index],
                "passage_id": chunk["id"]
            } for index, chunk in enumerate(res)]

            return reranked

        except BaseException as error:
            print('An exception occurred: {}'.format(error))
            return []


    def test_colbert(self):
        # 2312.05934v3.pdf
        file_data = get_all_files_queue()["files"]
        file_data["tree"] = file_data["result"]["tree"]
        file_data["result"] = None
        self.add_documents_to_index(file_data)


# TODO: enable when colbert get's better
colbert_local = ColbertLocal()