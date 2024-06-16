from typing import List

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

from file_processing.document_processor.embeddings import embeddings_model
from file_processing.document_processor.md_parser import semantic_markdown_chunks, html_to_plain_text
from file_processing.document_processor.semantic_text_splitter import SemanticChunker
from file_processing.document_processor.types import UUIDExtractedItemDict


# Subclass the Embeddings class using SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'multi-qa-mpnet-base-cos-v1'):
        self.model = SentenceTransformer(model_name, device="mps")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.encode(text, convert_to_tensor=False).tolist()


def chunk_into_semantic_chapters(model: Embeddings, text: str, uuid_items: UUIDExtractedItemDict = {}) -> List[str]:
    chunker = SemanticChunker(model)
    return chunker.split_text(text, uuid_items)


def test_semantic_chapter_chunking():
    with open('../raptor/demo/sample.txt', 'r') as file:
        text = file.read()
    out = chunk_into_semantic_chapters(text)
    print(out)


def test_md_chunking():
    with open("../raptor/demo/random_paper.md", 'r', encoding='utf-8') as file:
        md_content = file.read()
    headers_to_split_on = [
        ("h1", "Header 1"),
        # ("h2", "Header 2"),
        # ("h3", "Header 3"),
    ]

    semantic_chapters = []
    html_header_splits, uuid_items = semantic_markdown_chunks(md_content, headers_to_split_on, 300)
    for chunk in html_header_splits:
        plain_text = html_to_plain_text(chunk.page_content)
        out = chunk_into_semantic_chapters(plain_text, uuid_items)
        semantic_chapters.extend(out)
    print("ok")


if __name__ == "__main__":
    #test_semantic_chapter_chunking()
    test_md_chunking()
