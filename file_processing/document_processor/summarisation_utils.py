import os
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from file_processing.document_processor.md_parser import semantic_markdown_chunks, html_to_plain_text
from file_processing.document_processor.semantic_text_splitter import SemanticChunker
from file_processing.document_processor.types_local import UUIDExtractedItemDict


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


def chunk_into_semantic_chapters(model: Embeddings, text: str, uuid_items: UUIDExtractedItemDict = {},
                                 min_length: int = int(os.environ.get("MIN_CHUNK_LENGTH", "1500")),
                                 max_length: int = int(os.environ.get("MAX_CHUNK_LENGTH", "3000"))) -> List[str]:
    chunker = SemanticChunker(model)
    return chunker.split_text(text, uuid_items, min_length, max_length)


def test_semantic_chapter_chunking():
    with open('../raptor/demo/sample.txt', 'r') as file:
        text = file.read()
    out = chunk_into_semantic_chapters(text)
    print(out)


def test_md_chunking():
    with open("../raptor/demo/random_paper.md", 'r', encoding='utf-8') as file:
        md_content = file.read()
    headers_to_split_on = [
        ("#", "Header 1")
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
