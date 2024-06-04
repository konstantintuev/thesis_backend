from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

from file_processing.document_processor.text_splitter import SemanticChunker


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


def chunk_into_semantic_chapters(text: str) -> [str]:
    chunker = SemanticChunker(SentenceTransformerEmbeddings())
    return chunker.split_text(text)

def test_semantic_chapter_chunking():
    with open('../raptor/demo/sample.txt', 'r') as file:
        text = file.read()
    out = chunk_into_semantic_chapters(text)
    print(out)

if __name__ == "__main__":
    test_semantic_chapter_chunking()