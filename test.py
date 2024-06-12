import torch

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from file_processing.document_processor.embeddings import embeddings_model


def main():
    hf = embeddings_model
    embedding = hf.embed_query("hi this is harrison")
    len(embedding)

if __name__ == "__main__":
    main()