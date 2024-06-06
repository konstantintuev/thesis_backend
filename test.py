import torch

from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def main():
    model_name = "BAAI/bge-m3"


    model_kwargs = {"device": "mps"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    embedding = hf.embed_query("hi this is harrison")
    len(embedding)

if __name__ == "__main__":
    main()