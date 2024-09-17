from operator import attrgetter, itemgetter

import numpy
import torch
from FlagEmbedding import FlagLLMReranker
from colbert import Checkpoint
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score
from together import Together
from transformers import is_torch_npu_available

from file_processing.document_processor.rank_gpt import RankGPTRanker


reasonable_reranking = False

reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma',
                           use_fp16=True) if reasonable_reranking else None


def normalize(values):
    min_val = min(values)
    max_val = max(values)

    # Avoid division by zero if all values are the same
    if min_val == max_val:
        return [0.5] * len(values)

    return [(x - min_val) / (max_val - min_val) for x in values]

def do_reasonable_reranking(query, res, reorder):
    if not reasonable_reranking:
        return res

    rerank_score = reranker.compute_score([(query, chunk["content"]) for chunk in res],
                                               max_length=8192,
                                               batch_size=1,
                                               normalize=True)

    reranked = [{
        "content": chunk["content"],
        "score": rerank_score[index],
        "rank": chunk["rank"],
        "orig_score": chunk["score"] / 100,
        "passage_id": chunk["passage_id"],
        "document_metadata": chunk["document_metadata"]
    } for index, chunk in enumerate(res)]

    if reorder:
        reranked = sorted(reranked, key=itemgetter('score'), reverse=True)

    return reranked


def do_llm_reranking(query, res, reorder):
    from rerankers.documents import Document
    crazy_reranker = RankGPTRanker()
    reranked_orig = crazy_reranker.rank(query, [
        Document(chunk["content"], chunk["passage_id"])
        for chunk in res
    ])

    scores = normalize([chunk["score"] for chunk in res])

    reranked = [{
        "content": chunk["content"],
        "orig_rank": reranked_orig.get_result_by_docid(chunk["passage_id"]).rank,
        "rank": chunk["rank"],
        "score": scores[index],
        "unnormal_score": chunk["score"],
        "passage_id": chunk["passage_id"],
        "document_metadata": chunk["document_metadata"]
    } for index, chunk in enumerate(res)]

    if reorder:
        reranked = sorted(reranked, key=itemgetter('rank'), reverse=False)

    return reranked


client = Together()


def do_llama_rerank(query, res, reorder):
    response = client.rerank.create(
        # Based on Llama 3 8b - can't get better than this
        # Thank you salesforce: https://blog.salesforceairesearch.com/llamarank/
        # But is closed source for now! -> nice alternative above
        model="Salesforce/Llama-Rank-V1",
        query=query,
        documents=[chunk["content"] for chunk in res]
    )

    response.results.sort(key=attrgetter('index'))

    reranked = [{
        # Don't return content again
        #"content": chunk["content"],
        "orig_score": chunk["score"],
        "rank": chunk["rank"],
        "score": response.results[index].relevance_score,
        "passage_id": chunk["passage_id"],
        "document_metadata": chunk["document_metadata"]
    } for index, chunk in enumerate(res)]

    if reorder:
        reranked.sort(key=itemgetter('score'), reverse=True)

    return reranked


def get_colbert():
    config = ColBERTConfig(
        root="experiments",
        gpus=1
    )
    use_fp16 = True
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif is_torch_npu_available():
        device = torch.device("npu")
    else:
        device = torch.device("cpu")
        use_fp16 = False
        config = ColBERTConfig(
            root="experiments"
        )

    ckpt = Checkpoint("jinaai/jina-colbert-v1-en", colbert_config=config)
    if use_fp16:
        ckpt.half()
    ckpt = ckpt.to(device)
    return ckpt

ckpt = get_colbert()

# Cheap and efficient
def do_colbert_rerank(query, res, reorder):
    Q = ckpt.queryFromText([query])
    D = ckpt.docFromText([chunk["content"] for chunk in res], bsize=6, showprogress=True)[0]
    D_mask = torch.ones(D.shape[:2], dtype=torch.long)
    scores = colbert_score(Q, D, D_mask).flatten().cpu().numpy().tolist()
    ranking = numpy.argsort(scores)[::-1].tolist()

    scores = normalize(scores)

    reranked = [{
        # Don't return content again
        #"content": chunk["content"],
        "orig_score": chunk["score"],
        "rank": ranking[index],
        "score": scores[index],
        "passage_id": chunk["passage_id"],
        "document_metadata": chunk["document_metadata"]
    } for index, chunk in enumerate(res)]

    if reorder:
        reranked.sort(key=itemgetter('score'), reverse=True)

    return reranked
