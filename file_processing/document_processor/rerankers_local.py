from operator import attrgetter, itemgetter

from FlagEmbedding import FlagLLMReranker
from pylate import rank
from together import Together

from file_processing.document_processor.colbert_utils_pylate import colbert_local, normalize
from file_processing.document_processor.rank_gpt import RankGPTRanker


class RerankersLocal():
    reranker = None

    def do_reasonable_reranking(self, query, res, reorder):
        if not self.reranker:
            self.reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma',
                            use_fp16=True)

        rerank_score = self.reranker.compute_score([(query, chunk["content"]) for chunk in res],
                                              max_length=8192,
                                              batch_size=1,
                                              normalize=True)

        reranked = [{
            # "content": chunk["content"],
            "score": rerank_score[index],
            "orig_score": chunk["score"] / 100,
            "passage_id": chunk["passage_id"]
        } for index, chunk in enumerate(res)]

        if reorder:
            reranked = sorted(reranked, key=itemgetter('score'), reverse=True)

        return reranked

    def do_llm_reranking(self, query, res, reorder):
        from rerankers.documents import Document
        crazy_reranker = RankGPTRanker()
        reranked_orig = crazy_reranker.rank(query, [
            Document(chunk["content"], chunk["passage_id"])
            for chunk in res
        ])

        scores = normalize([chunk["score"] for chunk in res])

        # TODO: change rank into a score
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

    def do_llama_rerank(self, query, res, reorder):
        response = self.client.rerank.create(
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
            # "content": chunk["content"],
            "orig_score": chunk["score"],
            "score": response.results[index].relevance_score,
            "passage_id": chunk["passage_id"]
        } for index, chunk in enumerate(res)]

        if reorder:
            reranked.sort(key=itemgetter('score'), reverse=True)

        return reranked

    # Cheap and efficient
    def do_colbert_rerank(self, query, res, reorder):
        queries_embeddings = colbert_local.model.encode(
            [query],
            is_query=True,
            batch_size=colbert_local.get_default_batch_size()
        )
        documents_embeddings = colbert_local.model.encode(
            [[chunk["content"] for chunk in res]],
            is_query=False,
            batch_size=colbert_local.get_default_batch_size()
        )


        reranked_documents = rank.rerank(
            documents_ids=[[i for i in range(len(res))]],
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
        )[0] # [0] for just one query

        reranked_documents.sort(key=itemgetter('id'))

        scores = [chunk["score"] for chunk in reranked_documents]

        scores_normal = normalize(scores)

        reranked = [{
            # Don't return content again
            # "content": chunk["content"],
            "orig_score": scores[index],
            "score": scores_normal[index],
            "passage_id": chunk["passage_id"]
        } for index, chunk in enumerate(res)]

        if reorder:
            reranked.sort(key=itemgetter('score'), reverse=True)

        return reranked

    def init_rerankers(self):
        pass


rerankers_instance = RerankersLocal()