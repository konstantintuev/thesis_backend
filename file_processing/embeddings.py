import threading
import time
import uuid
from collections import OrderedDict
from typing import List

import psutil
import torch.backends.mps
from FlagEmbedding.bge_m3 import BGEM3FlagModel
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings

model_name = "BAAI/bge-m3"


class BGEM3Flag(Embeddings):
    def __init__(self):
        """
        https://huggingface.co/BAAI/bge-m3
        """
        self.model = BGEM3FlagModel('BAAI/bge-m3',
                                    use_fp16=True)

    def get_batch_size(self):
        if torch.backends.mps.is_available():
            # My mac with 8 GB of VRAM handles so much
            return 12
        # Else we do CPU
        return 4

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.model.encode(texts, batch_size=self.get_batch_size())['dense_vecs'].tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.encode(text, batch_size=self.get_batch_size())['dense_vecs'].tolist()

embeddings_model = BGEM3Flag()

"""
model_kwargs = {"device": "mps"}
encode_kwargs = {"normalize_embeddings": True}
HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)"""

"""Inspired by: https://gist.github.com/aeroaks/ac4dbed9c184607a330c"""


def TimerReset(*args, **kwargs):
    """ Global function for Timer """
    return _TimerReset(*args, **kwargs)


class _TimerReset(threading.Thread):
    """Call a function after a specified number of seconds:

    t = TimerReset(30.0, f, args=[], kwargs={})
    t.start() - to start the timer
    t.reset() - to reset the timer
    t.cancel() # stop the timer's action if it's still waiting
    """

    def __init__(self, interval, function, args=[], kwargs={}):
        threading.Thread.__init__(self)
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.finished = threading.Event()
        self.resetted = True

    def cancel(self):
        """Stop the timer if it hasn't finished yet"""
        self.finished.set()

    def run(self):
        while not self.finished.is_set():
            self.finished.wait(self.interval)
            self.function(*self.args, **self.kwargs)

    def reset(self, interval=None):
        """ Reset the timer """
        if interval:
            self.interval = interval

        self.resetted = True
        self.finished.clear()


class PendingLangchainEmbeddings(Embeddings):
    def __init__(self, model: Embeddings, initial_interval: int = 10, subsequent_interval: int = 5):
        self.model = model
        self.initial_interval = initial_interval
        self.subsequent_interval = subsequent_interval
        self.lock = threading.Lock()
        self.pending_requests = OrderedDict()
        self.timer = TimerReset(self.initial_interval, self._process_queue)
        self.timer.start()

    def _process_queue(self):
        with self.lock:
            if self.pending_requests:
                request_ids = list(self.pending_requests.keys())
                texts = [self.pending_requests[rid]["texts"] for rid in request_ids]
                all_texts = [text for sublist in texts for text in sublist]
            else:
                request_ids = []
                texts = []
                all_texts = []

        if all_texts:
            embeddings = self.model.embed_documents(all_texts)
            idx = 0
            results = {}
            for rid, text_list in zip(request_ids, texts):
                result = embeddings[idx:idx + len(text_list)]
                idx += len(text_list)
                results[rid] = result

            with self.lock:
                for rid in request_ids:
                    if rid in self.pending_requests:
                        self.pending_requests[rid]["result"] = results[rid]
                        with self.pending_requests[rid]["cv"]:
                            self.pending_requests[rid]["cv"].notify_all()

        self.timer.reset(self.subsequent_interval)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        request_id = str(uuid.uuid4())
        cv = threading.Condition()
        with self.lock:
            self.pending_requests[request_id] = {"texts": texts, "cv": cv}
            self.timer.reset(self.subsequent_interval)

        with cv:
            while "result" not in self.pending_requests[request_id]:
                cv.wait()
            result = self.pending_requests.pop(request_id)
            result_texts = result["texts"]
            if result_texts != texts:
                raise ValueError("Mismatch between input texts and result texts.")
            embeddings = result["result"]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.embed_query(text)


pending_embeddings_singleton = PendingLangchainEmbeddings(embeddings_model)
