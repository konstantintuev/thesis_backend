import gc
import inspect
import logging
import threading
import time
import uuid
from collections import OrderedDict
from typing import List

import torch
from FlagEmbedding.bge_m3 import BGEM3FlagModel
from langchain_core.embeddings import Embeddings

from file_processing.gpu_utils import get_batch_size

model_name = "BAAI/bge-m3"

logger = logging.getLogger(__name__)

class BGEM3Flag(Embeddings):
    def __init__(self):
        """
        https://huggingface.co/BAAI/bge-m3
        """
        self.model = BGEM3FlagModel('BAAI/bge-m3',
                                    use_fp16=True)

    def get_default_batch_size(self):
        # Twice as memory intensive as colbert
        return round(get_batch_size() / 2)

    def embed_documents(self, *args, **kwargs) -> List[List[float]]:
        """
        Embed search docs using flexible arguments.

        Expected usage:
        - embed_documents(texts, gpu_batch_size=default_batch_size)
        - embed_documents(texts)
        """
        # Extract texts and gpu_batch_size from args and kwargs
        texts = kwargs.get('texts')
        gpu_batch_size = kwargs.get('gpu_batch_size', self.get_default_batch_size())

        # Handle case when texts is passed as the first positional argument
        if len(args) > 0:
            texts = args[0]

        if texts is None:
            raise ValueError("The 'texts' argument is required.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Use the extracted parameters to embed documents
        return self.model.encode(texts, batch_size=gpu_batch_size)['dense_vecs'].tolist()

    def embed_query(self, *args, **kwargs) -> List[float]:
        """
        Embed a single query using flexible arguments.

        Expected usage:
        - embed_query(text, gpu_batch_size=default_batch_size)
        - embed_query(text)
        - embed_query(text="query", gpu_batch_size=default_batch_size)
        """
        # Extract text and gpu_batch_size from args and kwargs
        text = kwargs.get('text')
        gpu_batch_size = kwargs.get('gpu_batch_size', self.get_default_batch_size())

        # Handle case when text is passed as the first positional argument
        if len(args) > 0:
            text = args[0]

        if text is None:
            raise ValueError("The 'text' argument is required.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Use the extracted parameters to embed the query
        return self.model.encode([text], batch_size=gpu_batch_size)['dense_vecs'][0]


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
    def __init__(self, initial_interval: int = 10, subsequent_interval: int = 5):
        self.initial_interval = initial_interval
        self.subsequent_interval = subsequent_interval
        self.lock = threading.Lock()
        self.pending_requests = OrderedDict()
        self.timer = TimerReset(self.initial_interval, self._process_queue)
        self.timer.start()

    def init_mode(self, model: Embeddings):
        self.model = model

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
            tries = 0
            gpu_batch_size = self.model.get_default_batch_size() if hasattr(self.model, 'get_default_batch_size') else get_batch_size()
            emb_res = self.actual_embedding_process(all_texts, request_ids, texts, gpu_batch_size)
            while not emb_res:
                logging.error(f"Retry pending embedding: {tries}")
                tries += 1
                if emb_res is None:
                    # GPU VRAM Size ERROR
                    gpu_batch_size -= 1
                if gpu_batch_size <= 0:
                    gpu_batch_size = 1
                    # We tried to go below 1 for batch size -> wait out the other operations (5s)
                    time.sleep(5)

                emb_res = self.actual_embedding_process(all_texts, request_ids, texts, gpu_batch_size)

        self.timer.reset(self.subsequent_interval)

    def actual_embedding_process(self, all_texts, request_ids, texts, gpu_batch_size):
        try:
            sig = inspect.signature(self.model.embed_documents)
            embeddings = self.model.embed_documents(all_texts, gpu_batch_size=gpu_batch_size) \
                if len(sig.parameters) >= 2 else self.model.embed_documents(all_texts)

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
            return True
        except torch.OutOfMemoryError as e:
            logging.error(f"Pending embeddings error occurred, retrying -> {repr(e)}",)
            return None
        except BaseException as e:
            logging.error(f"Pending embeddings error occurred, retrying -> {repr(e)}")
            if "Invalid buffer size" in repr(e):
                return None

            return False

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

pending_embeddings_singleton = PendingLangchainEmbeddings()
