from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

import numpy as np
from litellm import completion
from sentence_transformers import CrossEncoder

from src.database.faiss_database import FaissDatabase
from src.datasets.dataset import LongMemEvalInstance
from src.utils.prompt import get_date_prompt


def _time_pruning(chunks, metadata, start, end, k):
    in_time_range = []
    out_time_range = []
    for i, chunk in enumerate(chunks):
        date = datetime.strptime(metadata[i]["date"], "%Y/%m/%d (%a) %H:%M")
        if start < date < end:
            in_time_range.append(chunk)
        else:
            out_time_range.append(chunk)

    return in_time_range[:k] if len(in_time_range) > 0 else out_time_range[:k]


def _generate_time_range(date_model_name: str, dataset_instance: LongMemEvalInstance) -> tuple[
                                                                                             datetime, datetime] | None:
    prompt = get_date_prompt(dataset_instance.question, dataset_instance.t_question)
    response = completion(model=date_model_name, messages=prompt)
    answer = response.choices[0].message.content.strip()

    answer = answer.replace('```json', '')
    answer = answer.replace('```', '').strip()
    try:
        time_range = json.loads(answer)
    except json.JSONDecodeError:
        return None

    if not time_range:
        return None

    start = datetime.strptime(time_range['start'], "%Y/%m/%d") - timedelta(days=2)
    end = datetime.strptime(time_range['end'], "%Y/%m/%d") + timedelta(days=2)

    return start, end


def _rerank_chunks(reranker: CrossEncoder, query: str, chunks: list[str], k: int) -> list[str]:
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    scored_chunks = [(score, chunk) for score, chunk in zip(scores, chunks)]
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [chunk for _, chunk in scored_chunks[:k]]


class SearchChunksPolicy(ABC):
    name: str

    @abstractmethod
    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase,
              dataset_instance: LongMemEvalInstance) -> list[str]:
        pass


class SimpleSearchChunkPolicy(SearchChunksPolicy):
    name = "simple_search"

    def __init__(self, k: int = 5, threshold: float = 0.6):
        self.k = k
        self.threshold = threshold

    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase, dataset_instance: LongMemEvalInstance) -> list[str]:
        chunks, _ = database.search(embedding_matrix, k=self.k, threshold=self.threshold)
        return chunks


class TimePruningSearchChunkPolicy(SearchChunksPolicy):
    name = "time_pruning_search"

    def __init__(self, date_model_name: str, k: int = 5, threshold: float = 0.3):
        self.date_model_name = date_model_name
        self.k = k
        self.threshold = threshold

    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase, dataset_instance: LongMemEvalInstance) -> list[str]:
        chunks, metadata = database.search(embedding_matrix, k=self.k * 2, threshold=self.threshold)
        if not chunks:
            return []

        time_range = _generate_time_range(self.date_model_name, dataset_instance)
        if not time_range:
            return chunks[:self.k]

        return _time_pruning(chunks, metadata, time_range[0], time_range[1], self.k)


class RerankSearchChunkPolicy(SearchChunksPolicy):
    name = "rerank_search"

    def __init__(self, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2", k: int = 3, threshold: float = 0.3):
        self.reranker = CrossEncoder(reranker_model)
        self.k = k
        self.threshold = threshold

    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase, dataset_instance: LongMemEvalInstance) -> list[str]:
        chunks, _ = database.search(embedding_matrix, k=self.k * 10, threshold=self.threshold)
        if not chunks:
            return []

        return _rerank_chunks(self.reranker, dataset_instance.question, chunks, k=self.k)


class RerankTimePruningSearchChunkPolicy(SearchChunksPolicy):
    name = "rerank_time_pruning_search"

    def __init__(self, date_model_name: str, reranker_model_name: str, k: int = 3, threshold: float = 0.3):
        self.date_model_name = date_model_name
        self.reranker = CrossEncoder(reranker_model_name)
        self.k = k
        self.threshold = threshold

    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase, dataset_instance: LongMemEvalInstance) -> list[str]:
        chunks, metadata = database.search(embedding_matrix, k=self.k * 10, threshold=self.threshold)
        if not chunks:
            return []

        time_range = _generate_time_range(self.date_model_name, dataset_instance)
        if not time_range:
            return _rerank_chunks(self.reranker, dataset_instance.question, chunks, k=self.k)

        pruned_chunks = _time_pruning(chunks, metadata, time_range[0], time_range[1], self.k * 10)
        return _rerank_chunks(self.reranker, dataset_instance.question, pruned_chunks, k=self.k)
