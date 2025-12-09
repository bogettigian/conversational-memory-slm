from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

import numpy as np
from litellm import completion
from sentence_transformers import CrossEncoder

from src.database.faiss_database import FaissDatabase
from src.database.graph_database import GraphDatabase
from src.datasets.dataset import LongMemEvalInstance
from src.policies.role_classifier import RoleClassifier
from src.utils.prompt import get_date_prompt


def _time_pruning(
        chunks: list[str],
        metadata: list[dict[str, str]],
        start: datetime,
        end: datetime,
        k: int,
) -> tuple[list[str], list[dict[str, str]]]:
    in_time_range = []
    out_time_range = []
    for i, chunk in enumerate(chunks):
        date = datetime.strptime(metadata[i]["date"], "%Y/%m/%d (%a) %H:%M")
        if start < date < end:
            in_time_range.append(chunk)
        else:
            out_time_range.append(chunk)

    return in_time_range[:k] if len(in_time_range) > 0 else out_time_range[
        :k
    ], metadata[:k]


def _generate_time_range(
        date_model_name: str, dataset_instance: LongMemEvalInstance
) -> tuple[datetime, datetime] | None:
    prompt = get_date_prompt(dataset_instance.question, dataset_instance.t_question)
    response = completion(model=date_model_name, messages=prompt)
    answer = response.choices[0].message.content.strip()

    answer = answer.replace("```json", "")
    answer = answer.replace("```", "").strip()
    try:
        time_range = json.loads(answer)
    except json.JSONDecodeError:
        return None

    if not time_range:
        return None

    start = datetime.strptime(time_range["start"], "%Y/%m/%d") - timedelta(days=2)
    end = datetime.strptime(time_range["end"], "%Y/%m/%d") + timedelta(days=2)

    return start, end


def _rerank_chunks(
        reranker: CrossEncoder,
        query: str,
        chunks: list[str],
        metadata: list[dict[str, str]],
        k: int,
) -> tuple[list[str], list[dict[str, str]]]:
    results = reranker.rank(query, chunks, top_k=k, return_documents=True)
    return [result["text"] for result in results], [
        metadata[result["corpus_id"]] for result in results
    ]


class SearchChunksPolicy(ABC):
    name: str

    @abstractmethod
    def apply(
            self,
            embedding_matrix: np.ndarray[np.float32],
            database: FaissDatabase,
            dataset_instance: LongMemEvalInstance,
            graph_db: GraphDatabase | None
    ) -> tuple[list[str], list[dict[str, str]]]:
        pass


class SimpleSearchChunkPolicy(SearchChunksPolicy):
    def __init__(self, k: int, threshold: float):
        self.k = k
        self.threshold = threshold
        self.name = f"simple_search_{k}_{threshold}"

    def apply(
            self,
            embedding_matrix: np.ndarray[np.float32],
            database: FaissDatabase,
            dataset_instance: LongMemEvalInstance,
            graph_db: GraphDatabase | None
    ) -> tuple[list[str], list[dict[str, str]]]:
        chunks, metadata = database.search(
            embedding_matrix, k=self.k, threshold=self.threshold
        )
        return chunks, metadata


class RerankSearchChunkPolicy(SearchChunksPolicy):
    def __init__(self, reranker_model: str, k: int, threshold: float):
        self.reranker = CrossEncoder(reranker_model)
        self.k = k
        self.threshold = threshold
        self.name = f"rerank_search_{reranker_model.replace('/', '_')}_{k}_{threshold}"

    def apply(
            self,
            embedding_matrix: np.ndarray[np.float32],
            database: FaissDatabase,
            dataset_instance: LongMemEvalInstance,
            graph_db: GraphDatabase | None
    ) -> tuple[list[str], list[dict[str, str]]]:
        chunks, metadata = database.search(
            embedding_matrix, k=self.k * 10, threshold=self.threshold
        )
        if not chunks:
            return [], []

        return _rerank_chunks(
            self.reranker, dataset_instance.question, chunks, metadata, k=self.k
        )


class RerankRoleSearchChunkPolicy(SearchChunksPolicy):
    def __init__(self, reranker_model_name: str, role_policy: RoleClassifier, k: int, threshold: float):
        self.reranker = CrossEncoder(reranker_model_name)
        self.role_policy = role_policy
        self.k = k
        self.threshold = threshold
        self.name = f"rerank_role_search_{reranker_model_name.replace('/', '_')}_{role_policy.name}_{k}_{threshold}"

    def apply(
            self,
            embedding_matrix: np.ndarray[np.float32],
            database: FaissDatabase,
            dataset_instance: LongMemEvalInstance,
            graph_db: GraphDatabase | None
    ) -> tuple[list[str], list[dict[str, str]]]:
        role = self.role_policy.classify(dataset_instance.question)
        chunks, metadata = database.search(
            embedding_matrix, role=role, k=self.k * 10, threshold=self.threshold
        )
        if not chunks:
            return [], []

        return _rerank_chunks(
            self.reranker, dataset_instance.question, chunks, metadata, k=self.k
        )


class RerankTimePruningSearchChunkPolicy(SearchChunksPolicy):
    def __init__(
            self,
            date_model_name: str,
            reranker_model_name: str,
            k: int,
            threshold: float,
    ):
        self.date_model_name = date_model_name
        self.reranker = CrossEncoder(reranker_model_name)
        self.k = k
        self.threshold = threshold
        self.name = f"rerank_time_pruning_search_{date_model_name.replace('/', '_')}_{reranker_model_name.replace('/', '_')}_{k}_{threshold}"

    def apply(
            self,
            embedding_matrix: np.ndarray[np.float32],
            database: FaissDatabase,
            dataset_instance: LongMemEvalInstance,
            graph_db: GraphDatabase | None
    ) -> tuple[list[str], list[dict[str, str]]]:
        chunks, metadata = database.search(
            embedding_matrix, k=self.k * 10, threshold=self.threshold
        )
        if not chunks:
            return [], []

        time_range = _generate_time_range(self.date_model_name, dataset_instance)
        if not time_range:
            return _rerank_chunks(
                self.reranker, dataset_instance.question, chunks, metadata, k=self.k
            )

        pruned_chunks, metadata = _time_pruning(
            chunks, metadata, time_range[0], time_range[1], self.k * 10
        )
        return _rerank_chunks(
            self.reranker, dataset_instance.question, pruned_chunks, metadata, k=self.k
        )


class RerankRoleGraphSearchChunkPolicy(SearchChunksPolicy):
    def __init__(self, reranker_model_name: str, role_policy: RoleClassifier, k: int,
                 threshold: float):
        self.reranker = CrossEncoder(reranker_model_name)
        self.role_policy = role_policy
        self.k = k
        self.threshold = threshold
        self.name = f"rerank_role_graph_search_{reranker_model_name.replace('/', '_')}_{role_policy.name}_{k}_{threshold}"

    def apply(
            self,
            embedding_matrix: np.ndarray[np.float32],
            database: FaissDatabase,
            dataset_instance: LongMemEvalInstance,
            graph_db: GraphDatabase | None
    ) -> tuple[list[str], list[dict[str, str]]]:
        role = self.role_policy.classify(dataset_instance.question)
        chunks, metadata = database.search(
            embedding_matrix, role=role, k=self.k * 5, threshold=self.threshold
        )
        if not chunks:
            return [], []

        chunks, metadata = graph_db.search_nodes([m['id'] for m in metadata])

        return _rerank_chunks(
            self.reranker, dataset_instance.question, chunks, metadata, k=self.k
        )
