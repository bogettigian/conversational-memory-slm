from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

import numpy as np
from litellm import completion

from src.database.faiss_database import FaissDatabase
from src.datasets.dataset import LongMemEvalInstance
from src.utils.prompt import get_date_prompt


class SearchChunksPolicy(ABC):
    name: str

    @abstractmethod
    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase, dataset_instance: LongMemEvalInstance) -> list[str]:
        """Search the database for the most relevant chunks to the question digest embedding.
        """
        pass


class SimpleSearchChunkPolicy(SearchChunksPolicy):
    name = "simple_search"

    def __init__(self, k: int = 5, threshold: float = 0.6):
        self.k = k
        self.threshold = threshold

    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase, dataset_instance: LongMemEvalInstance) -> list[str]:
        result = database.search(embedding_matrix, k=self.k, threshold=self.threshold)
        return result[0]


class TimePruningSearchChunkPolicy(SearchChunksPolicy):
    name = "time_pruning"

    def __init__(self, date_model_name: str, k: int = 5, threshold: float = 0.4):
        self.date_model_name = date_model_name
        self.k = k
        self.threshold = threshold

    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase, dataset_instance: LongMemEvalInstance) -> list[str]:
        prompt = get_date_prompt(dataset_instance.question, dataset_instance.t_question)
        response = completion(model=self.date_model_name, messages=prompt)
        answer = response.choices[0].message.content.strip()

        answer = answer.replace('```json', '')
        answer = answer.replace('```', '').strip()
        time_range = json.loads(answer.strip())

        result = database.search(embedding_matrix, k=self.k*2, threshold=self.threshold)

        if not time_range:
            return result[0][:self.k]

        start = datetime.strptime(time_range['start'], "%Y/%m/%d") - timedelta(days=2)
        end = datetime.strptime(time_range['end'], "%Y/%m/%d") + timedelta(days=2)

        in_time_range = []
        out_time_range = []
        for i, chunk in enumerate(result[0]):
            date = datetime.strptime(result[1][i]["date"], "%Y/%m/%d (%a) %H:%M")
            if start < date < end:
                in_time_range.append(chunk)
            else:
                out_time_range.append(chunk)

        return in_time_range[:self.k] if len(in_time_range) > 0 else out_time_range[:self.k]
