from abc import ABC, abstractmethod

from src.datasets.dataset import LongMemEvalInstance


QUERY_PREFIX = "search_query: "

class QuestionChunkPolicy(ABC):
    @abstractmethod
    def apply(self, dataset_instance: LongMemEvalInstance) -> str:
        """Reformulate question + session history to a chunk of text that will be embedded and used to search for
        the most relevant chunks in the database.
        """
        pass


class SimpleQuestionChunkPolicy(QuestionChunkPolicy):
    def apply(self, dataset_instance: LongMemEvalInstance) -> str:
        return f"{QUERY_PREFIX}{dataset_instance.question}"
