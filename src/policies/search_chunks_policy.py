from abc import ABC, abstractmethod

import numpy as np

from src.database.faiss_database import FaissDatabase


class SearchChunksPolicy(ABC):
    name: str
    @abstractmethod
    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase) -> list[str]:
        """Search the database for the most relevant chunks to the question digest embedding.
        """
        pass


class SimpleSearchChunkPolicy(SearchChunksPolicy):
    name = "simple_search"

    def apply(self, embedding_matrix: np.ndarray[np.float32], database: FaissDatabase) -> list[str]:
        result = database.search(embedding_matrix, k=5)
        return result
