import numpy as np
from litellm import embedding

from src.database.faiss_database import FaissDatabase
from src.datasets.dataset import LongMemEvalInstance, Session
from src.policies.question_chunk_policy import QuestionChunkPolicy
from src.policies.save_chunk_policy import SaveChunkPolicy
from src.policies.search_chunks_policy import SearchChunksPolicy


class RAGAgent:
    def __init__(
            self,
            embeddings_model_name: str,
            save_chunk_policy: SaveChunkPolicy,
            question_chunk_policy: QuestionChunkPolicy,
            search_chunks_policy: SearchChunksPolicy,
            db: FaissDatabase,
    ):
        self.embeddings_model_name = embeddings_model_name
        self.save_chunk_policy = save_chunk_policy
        self.question_chunk_policy = question_chunk_policy
        self.search_chunks_policy = search_chunks_policy
        self.db = db

    def save_embeddings(self, session_history: list[Session]) -> None:
        chunks = self.save_chunk_policy.apply(session_history)
        embeddings = embedding(model=self.embeddings_model_name, input=chunks)
        embeddings_matrix = RAGAgent.convert_to_embeddings_matrix(embeddings["data"])
        self.db.insert_embeddings(embeddings_matrix, chunks)

    def retrieve_chunks(self, dataset_instance: LongMemEvalInstance) -> list[dict[str, str]]:
        question_chunk = self.question_chunk_policy.apply(dataset_instance)
        question_embedding = embedding(model=self.embeddings_model_name, input=question_chunk)
        embeddings_matrix = RAGAgent.convert_to_embeddings_matrix(question_embedding["data"])
        chunks = self.search_chunks_policy.apply(embeddings_matrix, self.db)
        return chunks

    @staticmethod
    def convert_to_embeddings_matrix(embeddings: list[dict[str, str]]) -> np.ndarray[np.float32]:
        return np.array([item['embedding'] for item in embeddings], dtype=np.float32)
