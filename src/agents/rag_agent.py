import numpy as np
from sentence_transformers import SentenceTransformer

from src.database.faiss_database import FaissDatabase
from src.datasets.dataset import LongMemEvalInstance, Session
from src.policies.save_chunk_policy import SaveChunkPolicy
from src.policies.search_chunks_policy import SearchChunksPolicy


class RAGAgent:
    def __init__(
            self,
            embeddings_model_name: str,
            save_chunk_policy: SaveChunkPolicy,
            search_chunks_policy: SearchChunksPolicy,
            db: FaissDatabase,
    ):
        self.embedding_model = SentenceTransformer(embeddings_model_name, trust_remote_code=True)
        self.save_chunk_policy = save_chunk_policy
        self.search_chunks_policy = search_chunks_policy
        self.db = db

    def save_embeddings(self, session_history: list[Session]) -> None:
        chunks, metadata = self.save_chunk_policy.apply(session_history)
        embeddings_matrix = self.embedding_model.encode(chunks, convert_to_numpy=True).astype(np.float32)
        self.db.insert_embeddings(embeddings_matrix, chunks, metadata)

    def retrieve_chunks(self, dataset_instance: LongMemEvalInstance, role: str) -> tuple[list[str], list[dict[str, str]]]:
        question_embedding = self.embedding_model.encode(dataset_instance.question, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
        chunks, metadata = self.search_chunks_policy.apply(question_embedding, self.db, dataset_instance, role)
        return chunks, metadata
