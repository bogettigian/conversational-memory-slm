from dotenv import load_dotenv

from src.database.faiss_database import FaissDatabase
from src.experiment import run_experiment
from src.policies.save_chunk_policy import SlidingWindowSaveChunkPolicy
from src.policies.search_chunks_policy import RerankTimePruningSearchChunkPolicy

load_dotenv()

run_experiment(
    model_name="ollama/gemma3:4b",
    embeddings_model_name="ollama/nomic-embed-text",
    judge_model_name="ollama/gemma3:4b",
    save_chunk_policy=SlidingWindowSaveChunkPolicy(256, 64),
    search_chunks_policy=RerankTimePruningSearchChunkPolicy("ollama/gemma3:4b", "BAAI/bge-reranker-v2-m3"),
    database=FaissDatabase(768),
    dataset_type="short",
    dataset_set="longmemeval",
    limit=10
)
