from dotenv import load_dotenv

from src.database.faiss_database import FaissDatabase
from src.experiment import run_experiment
from src.policies.save_chunk_policy import ContextualSlidingWindowSaveChunkPolicy
from src.policies.search_chunks_policy import RerankTimePruningSearchChunkPolicy

load_dotenv()

run_experiment(
    model_name="ollama/gemma3:4b",
    embeddings_model_name="Qwen/Qwen3-Embedding-0.6B",
    judge_model_name="ollama/gemma3:4b",
    save_chunk_policy=ContextualSlidingWindowSaveChunkPolicy("ollama/gemma3:4b", 256, 64),
    search_chunks_policy=RerankTimePruningSearchChunkPolicy("ollama/gemma3:4b", "BAAI/bge-reranker-v2-m3"),
    database=FaissDatabase(1024),
    dataset_type="short",
    dataset_set="longmemeval",
    top_k=5,
)
