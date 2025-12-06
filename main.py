from dotenv import load_dotenv

from src.database.faiss_database import FaissDatabase
from src.experiment import run_experiment
from src.policies.save_chunk_policy import SlidingWindowSaveChunkPolicy
from src.policies.search_chunks_policy import RerankSearchChunkPolicy

load_dotenv()

run_experiment(
    model_name="ollama/gemma3:4b",
    embeddings_model_name="google/embeddinggemma-300m",
    judge_model_name="openai/gpt-5-nano",  # TODO: change to mini later on
    save_chunk_policy=SlidingWindowSaveChunkPolicy(256, 64),
    search_chunks_policy=RerankSearchChunkPolicy("BAAI/bge-reranker-v2-m3", 4),
    database=FaissDatabase(768),
    dataset_type="short",
    dataset_set="longmemeval",
    top_k=5,
)
