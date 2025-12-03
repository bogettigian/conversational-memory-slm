from dotenv import load_dotenv

from src.database.faiss_database import FaissDatabase
from src.experiment import run_experiment
from src.policies.save_chunk_policy import SlidingWindowSaveChunkPolicy
from src.policies.search_chunks_policy import TimePruningSearchChunkPolicy

load_dotenv()

run_experiment(
    model_name="ollama/gemma3:4b",
    embeddings_model_name="ollama/nomic-embed-text",
    judge_model_name="ollama/gemma3:4b",
    save_chunk_policy=SlidingWindowSaveChunkPolicy(256, 64),
    search_chunks_policy=TimePruningSearchChunkPolicy("ollama/gemma3:4b", 10,0.2),
    database=FaissDatabase(768),
    dataset_type="short",
    dataset_set="longmemeval",
    limit=10
)
