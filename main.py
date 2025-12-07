from dotenv import load_dotenv

from src.database.faiss_database import FaissDatabase
from src.experiment import run_experiment
from src.policies.role_classifier import FineTuningClassifier
from src.policies.save_chunk_policy import SlidingWindowSaveChunkPolicy
from src.policies.search_chunks_policy import RerankRoleSearchChunkPolicy

load_dotenv()

embeddings_model_name = "google/embeddinggemma-300m"
role_policy = FineTuningClassifier("./models/fine_tuned_classification_model")

run_experiment(
    model_name="ollama/gemma3:4b",
    embeddings_model_name=embeddings_model_name,
    judge_model_name="openai/gpt-5-nano",  # TODO: change to mini later on
    save_chunk_policy=SlidingWindowSaveChunkPolicy(256, 64, embeddings_model_name),
    search_chunks_policy=RerankRoleSearchChunkPolicy("BAAI/bge-reranker-v2-m3", role_policy, 6, 0.3),
    database=FaissDatabase(768),
    dataset_type="short",
    dataset_set="longmemeval",
    top_k=3,
)
