from dotenv import load_dotenv

from src.database.faiss_database import FaissDatabase
from src.database.graph_database import GraphDatabase
from src.experiment import run_experiment
from src.policies.role_classifier import FineTuningClassifier
from src.policies.save_chunk_policy import SlidingWindowSaveChunkPolicy
from src.policies.search_chunks_policy import RerankRoleSearchChunkPolicy, SimpleSearchChunkPolicy, \
    RerankSearchChunkPolicy, RerankTimePruningSearchChunkPolicy, RerankRoleGraphSearchChunkPolicy

load_dotenv()

# ###################
# ### Iteration 1 ###
# ###################
#
# embeddings_model_name = "google/embeddinggemma-300m"
#
# run_experiment(
#     model_name="ollama/gemma3:4b",
#     embeddings_model_name=embeddings_model_name,
#     judge_model_name="openai/gpt-5-mini",
#     save_chunk_policy=SlidingWindowSaveChunkPolicy(256, 64, embeddings_model_name),
#     search_chunks_policy=SimpleSearchChunkPolicy(6, 0.3),
#     vector_database=FaissDatabase(768),
#     graph_database=None,
#     dataset_type="short",
#     dataset_set="investigathon_evaluation",
#     top_k=3,
# )
#
# ###################
# ### Iteration 2 ###
# ###################
#
# embeddings_model_name = "google/embeddinggemma-300m"
#
# run_experiment(
#     model_name="ollama/gemma3:4b",
#     embeddings_model_name=embeddings_model_name,
#     judge_model_name="openai/gpt-5-mini",
#     save_chunk_policy=SlidingWindowSaveChunkPolicy(256, 64, embeddings_model_name),
#     search_chunks_policy=RerankSearchChunkPolicy("BAAI/bge-reranker-v2-m3", 6, 0.3),
#     vector_database=FaissDatabase(768),
#     graph_database=None,
#     dataset_type="short",
#     dataset_set="investigathon_evaluation",
#     top_k=3,
# )
#
# ###################
# ### Iteration 3 ###
# ###################
#
# embeddings_model_name = "google/embeddinggemma-300m"
#
# run_experiment(
#     model_name="ollama/gemma3:4b",
#     embeddings_model_name=embeddings_model_name,
#     judge_model_name="openai/gpt-5-mini",
#     save_chunk_policy=SlidingWindowSaveChunkPolicy(256, 64, embeddings_model_name),
#     search_chunks_policy=RerankTimePruningSearchChunkPolicy("ollama/gemma3:4b", "BAAI/bge-reranker-v2-m3", 6, 0.3),
#     vector_database=FaissDatabase(768),
#     graph_database=None,
#     dataset_type="short",
#     dataset_set="investigathon_evaluation",
#     top_k=3,
# )
#
# ###################
# ### Iteration 4 ###
# ###################
#
# embeddings_model_name = "google/embeddinggemma-300m"
# role_policy = FineTuningClassifier("./models/fine_tuned_classification_model")
#
# run_experiment(
#     model_name="ollama/gemma3:4b",
#     embeddings_model_name=embeddings_model_name,
#     judge_model_name="openai/gpt-5-mini",
#     save_chunk_policy=SlidingWindowSaveChunkPolicy(256, 64, embeddings_model_name),
#     search_chunks_policy=RerankRoleSearchChunkPolicy("BAAI/bge-reranker-v2-m3", role_policy, 6, 0.3),
#     vector_database=FaissDatabase(768),
#     graph_database=None,
#     dataset_type="short",
#     dataset_set="investigathon_evaluation",
#     top_k=3,
# )

###################
### Iteration 5 ###
###################

embeddings_model_name = "google/embeddinggemma-300m"
role_policy = FineTuningClassifier("./models/fine_tuned_classification_model")
graph_db = GraphDatabase(embeddings_model_name, FaissDatabase(768))

run_experiment(
    model_name="ollama/gemma3:4b",
    embeddings_model_name=embeddings_model_name,
    judge_model_name="openai/gpt-5-mini",
    save_chunk_policy=SlidingWindowSaveChunkPolicy(256, 64, embeddings_model_name),
    search_chunks_policy=RerankRoleGraphSearchChunkPolicy("BAAI/bge-reranker-v2-m3", role_policy, 6, 0.3),
    vector_database=FaissDatabase(768),
    graph_database=graph_db,
    dataset_type="short",
    dataset_set="investigathon_evaluation",
    top_k=3,
)

