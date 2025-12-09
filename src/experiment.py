import json
import os
import time

from litellm import completion

from src.agents.judge_agent import JudgeAgent
from src.agents.rag_agent import RAGAgent
from src.database.faiss_database import FaissDatabase
from src.database.graph_database import GraphDatabase
from src.datasets.dataset import LongMemEvalDataset
from src.policies.save_chunk_policy import SaveChunkPolicy
from src.policies.search_chunks_policy import SearchChunksPolicy
from src.utils.metrics import metric_generator
from src.utils.prompt import get_prompt


def run_experiment(
        model_name: str,
        embeddings_model_name: str,
        judge_model_name: str | None,
        save_chunk_policy: SaveChunkPolicy,
        search_chunks_policy: SearchChunksPolicy,
        vector_database: FaissDatabase,
        graph_database: GraphDatabase | None,
        dataset_type: str,
        dataset_set: str,
        top_k: int,
        limit: int | None = None,
):
    dataset = LongMemEvalDataset(dataset_type, dataset_set)
    results_dir = (f"data/results/{dataset.dataset_set}/{dataset.dataset_type}/"
                   f"MODEL_{model_name.replace('/', '_')}/"
                   f"EMB_{embeddings_model_name.replace('/', '_')}/"
                   f"SAVE_{save_chunk_policy.name}/"
                   f"SEARCH_{search_chunks_policy.name}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nResults will be saved to: {results_dir}")
    print(f"Processing samples...")
    print("=" * 100)

    rag = RAGAgent(
        embeddings_model_name,
        save_chunk_policy,
        search_chunks_policy,
        vector_database,
        graph_database
    )
    judge = JudgeAgent(judge_model_name) if judge_model_name else None

    for instance in dataset[:limit]:
        result_file = f"{results_dir}/{instance.question_id}.json"

        if os.path.exists(result_file):
            print(f"Skipping {instance.question_id} because it already exists", flush=True)
            continue
        vector_database.clear()
        if graph_database:
            graph_database.clear()

        start_time = time.time()
        rag.save_embeddings(instance.sessions)
        offline_latency = time.time() - start_time

        best_result = None
        for attempt in range(1, top_k + 1):
            start_time = time.time()
            chunks, metadata = rag.retrieve_chunks(instance)
            prompt = get_prompt(instance, chunks, metadata)
            response = completion(model=model_name, messages=prompt)
            predicted_answer = response.choices[0].message.content
            online_latency = time.time() - start_time

            result = {
                "question_id": instance.question_id,
                "question": instance.question,
                "question_type": instance.question_type,
                "predicted_answer": predicted_answer,
                "online_latency": online_latency,
                "offline_latency": offline_latency,
                "context_length": len(prompt[0]["content"]),
                "attempts": attempt,
            }

            print(f"  Attempt {attempt}/{top_k}")
            print(f"  Question: {instance.question}...")
            print(f"  Question type: {instance.question_type}...")
            print(f"  Predicted: {predicted_answer}")
            print(f"  Online Latency: {online_latency}")
            print(f"  Offline Latency: {offline_latency}")
            print(f"  Context length: {len(prompt[0]['content'])}")

            if judge:
                answer_is_correct = judge.judge(instance, predicted_answer)
                result["answer"] = instance.answer
                result["answer_is_correct"] = answer_is_correct
                print(f"  Ground Truth: {instance.answer}")
                print(f"  Correct: {answer_is_correct}")

                if answer_is_correct:
                    best_result = result
                    print(f"  ✓ Correct answer found on attempt {attempt}")
                    break
                else:
                    best_result = result
            else:
                best_result = result
                break  # No judge, so no point in retrying

        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(best_result, file, indent=2)
        print("-" * 100)
    metric_generator(results_dir)
    print("EVALUATION COMPLETE")
