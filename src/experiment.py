import json
import os
import time

from litellm import completion

from src.agents.judge_agent import JudgeAgent
from src.agents.rag_agent import RAGAgent
from src.database.faiss_database import FaissDatabase
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
        database: FaissDatabase,
        dataset_type: str,
        dataset_set: str,
        limit: int = None,
):
    dataset = LongMemEvalDataset(dataset_type, dataset_set)
    results_dir = f"data/results/{dataset.dataset_set}/{dataset.dataset_type}/EMB_{embeddings_model_name.replace('/', '_')}_MODEL_{model_name.replace('/', '_')}_SAVE_{save_chunk_policy.name}_SEARCH_{search_chunks_policy.name}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nResults will be saved to: {results_dir}")
    print(f"Processing samples...")
    print("=" * 100)

    rag = RAGAgent(
        embeddings_model_name,
        save_chunk_policy,
        search_chunks_policy,
        database
    )
    judge = JudgeAgent(judge_model_name) if judge_model_name else None

    for instance in dataset[:limit]:
        result_file = f"{results_dir}/{instance.question_id}.json"

        if os.path.exists(result_file):
            print(f"Skipping {instance.question_id} because it already exists", flush=True)
            continue
        database.clear()
        rag.save_embeddings(instance.sessions)

        start_time = time.time()
        chunks = rag.retrieve_chunks(instance)
        prompt = get_prompt(instance.question, chunks)
        response = completion(model=model_name, messages=prompt)
        predicted_answer = response.choices[0].message.content
        latency = time.time() - start_time

        with open(result_file, "w", encoding="utf-8") as file:
            result = {
                "question_id": instance.question_id,
                "question": instance.question,
                "predicted_answer": predicted_answer,
                "latency": latency,
                "context_length": len(prompt[0]["content"]),
            }
            print(f"  Question: {instance.question}...")
            print(f"  Predicted: {predicted_answer}")
            print(f"  Latency: {latency}")
            print(f"  Context length: {len(prompt[0]["content"])}")

            if judge:
                answer_is_correct = judge.judge(instance, predicted_answer)
                result["answer"] = instance.answer
                result["answer_is_correct"] = answer_is_correct
                print(f"  Ground Truth: {instance.answer}")
                print(f"  Correct: {answer_is_correct}")

            json.dump(result, file, indent=2)
        print("-" * 100)
    metric_generator(results_dir)
    print("EVALUATION COMPLETE")
