import json
import os
import time

from litellm import completion

from src.agents.judge_agent import JudgeAgent
from src.agents.rag_agent import RAGAgent
from src.database.faiss_database import FaissDatabase
from src.datasets.dataset import LongMemEvalDataset
from src.policies.question_chunk_policy import QuestionChunkPolicy
from src.policies.save_chunk_policy import SaveChunkPolicy
from src.policies.search_chunks_policy import SearchChunksPolicy


def get_prompt(question, context):
    prompt = f"""
    You are a helpful assistant that answers a question based on the evidence.
    The evidence is: {context}
    The question is: {question}
    Return the answer to the question.
    """
    return [{"role": "user", "content": prompt}]


def run_experiment(
        model_name: str,
        embeddings_model_name: str,
        judge_model_name: str | None,
        save_chunk_policy: SaveChunkPolicy,
        question_chunk_policy: QuestionChunkPolicy,
        search_chunks_policy: SearchChunksPolicy,
        database: FaissDatabase,
        dataset_type: str,
        dataset_set: str,
):
    dataset = LongMemEvalDataset(dataset_type, dataset_set)
    results_dir = f"data/results/{dataset.dataset_set}/{dataset.dataset_type}/embeddings_{embeddings_model_name.replace('/', '_')}_memory_{model_name.replace('/', '_')}_judge_{judge_model_name.replace('/', '_')}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nResults will be saved to: {results_dir}")
    print(f"Processing samples...")
    print("=" * 100)

    rag = RAGAgent(
        embeddings_model_name,
        save_chunk_policy,
        question_chunk_policy,
        search_chunks_policy,
        database
    )
    judge = JudgeAgent(judge_model_name) if judge_model_name else None

    for instance in dataset[:]:
        result_file = f"{results_dir}/{instance.question_id}.json"

        if os.path.exists(result_file):
            print(f"Skipping {instance.question_id} because it already exists", flush=True)
            continue
        database.clear()
        rag.save_embeddings(instance.sessions)

        start_time = time.time()
        context = rag.retrieve_chunks(instance)
        prompt = get_prompt(instance.question, context)
        response = completion(model=model_name, messages=prompt)
        predicted_answer = response.choices[0].message.content
        latency = time.time() - start_time

        with open(result_file, "w", encoding="utf-8") as f:
            result = {
                "question_id": instance.question_id,
                "question": instance.question,
                "predicted_answer": predicted_answer,
                "latency": latency,
            }
            print(f"  Question: {instance.question}...")
            print(f"  Predicted: {predicted_answer}")
            print(f"  Latency: {latency}")

            if judge:
                answer_is_correct = judge.judge(instance, predicted_answer)
                result["answer"] = instance.answer
                result["answer_is_correct"] = answer_is_correct
                print(f"  Ground Truth: {instance.answer}")
                print(f"  Correct: {answer_is_correct}")

            json.dump(result, f, indent=2)
        print("-" * 100)
    print("EVALUATION COMPLETE")
