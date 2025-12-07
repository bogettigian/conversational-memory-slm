"""
Build a dataset with only the questions answered incorrectly in an experiment.

Example:
    python scripts/create_wrong_questions_dataset.py \\
        --results-dir data/results/longmemeval/short/EMB_x_MODEL_y_SAVE_z_SEARCH_w \\
        --dataset-type short \\
        --dataset-set longmemeval \\
        --output wrong_questions_longmemeval_short.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


VALID_TYPES = {"oracle", "short"}
VALID_SETS = {"longmemeval", "investigathon_evaluation", "investigathon_held_out"}


def resolve_dataset_path(dataset_type: str, dataset_set: str) -> Path:
    if dataset_type not in VALID_TYPES:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Must be one of {sorted(VALID_TYPES)}")
    if dataset_set not in VALID_SETS:
        raise ValueError(f"Invalid dataset set: {dataset_set}. Must be one of {sorted(VALID_SETS)}")

    if dataset_set == "longmemeval":
        path = {
            "oracle": "data/longmemeval/longmemeval_oracle.json",
            "short": "data/longmemeval/longmemeval_s_cleaned.json",
        }[dataset_type]
    elif dataset_set == "investigathon_evaluation":
        path = {
            "oracle": "data/investigathon/Investigathon_LLMTrack_Evaluation_oracle.json",
            "short": "data/investigathon/Investigathon_LLMTrack_Evaluation_s_cleaned.json",
        }[dataset_type]
    else:  # investigathon_held_out
        if dataset_type != "short":
            raise ValueError("Held-out set is only available with dataset_type='short'")
        path = "data/investigathon/Investigathon_LLMTrack_HeldOut_s_cleaned.json"

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    return dataset_path


def load_dataset_map(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["question_id"]: item for item in data}


def collect_wrong_results(results_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    missing_correct_flag: List[str] = []
    wrong_results: List[Dict[str, Any]] = []

    for result_file in results_dir.glob("*.json"):
        if result_file.name == "metrics.json":
            continue
        try:
            with result_file.open("r", encoding="utf-8") as f:
                result = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping non-JSON file: {result_file}")
            continue

        if "answer_is_correct" not in result:
            missing_correct_flag.append(result_file.name)
            continue

        if result.get("answer_is_correct") is False:
            wrong_results.append(result)

    return wrong_results, missing_correct_flag


def build_wrong_dataset(
    wrong_results: List[Dict[str, Any]], dataset_map: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    missing_in_dataset: List[str] = []
    wrong_dataset: List[Dict[str, Any]] = []

    for result in wrong_results:
        qid = result.get("question_id")
        if qid is None:
            continue

        base_entry = dataset_map.get(qid)
        if not base_entry:
            missing_in_dataset.append(qid)
            continue

        entry = dict(base_entry)
        entry.update(
            {
                "predicted_answer": result.get("predicted_answer"),
                "answer_is_correct": result.get("answer_is_correct", False),
                "attempts": result.get("attempts"),
                "online_latency": result.get("online_latency"),
                "offline_latency": result.get("offline_latency"),
                "context_length": result.get("context_length"),
                "experiment_result_path": str(result.get("result_file_path", "")),
            }
        )
        wrong_dataset.append(entry)

    return wrong_dataset, missing_in_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a dataset with only incorrectly answered questions.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing per-question result JSON files (output of run_experiment).",
    )
    parser.add_argument(
        "--dataset-type",
        default="short",
        choices=sorted(VALID_TYPES),
        help="Dataset type used in the experiment.",
    )
    parser.add_argument(
        "--dataset-set",
        default="longmemeval",
        choices=sorted(VALID_SETS),
        help="Dataset set used in the experiment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the filtered dataset. Defaults to <results-dir>/wrong_questions.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    dataset_path = resolve_dataset_path(args.dataset_type, args.dataset_set)
    dataset_map = load_dataset_map(dataset_path)

    wrong_results, missing_correct_flag = collect_wrong_results(args.results_dir)
    wrong_dataset, missing_in_dataset = build_wrong_dataset(wrong_results, dataset_map)

    output_path = args.output or args.results_dir / "wrong_questions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(wrong_dataset, f, indent=2, ensure_ascii=False)

    print(f"Total results read: {len(list(args.results_dir.glob('*.json')))}")
    print(f"Incorrect answers found: {len(wrong_dataset)}")
    print(f"Saved filtered dataset to: {output_path}")

    if missing_correct_flag:
        print(
            f"Skipped {len(missing_correct_flag)} files without 'answer_is_correct' "
            f"(first few: {missing_correct_flag[:5]})"
        )
    if missing_in_dataset:
        print(
            f"Skipped {len(missing_in_dataset)} IDs not found in dataset "
            f"(first few: {missing_in_dataset[:5]})"
        )


if __name__ == "__main__":
    main()

