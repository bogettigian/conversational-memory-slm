import argparse
import json
from collections import Counter
from statistics import mean, median
from pathlib import Path

import matplotlib.pyplot as plt


def load_dataset(dataset_path: Path):
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_has_answer_stats(data):
    per_question_counts = []
    role_counts = Counter()
    questions_with_both_roles = 0
    questions_assistant_only = 0
    questions_user_only = 0
    questions_no_answer = 0

    for item in data:
        message_count = 0
        roles_with_answer = set()
        for session in item.get("haystack_sessions", []):
            for message in session:
                if not message.get("has_answer", False):
                    continue

                message_count += 1
                role = message.get("role")
                if role in {"user", "assistant"}:
                    role_counts[role] += 1
                    roles_with_answer.add(role)

        per_question_counts.append(message_count)
        if {"user", "assistant"} <= roles_with_answer:
            questions_with_both_roles += 1
        elif roles_with_answer == {"assistant"}:
            questions_assistant_only += 1
        elif roles_with_answer == {"user"}:
            questions_user_only += 1
        elif not roles_with_answer:
            questions_no_answer += 1

    return (
        per_question_counts,
        role_counts,
        questions_with_both_roles,
        questions_assistant_only,
        questions_user_only,
        questions_no_answer,
    )


def plot_per_question_hist(counts, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    freq = Counter(counts)
    x_vals = sorted(freq.keys())
    y_vals = [freq[x] for x in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, marker="o", color="#4C72B0")
    plt.xlabel("Messages with has_answer=True per question")
    plt.ylabel("Number of questions")
    plt.title("Distribution of has_answer messages per question (line chart)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved per-question line chart to {output_path}")


def plot_role_bar(role_counts: Counter, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    roles = ["user", "assistant"]
    values = [role_counts.get(r, 0) for r in roles]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(roles, values, color=["#55A868", "#C44E52"], alpha=0.85)
    plt.title("Messages with has_answer=True by role")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved role distribution bar plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot has_answer distributions for the longmemeval dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/longmemeval/longmemeval_oracle.json"),
        help="Path to dataset JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to store generated plots.",
    )
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    (
        per_question_counts,
        role_counts,
        questions_with_both_roles,
        questions_assistant_only,
        questions_user_only,
        questions_no_answer,
    ) = compute_has_answer_stats(data)

    if not per_question_counts:
        raise ValueError("No questions found in dataset.")

    print(
        f"Questions={len(per_question_counts)}, "
        f"min={min(per_question_counts)}, "
        f"max={max(per_question_counts)}, "
        f"mean={mean(per_question_counts):.2f}, "
        f"median={median(per_question_counts):.2f}"
    )
    print(
        "Messages with has_answer=True by role: "
        + ", ".join(f"{role}={count}" for role, count in role_counts.items())
    )
    print(
        f"Questions with has_answer=True from both user and assistant: "
        f"{questions_with_both_roles}"
    )
    print(
        f"Questions with has_answer=True only in assistant messages: "
        f"{questions_assistant_only}"
    )
    print(
        f"Questions with has_answer=True only in user messages: "
        f"{questions_user_only}"
    )
    print(f"Questions with no has_answer=True messages: {questions_no_answer}")
    print(
        f"Sanity check total: "
        f"{questions_with_both_roles + questions_assistant_only + questions_user_only + questions_no_answer}"
    )

    plot_per_question_hist(
        per_question_counts, args.output_dir / "has_answer_per_question_hist.png"
    )
    plot_role_bar(role_counts, args.output_dir / "has_answer_by_role.png")


if __name__ == "__main__":
    main()

