import argparse
import json
from collections import defaultdict
from statistics import median
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def load_messages(dataset_path: Path):
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        for session in item.get("haystack_sessions", []):
            for message in session:
                yield message


def compute_lengths(messages):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.model_max_length = 1_000_000  # avoid truncation warnings; only counting tokens
    lengths = defaultdict(list)

    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue

        content = message.get("content", "")
        token_count = len(tokenizer.encode(content, add_special_tokens=False))
        lengths[role].append(token_count)

    if not lengths:
        raise ValueError("No user or assistant messages found in the dataset.")

    return lengths


def plot_boxplot(lengths, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    roles = ["user", "assistant"]
    data = [lengths.get(role, []) for role in roles]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, tick_labels=roles, patch_artist=True)
    ax.set_title("Message length distribution by role")
    ax.set_ylabel("Token count (bert-base-uncased)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved boxplot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot token-length distribution of messages by role."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/longmemeval/longmemeval_oracle.json"),
        help="Path to dataset JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/message_length_boxplot.png"),
        help="Where to save the boxplot.",
    )
    args = parser.parse_args()

    messages = load_messages(args.dataset)
    lengths = compute_lengths(messages)

    for role, values in lengths.items():
        print(
            f"{role}: count={len(values)}, mean={sum(values)/len(values):.2f}, "
            f"median={median(values):.2f}"
        )

    plot_boxplot(lengths, args.output)


if __name__ == "__main__":
    main()

