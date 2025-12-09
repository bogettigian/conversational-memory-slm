"""
Script to generate plots showing:
1. Distribution of questions per question type in the dataset
2. Correct answers over total for each question type from experiment results
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_dataset(dataset_path: str) -> list[dict]:
    """Load the dataset from a JSON file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_questions_by_type(data: list[dict]) -> dict[str, int]:
    """Count the number of questions per question type."""
    return dict(Counter(item["question_type"] for item in data))


def load_experiment_results(results_dir: Path) -> list[dict]:
    """Load all experiment result files from a directory."""
    results = []
    for result_file in results_dir.glob("*.json"):
        if result_file.name == "metrics.json":
            continue
        with open(result_file, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def calculate_accuracy_by_type(results: list[dict]) -> dict[str, dict]:
    """Calculate correct/total for each question type."""
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in results:
        qtype = result["question_type"]
        stats[qtype]["total"] += 1
        if result.get("answer_is_correct", False):
            stats[qtype]["correct"] += 1
    
    return dict(stats)


def plot_question_types(counts: dict[str, int], output_path: str = None, title: str = "Questions per Type"):
    """Generate a bar plot showing question counts per type."""
    # Sort by count descending
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    types = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette
    colors = plt.cm.viridis([i / len(types) for i in range(len(types))])

    # Create bar chart
    bars = ax.bar(types, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{value}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Styling
    ax.set_xlabel("Question Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Questions", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add total count annotation
    total = sum(values)
    ax.text(
        0.98,
        0.98,
        f"Total: {total} questions",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_accuracy_by_type(stats: dict[str, dict], output_path: str = None, title: str = "Accuracy per Question Type"):
    """Generate a bar plot showing correct/total for each question type."""
    # Sort by accuracy descending
    sorted_items = sorted(
        stats.items(), 
        key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0, 
        reverse=True
    )
    types = [item[0] for item in sorted_items]
    correct = [item[1]["correct"] for item in sorted_items]
    total = [item[1]["total"] for item in sorted_items]
    accuracy = [c / t if t > 0 else 0 for c, t in zip(correct, total)]

    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color based on accuracy (green = high, red = low)
    colors = plt.cm.RdYlGn([acc for acc in accuracy])

    # Create bar chart
    bars = ax.bar(types, accuracy, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on top of bars (showing correct/total and percentage)
    for bar, c, t, acc in zip(bars, correct, total, accuracy):
        height = bar.get_height()
        ax.annotate(
            f"{c}/{t}\n({acc:.1%})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Styling
    ax.set_xlabel("Question Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.15)  # Leave room for labels

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # Add grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add overall accuracy annotation
    total_correct = sum(correct)
    total_questions = sum(total)
    overall_acc = total_correct / total_questions if total_questions > 0 else 0
    ax.text(
        0.98,
        0.98,
        f"Overall: {total_correct}/{total_questions} ({overall_acc:.1%})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "plots"
    output_dir.mkdir(exist_ok=True)

    # Part 1: Plot dataset distribution
    print("\n" + "=" * 60)
    print("DATASET DISTRIBUTION")
    print("=" * 60)

    datasets = {
        "LongMemEval Oracle": project_root / "data/longmemeval/longmemeval_oracle.json",
        "LongMemEval Short": project_root / "data/longmemeval/longmemeval_s_cleaned.json",
        "Investigathon Oracle": project_root / "data/investigathon/investigathon_LLMTrack_Evaluation_oracle.json",
        "Investigathon Short": project_root / "data/investigathon/investigathon_LLMTrack_Evaluation_s_cleaned.json",
    }

    for name, dataset_path in datasets.items():
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            continue

        print(f"\nProcessing: {name}")
        data = load_dataset(str(dataset_path))
        counts = count_questions_by_type(data)

        print(f"Question type distribution:")
        for qtype, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {qtype}: {count}")
        print(f"  Total: {len(data)}")

        safe_name = name.lower().replace(" ", "_")
        output_path = output_dir / f"question_types_{safe_name}.png"
        plot_question_types(counts, str(output_path), title=f"{name} - Questions per Type")

    # Part 2: Plot experiment results accuracy by type
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS - ACCURACY BY QUESTION TYPE")
    print("=" * 60)

    results_base_dir = project_root / "data/results"
    if not results_base_dir.exists():
        print(f"\nNo results directory found at: {results_base_dir}")
        print("Run main.py first to generate experiment results.")
        return

    # Find all experiment result directories
    experiment_dirs = []
    for dataset_dir in results_base_dir.iterdir():
        if dataset_dir.is_dir():
            for type_dir in dataset_dir.iterdir():
                if type_dir.is_dir():
                    for model_exp_dir in type_dir.iterdir():
                        if model_exp_dir.is_dir():
                            for emb_exp_dir in model_exp_dir.iterdir():
                                if emb_exp_dir.is_dir():
                                    for save_exp_dir in emb_exp_dir.iterdir():
                                        if save_exp_dir.is_dir():
                                            for search_exp_dir in save_exp_dir.iterdir():
                                                if search_exp_dir.is_dir():
                                                    experiment_dirs.append(search_exp_dir)

    if not experiment_dirs:
        print("\nNo experiment results found.")
        print("Run main.py first to generate experiment results.")
        return

    for exp_dir in experiment_dirs:
        print(f"\nProcessing: {exp_dir.name[:50]}...")
        
        results = load_experiment_results(exp_dir)
        if not results:
            print(f"  No results found in {exp_dir}")
            continue

        stats = calculate_accuracy_by_type(results)

        print(f"  Accuracy by question type:")
        for qtype, s in sorted(stats.items(), key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0, reverse=True):
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0
            print(f"    {qtype}: {s['correct']}/{s['total']} ({acc:.1%})")

        total_correct = sum(s["correct"] for s in stats.values())
        total_questions = sum(s["total"] for s in stats.values())
        overall_acc = total_correct / total_questions if total_questions > 0 else 0
        print(f"  Overall: {total_correct}/{total_questions} ({overall_acc:.1%})")

        # Create a safe filename from experiment directory
        safe_name = exp_dir.name[:80].replace("/", "_")
        output_path = output_dir / f"accuracy_{safe_name}.png"
        
        # Extract a shorter title
        parts = exp_dir.name.split("_")
        model_idx = next((i for i, p in enumerate(parts) if p == "MODEL"), None)
        if model_idx and model_idx + 1 < len(parts):
            model_name = parts[model_idx + 1]
        else:
            model_name = exp_dir.name[:40]
        
        plot_accuracy_by_type(
            stats, 
            str(output_path), 
            title=f"Accuracy per Question Type\n({model_name})"
        )

    print(f"\n{'=' * 60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
