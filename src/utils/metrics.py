import json
import os

import numpy as np


def metric_generator(path_dir: str):
    asserts = []
    latencies = []
    contexts_length = []
    for filename in os.listdir(path_dir):
        if filename != "metrics.json":
            with open(os.path.join(path_dir, filename), "r", encoding="utf-8") as file:
                data = json.load(file)
                asserts.append(data["answer_is_correct"])
                latencies.append(data["latency"])
                contexts_length.append(data["context_length"])

    score = sum([1 if a else 0 for a in asserts]) / len(asserts)
    latency_mean = np.mean(latencies)
    latency_var = np.var(latencies)
    context_avg = np.average(contexts_length)

    result_file = f"{path_dir}/metrics.json"
    with open(result_file, "w", encoding="utf-8") as file:
        result = {
            "score": score,
            "latency_mean": latency_mean,
            "latency_var": latency_var,
            "context_avg": context_avg,
        }
        print(f"  Score: {score}")
        print(f"  Latency mean: {latency_mean}")
        print(f"  Latency variance: {latency_var}")
        print(f"  Context average: {context_avg}")

        json.dump(result, file, indent=2)
