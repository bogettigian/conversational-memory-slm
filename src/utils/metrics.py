import json
import os

import numpy as np


def metric_generator(path_dir: str):
    asserts = []
    online_latencies = []
    offline_latencies = []
    contexts_length = []
    for filename in os.listdir(path_dir):
        if filename != "metrics.json":
            with open(os.path.join(path_dir, filename), "r", encoding="utf-8") as file:
                data = json.load(file)
                asserts.append(data["answer_is_correct"])
                online_latencies.append(data["online_latency"])
                offline_latencies.append(data["offline_latency"])
                contexts_length.append(data["context_length"])

    score = sum([1 if a else 0 for a in asserts]) / len(asserts)
    online_latency_mean = np.mean(online_latencies)
    online_latency_var = np.var(online_latencies)
    offline_latency_mean = np.mean(offline_latencies)
    offline_latency_var = np.var(offline_latencies)
    context_avg = np.average(contexts_length)

    result_file = f"{path_dir}/metrics.json"
    with open(result_file, "w", encoding="utf-8") as file:
        result = {
            "score": score,
            "online_latency_mean": online_latency_mean,
            "online_latency_var": online_latency_var,
            "offline_latency_mean": offline_latency_mean,
            "offline_latency_var": offline_latency_var,
            "context_avg": context_avg,
        }
        print(f"  Score: {score}")
        print(f"  Online latency mean: {online_latency_mean}")
        print(f"  Online latency variance: {online_latency_var}")
        print(f"  Offline latency mean: {offline_latency_mean}")
        print(f"  Offline latency variance: {offline_latency_var}")
        print(f"  Context average: {context_avg}")

        json.dump(result, file, indent=2)
