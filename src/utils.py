import numpy as np
import os
import json

def count_nodes(individual):
    return len(individual)

def save_results(name, results):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
