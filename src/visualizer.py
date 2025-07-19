import json, os
import matplotlib.pyplot as plt

def save_results(name, results):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

def save_pareto_plot(name, all_models, pareto_models, task_type):
    os.makedirs("plots", exist_ok=True)

    ylabel = "MSE" if task_type == "regression" else "Accuracy"

    x_all = [ind.fitness.values[1] for ind in all_models]
    y_all = [ind.fitness.values[0] for ind in all_models]

    x_pareto = [ind.fitness.values[1] for ind in pareto_models]
    y_pareto = [ind.fitness.values[0] for ind in pareto_models]

    plt.figure()

    if len(x_all) > 1:
        plt.plot(x_all, y_all, 'b--', label="GP Models", alpha=0.5)
        plt.scatter(x_all, y_all, c='blue')
    else:
        plt.scatter(x_all, y_all, c='blue', label="GP Models")

    if len(x_pareto) > 1:
        plt.plot(x_pareto, y_pareto, 'ro--', label="Pareto Front")
    else:
        plt.scatter(x_pareto, y_pareto, c='red', label="Pareto Front")

    plt.xlabel("Model Complexity (features used)")
    plt.ylabel(ylabel)
    plt.title(f"Pareto Front - {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{name}_pareto.png")
    plt.close()
