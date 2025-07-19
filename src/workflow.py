import os
from sklearn.model_selection import KFold
from src.loader import load_datasets
from src.evaluator import evaluate_model, evaluate_baselines
from src.gp_model import GeneticProgrammingModel
from src.visualizer import save_results, save_pareto_plot

def run_experiment():
    datasets = load_datasets()
    for name, (X, y, task_type) in datasets.items():
        print(f"\nRunning on dataset: {name}")
        fold_results = []
        gp_all_individuals = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            gp = GeneticProgrammingModel(task_type=task_type)
            pareto_front = gp.run(X_train, y_train)
            gp_all_individuals.extend(pareto_front)
            results = evaluate_model(pareto_front, X_test, y_test, task_type, toolbox=gp.toolbox)
            fold_results.extend(results)

        save_results(name, fold_results)
        save_pareto_plot(name, gp_all_individuals, pareto_front, task_type)

        gp_final = GeneticProgrammingModel(task_type=task_type)
        pareto_front_final = gp_final.run(X, y)
        top_results = evaluate_model(pareto_front_final, X, y, task_type, toolbox=gp_final.toolbox)
        print("\nTop GP Models:")
        for i, res in enumerate(top_results[:3]):
            print(f"Model {i+1}:")
            print(f"  Expression: {res['expression']}")
            print(f"  Features used: {res['features_used']}")

        # Baselines
        baseline_results = evaluate_baselines(X, y, task_type)
        print("\nBaseline comparisons:")
        for model_name, metric in baseline_results.items():
            if task_type == "regression":
                print(f"  {model_name}: MSE = {metric:.4f}")
            else:
                print(f"  {model_name}: Accuracy = {metric:.4f}")
