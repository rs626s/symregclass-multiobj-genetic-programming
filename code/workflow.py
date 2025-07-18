import numpy as np
import os
import logging
import copy
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    mean_squared_error, r2_score
)
from deap import tools, creator
from tqdm import tqdm

from gp_model import GPModel
from visualizer import Visualizer as V
from loader import load_classification_data, load_regression_data
from baselines import run_baseline
from evaluator import Evaluator
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def run_workflow(dataset_name, cfg, task="classification"):
    assert task in ["classification", "regression"], "Invalid task type"

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        if task == "classification":
            X, y, feature_names = load_classification_data(dataset_name)
        else:
            X, y, feature_names = load_regression_data(dataset_name)
    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        return

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gp_results, baseline_results = [], []
    all_fronts = []
    baseline_scores = []
    
    base_dir = os.path.join(cfg['visualization'].get('plot_dir', "outputs"), task)
    dataset_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)  # Make sure the folder exists

    # Save the file inside the dataset folder
    output_file = os.path.join(dataset_dir, f"{dataset_name}_all_folds_equations.txt")
    with open(output_file, "w") as f:
        f.write(f"GP Equations for dataset: {dataset_name}\n\n")

    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(X), desc="Processing folds")):
        print(f"\n--- Fold {fold + 1} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        gp = GPModel(mode=task)
        gp.fit(X_train, y_train, feature_names=feature_names)

        if not gp.pareto_front or not isinstance(gp.pareto_front[0], creator.Individual):
            logging.warning(f"[Fold {fold}] Invalid or empty Pareto front. Skipping.")
            continue

        best_eq = gp.get_best_equation()
        complexity = gp.get_complexity_metrics()[0]

        if task == "classification":
            result = Evaluator.evaluate_classification(gp, X_test, y_test, y_train)
            print(f"GP Accuracy: {result['accuracy']:.4f}")
            print(f"GP Complexity: {result['complexity']}")
            print(f"GP Best Equation: {best_eq}")
            print(f"Confusion Matrix (GP):\n{result['confusion_matrix']}")
            cm = result['confusion_matrix']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title(f"Confusion Matrix - Fold {fold + 1}")
            cm_path = os.path.join(dataset_dir, f"fold_{fold + 1}_gp_confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            gp_results.append({
                'accuracy': result['accuracy'],
                'equation': best_eq,
                'complexity': result['complexity'],
                'confusion_matrix': result['confusion_matrix']
            })
        else:
            result = Evaluator.evaluate_regression(gp, X_test, y_test)
            print(f"GP MSE: {result['mse']:.4f}, Square(R): {result['r2']:.4f}")
            print(f"GP Complexity: {result['complexity']}")
            print(f"GP Best Equation: {best_eq}")

            gp_results.append({
                'mse': result['mse'],
                'r2': result['r2'],
                'equation': best_eq,
                'complexity': result['complexity']
            })
            
        # Baseline
        if task == "classification":
            baseline_pred, model_name = run_baseline(X_train, y_train, X_test, y_test, task=task)
            baseline_score = accuracy_score(y_test, baseline_pred)
            print(f"Baseline Accuracy ({model_name}): {baseline_score:.4f}")
            cm = confusion_matrix(y_test, baseline_pred)
            formatted_cm = '[' + ';'.join(['[' + ','.join(map(str, row)) + ']' for row in cm]) + ']'
            baseline_results.append({
                'accuracy': baseline_score,
                'model': model_name,
                'confusion_matrix': formatted_cm
            })
            baseline_scores.append(baseline_score)
        else:
            baseline_mse_dict = run_baseline(X_train, y_train, X_test, y_test, task="regression")
            for model_name, mse in baseline_mse_dict.items():
                baseline_results.append({'model': model_name, 'mse': mse})
                baseline_scores.append(mse)

        all_fronts.append(gp.pareto_front)
        
        with open(output_file, "a") as f:
            f.write(f"=== Fold {fold} ===\n")
            f.write(f"Best equation: {best_eq}\n")
            
            if task == "classification":
                f.write(f"Complexity: {result['complexity']}\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            else:
                f.write(f"Complexity: {result['complexity']}\n")
                f.write(f"MSE: {result['mse']:.4f}, Square(R): {result['r2']:.4f}\n")
            
            f.write("\n")


    if gp_results:
        combined_front = tools.ParetoFront()
        for front in all_fronts:
            combined_front.update([copy.deepcopy(ind) for ind in front])

        V.plot_interactive_pareto(combined_front, feature_names, f"Interactive Pareto - {task.capitalize()} - {dataset_name}", task, dataset_name)
        V.plot_pareto_2d(combined_front, f"{dataset_name} {task.capitalize()} Pareto Front", task, dataset_name, baseline_score=np.mean(baseline_scores))
        V.plot_feature_usage(combined_front, feature_names, task, dataset_name)

        gp_df = pd.DataFrame(gp_results)
        baseline_df = pd.DataFrame(baseline_results)
        
        base_dir = os.path.join(cfg['visualization'].get('plot_dir', "outputs"), task)
        results_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        baseline_df.to_csv(os.path.join(results_dir, f"{dataset_name}_baseline_fold_results.csv"), index=False)

        # Optional enhancement: add best baseline info to GP fold results
        if task == "regression":
            gp_df["fold"] = range(len(gp_df))
            baseline_df["fold"] = baseline_df.index // (len(baseline_df) // len(gp_df))
            best_baseline = baseline_df.loc[baseline_df.groupby("fold")["mse"].idxmin()]
            best_baseline.rename(columns={"model": "best_baseline_model", "mse": "baseline_mse"}, inplace=True)
            merged = pd.merge(gp_df, best_baseline[["fold", "best_baseline_model", "baseline_mse"]], on="fold", how="left")
            merged["gp_better_than_baseline"] = merged["mse"] < merged["baseline_mse"]
            merged.to_csv(os.path.join(results_dir, f"{dataset_name}_gp_fold_results.csv"), index=False)

        elif task == "classification":
            gp_df["fold"] = range(len(gp_df))
            baseline_df["fold"] = range(len(baseline_df))
            best_baseline = baseline_df.loc[baseline_df.groupby("fold")["accuracy"].idxmax()]
            best_baseline.rename(columns={"model": "best_baseline_model", "accuracy": "baseline_accuracy"}, inplace=True)
            merged = pd.merge(gp_df, best_baseline[["fold", "best_baseline_model", "baseline_accuracy"]], on="fold", how="left")
            merged["gp_better_than_baseline"] = merged["accuracy"] > merged["baseline_accuracy"]
            merged.to_csv(os.path.join(results_dir, f"{dataset_name}_gp_fold_results.csv"), index=False)

        logging.info("\n=== Final Summary ===")
        folds = list(range(1, len(gp_results) + 1))

        if task == "classification":
            gp_scores = [r['accuracy'] for r in gp_results]
            baseline_scores = list(merged['baseline_accuracy'])
            metric = "Accuracy"
        else:
            gp_scores = [r['mse'] for r in gp_results]
            baseline_scores = list(merged['baseline_mse'])
            metric = "MSE"

        plt.figure(figsize=(10, 5))
        plt.bar(folds, gp_scores, width=0.35, label='GP', align='center')
        plt.bar([f + 0.35 for f in folds], baseline_scores, width=0.35, label='Baseline', align='center')
        plt.xlabel("Fold")
        plt.ylabel(metric)
        plt.title(f"{metric} per Fold: GP vs Baseline")
        plt.xticks([f + 0.175 for f in folds], folds)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f"{dataset_name}_gp_vs_baseline_{task}.png")
        plt.savefig(plot_path)
        plt.close()
        
        V.plot_complexity_histogram(combined_front, task, dataset_name)

        if task == "classification":
            logging.info(f"GP Avg Accuracy: {np.mean([r['accuracy'] for r in gp_results]):.4f}")
            logging.info(f"Baseline Avg Accuracy: {np.mean(baseline_scores):.4f}")
        else:
            logging.info(f"GP Avg MSE: {np.mean([r['mse'] for r in gp_results]):.4f}")
            logging.info(f"GP Avg Square(R): {np.mean([r['r2'] for r in gp_results]):.4f}")
            logging.info(f"Baseline Avg MSE: {np.mean(baseline_scores):.4f}")
            logging.info(f"Pareto Front Size (avg +/- std): {np.mean([len(f) for f in all_fronts]):.1f} +/- {np.std([len(f) for f in all_fronts]):.1f}")
    else:
        logging.warning("No valid results were produced across any fold.")
