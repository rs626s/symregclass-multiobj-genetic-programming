import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import os
from loader import cfg
from deap import gp
import logging
import pandas as pd
import sys

class Visualizer:
    @staticmethod
    def plot_pareto_2d(front, title="Pareto Front", task_type="classification", dataset_name="unknown", baseline_score=None):
        if not front:
            logging.warning("[Visualizer] Empty Pareto front provided")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        errors, complexities = [], []
        for ind in front:
            if hasattr(ind, "fitness") and hasattr(ind.fitness, "values") and len(ind.fitness.values) >= 2:
                err, comp = ind.fitness.values[:2]
                errors.append(err)
                complexities.append(comp)
            else:
                logging.warning(f"[Visualizer] Skipping malformed individual: {ind}")

        if not errors or not complexities:
            logging.warning("[Visualizer] No valid individuals found for plotting.")
            return

        ax.scatter(complexities, errors, c='blue', s=50, edgecolors='k', label='GP Solutions')

        if baseline_score is not None:
            ax.axhline(y=baseline_score, color='red', linestyle='--', label=f'Baseline (Error={baseline_score:.2f})')

        ylabel = "MSE" if task_type.lower() == "regression" or "regression" in title.lower() else "Error Rate"
        ax.set_xlabel("Complexity (Tree Size)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nTrade-off: Accuracy vs. Interpretability")
        ax.legend()
        ax.grid(True)

        if cfg.get('visualization', {}).get('save_plots', False):
            base_dir = os.path.join(cfg['visualization'].get('plot_dir', "outputs"), task_type)
            plot_dir = os.path.join(base_dir, dataset_name)
            os.makedirs(plot_dir, exist_ok=True)
            save_path = f"{plot_dir}/{dataset_name}_pareto_2d_{task_type}.png"
            plt.savefig(save_path)
            logging.info(f"[Visualizer] Plot saved to: {save_path}")                
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_feature_usage(front, feature_names, task_type="classification", dataset_name="unknown"):
        if not front or feature_names is None or len(feature_names) == 0:
            logging.warning("[Visualizer] Empty feature names or Pareto front")
            return

        feature_names = list(feature_names)
        usage = {name: 0 for name in feature_names}
        for ind in front:
            for node in ind:
                if isinstance(node, gp.Terminal) and node.value is None:
                    name = node.name
                    if name in usage:
                        usage[name] += 1

        plt.figure(figsize=(10, 4))
        plt.bar(usage.keys(), usage.values())
        plt.title("Feature Usage in Pareto Front")
        plt.xticks(rotation=45)
        plt.tight_layout()

        base_dir = os.path.join(cfg['visualization'].get('plot_dir', "outputs"), task_type)
        plot_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(plot_dir, exist_ok=True)
        save_path = f"{plot_dir}/{dataset_name}_feature_usage_{task_type}.png"
        plt.savefig(save_path)
        logging.info(f"[Visualizer] Feature usage plot saved to: {save_path}")
        plt.close()

    @staticmethod
    def plot_interactive_pareto(pareto_front, feature_names=None, title="Pareto Front (Interactive)", task_type="classification", dataset_name="unknown"):
        try:
            objectives = [
                (ind.fitness.values[0], ind.fitness.values[1])
                for ind in pareto_front
                if hasattr(ind, "fitness") and len(ind.fitness.values) >= 2
            ]

            if not objectives:
                print("[Visualizer] No valid individuals to plot.")
                return

            df = pd.DataFrame(objectives, columns=["error", "complexity"])

            if task_type.lower() == "classification":
                df["accuracy"] = 1 - df["error"]
                y_axis = "accuracy"
                y_label = "Accuracy"
            else:
                y_axis = "error"
                y_label = "MSE"

            fig = px.scatter(
                df, x="complexity", y=y_axis, title=title,
                labels={"complexity": "Model Complexity", y_axis: y_label},
                hover_data=["error"]
            )

            base_dir = os.path.join(cfg['visualization'].get('plot_dir', "outputs"), task_type)
            plot_dir = os.path.join(base_dir, dataset_name)
            os.makedirs(plot_dir, exist_ok=True)
            html_filename = f"{plot_dir}/{dataset_name}_interactive_pareto_{task_type}.html"
            fig.write_html(html_filename)
            logging.info(f"[Visualizer] Saved interactive Pareto plot to {html_filename}")

        except ImportError:
            print("[Visualizer] Required libraries (plotly, pandas) not installed.")

    @staticmethod
    def plot_complexity_histogram(pareto_front, task_type="classification", dataset_name="unknown"):
        import matplotlib.pyplot as plt

        if not pareto_front:
            logging.warning("[Visualizer] Empty front for complexity histogram")
            return

        complexities = [len(ind) for ind in pareto_front]

        plt.figure(figsize=(8, 5))
        plt.hist(complexities, bins=10, edgecolor='black')
        plt.xlabel("Tree Size (Complexity)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Model Complexity in Pareto Front")

        base_dir = os.path.join(cfg['visualization'].get('plot_dir', "outputs"), task_type)
        plot_dir = os.path.join(base_dir, dataset_name)
        os.makedirs(plot_dir, exist_ok=True)
        save_path = f"{plot_dir}/{dataset_name}_complexity_hist_{task_type}.png"
        plt.savefig(save_path)
        logging.info(f"[Visualizer] Complexity histogram saved to: {save_path}")
        plt.close()
