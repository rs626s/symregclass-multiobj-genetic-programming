# Multi-objective Genetic Programming for Symbolic Regression and Classification

This project implements a Genetic Programming (GP) system using the DEAP framework to solve both **regression** and **classification** problems. It applies **multi-objective optimization** (via NSGA-II) to evolve symbolic expressions that balance **model accuracy** and **complexity** (number of features used).

## 📁 Project Structure

```
mogp-symgregclass/
│
├── main.py                     # Entry point: runs experiments on datasets
├── results/                    # Generated JSON files with fold results
├── plots/                      # Saved Pareto front visualizations
│
├── src/
│   ├── baseline.py             # Baseline model evaluations (e.g., Linear, Random Forest, SVM)
│   ├── evaluator.py            # Evaluation functions for individuals and models
│   ├── extractor.py            # Expression tree parsing and feature usage analysis
│   ├── feature_usage.py        # Counts feature usage in GP expressions
│   ├── gp_model.py             # GP class implementing DEAP NSGA-II logic
│   ├── loader.py               # Dataset loading and preprocessing
│   ├── utils.py                # Helper functions (e.g., for simplification)
│   ├── visualizer.py           # Result saving and Pareto front plotting
│   └── workflow.py             # Main orchestration of GP runs, evaluation, and plots
```

## ⚙️ Features

- Multi-objective GP using DEAP's NSGA-II (`eaMuPlusLambda`)
- Built-in support for:
  - Symbolic **regression** (minimize MSE)
  - Symbolic **classification** (maximize accuracy)
- Parsimony pressure via individual length constraint
- Support for expressive function set: `+`, `-`, `*`, protected `/`, `sin`, `cos`, `neg`, `gt`, `if_then_else`
- Feature usage tracking for model interpretability
- 5-fold cross-validation and baseline comparison
- Results and plots are auto-saved per dataset

## 🧪 Datasets Supported

- Diabetes
- California Housing
- Iris
- Wine
- Digits
- Breast Cancer

> Data is loaded using `scikit-learn` datasets and optionally subsampled.

## 🚀 Running the Project

```bash
python main.py
```

All output is saved in:

- `results/<Dataset>_results.json`
- `plots/<Dataset>_pareto.png`

## 📈 Output Example

Each fold produces:
- Evaluation of GP individuals on test set
- Accuracy / MSE, number of features, expression string

Example JSON output (in `results/Diabetes_results.json`):
```json
[
  {
    "expression": "add(ARG0, mul(ARG3, ARG5))",
    "mse": 38.27,
    "features_used": 3
  },
  ...
]
```

## 📊 Pareto Plot

The plots show:
- All GP models (blue)
- Pareto-optimal models (red)

Axes:
- X-axis: Feature count (model complexity)
- Y-axis: MSE (regression) or Accuracy (classification)

## 📦 Requirements

- Python ≥ 3.8
- DEAP
- scikit-learn
- numpy
- matplotlib

Install with:

```bash
pip install -r requirements.txt
```

## 📌 Notes

- Ephemeral constants are generated using `functools.partial` to avoid pickling issues
- Random seed is fixed (`random_state=42`) for reproducibility
- GP population size, depth, and generations are configurable in `gp_model.py`

## 📚 References

- Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*
- DEAP documentation: [https://deap.readthedocs.io](https://deap.readthedocs.io)
