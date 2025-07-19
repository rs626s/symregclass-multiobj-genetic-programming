import operator
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def evaluate_model(pareto_front, X_test, y_test, task_type, toolbox):
    results = []
    for ind in pareto_front:
        func = toolbox.compile(expr=ind)
        try:
            predictions = np.array([func(*x) for x in X_test])
        except Exception:
            continue

        if np.all(predictions == predictions[0]):
            continue  # Skip constant models

        if task_type == "regression":
            metric = mean_squared_error(y_test, predictions)
        else:
            predictions = np.round(predictions).astype(int)
            predictions = np.clip(predictions, np.min(y_test), np.max(y_test))
            metric = accuracy_score(y_test, predictions)

        expr = str(ind)
        features_used = sum(('ARG' in token or 'x' in token) for token in expr.split())
        results.append({
            "expression": expr,
            "features_used": features_used,
            "metric": metric
        })

    results.sort(key=lambda x: x["metric"])
    return results

def evaluate_baselines(X, y, task_type):
    results = {}
    if task_type == "regression":
        models = {
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression()
        }
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
            results[name] = -np.mean(scores)
    else:
        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            results[name] = np.mean(scores)
    return results
