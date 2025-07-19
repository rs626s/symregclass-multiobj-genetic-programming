from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

def evaluate_baselines(X, y, task_type):
    models = []
    if task_type == "classification":
        models = [
            ("Decision Tree", DecisionTreeClassifier()),
            ("Random Forest", RandomForestClassifier()),
            ("Logistic Regression", LogisticRegression(max_iter=2000))
        ]
        scoring = "accuracy"
    else:
        models = [
            ("Decision Tree", DecisionTreeRegressor()),
            ("Random Forest", RandomForestRegressor()),
            ("Linear Regression", LinearRegression())
        ]
        scoring = "neg_mean_squared_error"

    results = []
    for name, model in models:
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        score = np.mean(scores)
        results.append({
            "model": name,
            "metric": "Accuracy" if task_type == "classification" else "MSE",
            "value": score if task_type == "classification" else -1 * score
        })
    return results
