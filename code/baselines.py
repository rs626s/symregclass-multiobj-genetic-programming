import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

def run_baseline(X_train, y_train, X_test, y_test, task="classification"):
    if task == "classification":
        models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=3),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
    elif task == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
        }
    else:
        raise ValueError("Task must be either 'classification' or 'regression'")

    results = {}
    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = preds

        if task == "classification":
            scores[name] = np.mean(preds == y_test)
        else:
            scores[name] = np.mean((preds - y_test) ** 2)

    if task == "classification":
        best_model = max(scores, key=scores.get)
        print(f"[Baseline] Best classification model: {best_model}, Accuracy: {scores[best_model]:.4f}")
        return results[best_model], best_model
    else:
        print("[Baseline] Regression model MSEs:")
        for model_name, mse in scores.items():
            print(f"  {model_name}: {mse:.4f}")
        return scores  # <- return all MSEs for regression

