import numpy as np
from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing,
    load_iris,
    load_wine,
    load_digits,
    load_breast_cancer
)
from sklearn.utils import resample

def load_datasets():
    datasets = {}

    # --- Regression Datasets ---

    # Diabetes
    X, y = load_diabetes(return_X_y=True)
    datasets["Diabetes"] = (X, y, "regression")

    # California Housing
    data = fetch_california_housing(as_frame=True)
    X_df = data.data
    y = data.target

    # Downsample large datasets for faster experimentation
    if len(X_df) > 5000:
        X_df, y = resample(X_df, y, n_samples=3000, random_state=42)
        X_df.reset_index(drop=True, inplace=True)  # Reset index to fix indexing issues
        y = y.reset_index(drop=True)

    X = X_df.to_numpy()
    datasets["California"] = (X, y, "regression")

    # --- Classification Datasets ---

    # Iris
    X, y = load_iris(return_X_y=True)
    datasets["Iris"] = (X, y, "classification")

    # Wine
    X, y = load_wine(return_X_y=True)
    datasets["Wine"] = (X, y, "classification")

    # Digits
    X, y = load_digits(return_X_y=True)
    if len(X) > 5000:
        X, y = resample(X, y, n_samples=3000, random_state=42)
    datasets["Digits"] = (X, y, "classification")

    # Breast Cancer
    X, y = load_breast_cancer(return_X_y=True)
    datasets["BreastCancer"] = (X, y, "classification")

    return datasets
