import yaml
import os
import logging
from pathlib import Path
from sklearn.datasets import (
    load_diabetes, load_iris, load_breast_cancer,
    fetch_openml, load_wine, fetch_california_housing, load_digits
)

from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import logging

def load_config():
    config_path = Path(__file__).parent / 'config.yaml'
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            logging.info("[Config Loader] Successfully loaded configuration.")
            if 'gp' in cfg:
                gp_cfg = cfg['gp']
                print("[Config] Genetic Programming Parameters:")
                for k, v in gp_cfg.items():
                    print(f"  {k}: {v}")
            return cfg
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing config file: {str(e)}")

cfg = load_config()

# ---- DATA LOADING ----
def load_regression_data(name):
    datasets = {
        'diabetes': load_diabetes,
        'boston': lambda: fetch_openml(name='boston', version=1, as_frame=False),
        'california_housing': fetch_california_housing
    }

    if name not in datasets:
        raise ValueError(f"Unknown regression dataset: {name}")

    try:
        data = datasets[name]()
    except Exception as e:
        logging.error(f"Dataset loading failed: {e}")
        raise

    # Extract raw features and targets
    if hasattr(data, 'data') and hasattr(data, 'target'):
        X_raw = data.data
        y = data.target
    else:
        raise RuntimeError("Dataset does not contain expected 'data' and 'target' attributes.")

    # Downsample large datasets
    if name == "california_housing" and X_raw.shape[0] > 5000:
        logging.info(f"Downsampling California Housing from {X_raw.shape[0]} to 2000 samples for faster GP.")
        X_raw, y = resample(X_raw, y, n_samples=2000, random_state=42)

    # Scale features
    X = StandardScaler().fit_transform(X_raw)

    # Safe feature name extraction
    if hasattr(data, 'feature_names'):
        features = data.feature_names
    elif hasattr(data, 'data') and hasattr(data.data, 'columns'):
        features = list(data.data.columns)
    else:
        features = [f'x{i}' for i in range(X.shape[1])]

    logging.info(f"Successfully loaded regression dataset: {name} with shape {X.shape}")
    return X, y, features


def load_classification_data(name):
    datasets = {
        'iris': load_iris,
        'breast_cancer': load_breast_cancer,
        'wine': load_wine,
        'digits': load_digits
    }

    if name not in datasets:
        raise ValueError(f"Unknown classification dataset: {name}")

    data = datasets[name]()
    X = StandardScaler().fit_transform(data.data)

    if hasattr(data, 'feature_names'):
        features = data.feature_names
    elif hasattr(data, 'data') and hasattr(data.data, 'columns'):
        features = list(data.data.columns)
    else:
        features = [f'x{i}' for i in range(X.shape[1])]

    logging.info(f"Successfully loaded classification dataset: {name}")
    return X, data.target, features