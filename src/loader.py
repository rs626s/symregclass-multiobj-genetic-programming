import numpy as np                                          #For numerical operations and array handling
from sklearn.datasets import (
    load_diabetes,                                          #Regression dataset    
    fetch_california_housing,                               #Large regression dataset
    load_iris,                                              #Classification dataset
    load_wine,                                              #Classification dataset
    load_digits,                                            #Classification dataset (image-based)
    load_breast_cancer                                      #Classification dataset
)
from sklearn.utils import resample                          #For downsampling large datasets

def load_datasets():
    datasets = {}                                           #Dictionary to hold all datasets with name as key

    #--- Regression Datasets ---

    X, y = load_diabetes(return_X_y=True)                   #Load the Diabetes dataset (for regression)    
    datasets["Diabetes"] = (X, y, "regression")

    data = fetch_california_housing(as_frame=True)          #Load the California Housing dataset (for regression)
    X_df = data.data                                        #Features as DataFrame
    y = data.target                                         #Target values

    if len(X_df) > 5000:                                    #Downsample if dataset is too large (for faster processing)
        X_df, y = resample(X_df, y, n_samples=3000, random_state=42)        #Sample 3000 rows
        X_df.reset_index(drop=True, inplace=True)                           #Reset index to avoid mismatches
        y = y.reset_index(drop=True)

    X = X_df.to_numpy()                                                     #Convert DataFrame to NumPy array
    datasets["California"] = (X, y, "regression")                           #Add to datasets dictionary

    #--- Classification Datasets ---

    X, y = load_iris(return_X_y=True)                                       #Load Iris dataset (3-class classification)
    datasets["Iris"] = (X, y, "classification")

    X, y = load_wine(return_X_y=True)                                       #Load Wine dataset (multi-class classification)
    datasets["Wine"] = (X, y, "classification")

    X, y = load_digits(return_X_y=True)                                     #Load Digits dataset (multi-class classification of 8x8 image digits)
    if len(X) > 5000:
        X, y = resample(X, y, n_samples=3000, random_state=42)              #Optional downsampling
    datasets["Digits"] = (X, y, "classification")

    X, y = load_breast_cancer(return_X_y=True)                              #Load Breast Cancer dataset (binary classification)
    datasets["BreastCancer"] = (X, y, "classification")

    return datasets
