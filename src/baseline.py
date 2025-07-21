import warnings                                                                     #Used to manage and suppress warning messages    
from sklearn.exceptions import ConvergenceWarning                                   #Specific warning type for models like LogisticRegression
warnings.filterwarnings("ignore", category=ConvergenceWarning)                      #Ignore convergence warnings from scikit-learn
from sklearn.model_selection import cross_val_score                                 #Import cross-validation utility

# Import classification and regression model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np                                                                  #For numerical operations like averaging scores

def evaluate_baselines(X, y, task_type):
    models = []                                                                     #List to store (name, model) pairs
    if task_type == "classification":                                               #Choose models and scoring method based on task type
        models = [
            ("Decision Tree", DecisionTreeClassifier()),                            #Basic tree classifier
            ("Random Forest", RandomForestClassifier()),                            #Ensemble classifier
            ("Logistic Regression", LogisticRegression(max_iter=2000))              #Logistic Regression with higher iteration limit
        ]
        scoring = "accuracy"                                                        #Metric for classification
    else:
        models = [
            ("Decision Tree", DecisionTreeRegressor()),
            ("Random Forest", RandomForestRegressor()),
            ("Linear Regression", LinearRegression())
        ]
        scoring = "neg_mean_squared_error"                                          #scikit-learn uses negative MSE for regression scoring

    results = []                                                                    #To store final results for each model
    for name, model in models:
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)                #Perform 5-fold CV
        score = np.mean(scores)                                                     #Average the scores                
        results.append({    
            "model": name,
            "metric": "Accuracy" if task_type == "classification" else "MSE",
            "value": score if task_type == "classification" else -1 * score
        })
    return results                                                                  #Return list of model results
