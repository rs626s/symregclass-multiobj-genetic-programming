import warnings                                                                     #Import Python's warnings module to handle messages
from sklearn.exceptions import ConvergenceWarning                                   #Import specific warning for models like Logistic        
warnings.filterwarnings("ignore", category=ConvergenceWarning)                      #Suppress convergence warnings from scikit-learn
import operator                                                                     #Provides basic mathematical operations like add, sub, etc.
import numpy as np                                                                  #For numerical operations, arrays, and statistics
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score                                 #For cross-validation    
from sklearn.linear_model import LinearRegression, LogisticRegression               #Import linear models for regression and classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor              #Import tree-based models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor          #Random Forest models

def evaluate_model(pareto_front, X_test, y_test, task_type, toolbox):
    results = []                                                                    #Store evaluation results for each individual
    for ind in pareto_front:
        func = toolbox.compile(expr=ind)                                            #Convert GP expression tree into a callable Python function
        try:
            predictions = np.array([func(*x) for x in X_test])                      #Generate predictions by applying the compiled function
        except Exception:
            continue                                                                #Skip if the function throws an error (e.g., division by zero)

        if np.all(predictions == predictions[0]):                                   #Skip models that predict the same value for all inputs
            continue                                                                #Skip constant models

        if task_type == "regression":
            metric = mean_squared_error(y_test, predictions)                        #Use Mean Squared Error for regression    
        else:
            predictions = np.round(predictions).astype(int)                         #Round predictions to nearest integer for classification
            predictions = np.clip(predictions, np.min(y_test), np.max(y_test))      #Ensure predictions are within valid label range
            metric = accuracy_score(y_test, predictions)                            #Use Accuracy for classification

        expr = str(ind)                                                             #Convert individual expression to string
        features_used = sum(('ARG' in token or 'x' in token) for token in expr.split()) #Count number of input features used in the expression
        results.append({
            "expression": expr,
            "features_used": features_used,
            "metric": metric
        })

    results.sort(key=lambda x: x["metric"])                     #Sort results by metric (lower is better for MSE, higher is better for Accuracy)
    return results                                              #Return evaluated results    

def evaluate_baselines(X, y, task_type):
    results = {}                                                #Dictionary to hold average performance for each model        
    if task_type == "regression":
        models = {
            "Decision Tree": DecisionTreeRegressor(),           #Basic decision tree for regression
            "Random Forest": RandomForestRegressor(),           #Ensemble of decision trees
            "Linear Regression": LinearRegression()             #Standard linear regression
        }
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")   #Evaluate model using 5-fold cross-validation
            results[name] = -np.mean(scores)                                                #Convert to actual MSE by negating
    else:
        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")     #Evaluate classification models using 5-fold cross-validation
            results[name] = np.mean(scores)                                     #Average accuracy across folds
    return results                                                              #Return dictionary of model scores    
