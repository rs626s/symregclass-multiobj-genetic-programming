#Standard library and utility imports
import os
from sklearn.model_selection import KFold

#Import project-specific module
from src.loader import load_datasets
from src.evaluator import evaluate_model, evaluate_baselines
from src.gp_model import GeneticProgrammingModel
from src.visualizer import save_results, save_pareto_plot

#Import baseline models for fallback complexity check
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

def run_experiment():
    datasets = load_datasets()                                                  #Load all datasets (regression + classification)
    for name, (X, y, task_type) in datasets.items():
        print(f"\nRunning on dataset: {name}")
        fold_results = []                                                       #To store results from each fold
        gp_all_individuals = []                                                 #To store all GP individuals
        kf = KFold(n_splits=5, shuffle=True, random_state=42)                   #5-fold cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            #Split data into train and test sets for current fold
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            #Run GP model
            gp = GeneticProgrammingModel(task_type=task_type)
            pareto_front = gp.run(X_train, y_train)                             #Train GP and get Pareto front
            gp_all_individuals.extend(pareto_front)                             #Save individuals for plotting
            results = evaluate_model(pareto_front, X_test, y_test, task_type, toolbox=gp.toolbox)       #Evaluate Pareto front on test set
            fold_results.extend(results)

        save_results(name, fold_results)                                        #Save results
        save_pareto_plot(name, gp_all_individuals, pareto_front, task_type)     #Save plot Pareto front

        #Re-train GP on full dataset for final evaluation
        gp_final = GeneticProgrammingModel(task_type=task_type)
        pareto_front_final = gp_final.run(X, y)
        top_results = evaluate_model(pareto_front_final, X, y, task_type, toolbox=gp_final.toolbox)
        print("\nTop GP Models:")
        for i, res in enumerate(top_results[:3]):
            print(f"Model {i+1}:")
            print(f"  Expression: {res['expression']}")
            print(f"  Features used: {res['features_used']}")
        
        baseline_results = evaluate_baselines(X, y, task_type)                  #Evaluate baseline models on full dataset
        top_gp_score = top_results[0]["metric"]                                 #Identify best baseline score and compare with GP
        best_baseline_model = max(baseline_results, key=baseline_results.get if task_type == "classification" else lambda k: -baseline_results[k])
        best_baseline_score = baseline_results[best_baseline_model]

        epsilon = 0.01                                                          #Tolerance for hypothesis comparison

        #Hypothesis check: Is GP as good or better than best baseline?
        if task_type == "classification":
            hypothesis_supported = top_gp_score >= (best_baseline_score - epsilon)
        else:
            hypothesis_supported = top_gp_score <= (best_baseline_score + epsilon)

        #Select baseline model for complexity comparison
        if best_baseline_model == "Decision Tree":
            model = DecisionTreeClassifier() if task_type == "classification" else DecisionTreeRegressor()
        elif best_baseline_model == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            model = RandomForestClassifier() if task_type == "classification" else RandomForestRegressor()
        elif best_baseline_model == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
        else:                                                                   #Linear Regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

        model.fit(X, y)                                                         #Fit baseline model on full dataset

        #Estimate complexity of baseline model
        try:
            if hasattr(model, "tree_"):
                baseline_complexity = model.tree_.node_count                                    #For decision trees    
            elif hasattr(model, "estimators_"):
                baseline_complexity = sum(est.tree_.node_count for est in model.estimators_)    #For forests    
            elif hasattr(model, "coef_"):
                baseline_complexity = sum(model.coef_[0] != 0) if task_type == "classification" else sum(model.coef_ != 0)
            else:
                baseline_complexity = 100                                       #Fallback if unknown model type        
        except:
            baseline_complexity = 100                                           #Fallback in case of exception    

        #Complexity of GP = number of features used in best GP individual
        gp_complexity = top_results[0]["features_used"]
        is_simpler = gp_complexity < baseline_complexity

        #Print hypothesis evaluation result
        print(f"\nHypothesis Evaluation for {name}:")
        print(f"  Accuracy Condition: {'Passed' if hypothesis_supported else 'Failed'}")
        print(f"  Complexity Condition: {'Passed' if is_simpler else 'Failed'}")

        if hypothesis_supported and is_simpler:
            print("  Hypothesis Supported: Both conditions met.")
        else:
            print("  Hypothesis Not Supported: One or more conditions failed.")

        #Print baseline results for comparison
        print("\nBaseline comparisons:")                        
        for model_name, metric in baseline_results.items():
            if task_type == "regression":
                print(f"  {model_name}: MSE = {metric:.4f}")
            else:
                print(f"  {model_name}: Accuracy = {metric:.4f}")

