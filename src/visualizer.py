import json, os                                                             #For saving data and creating directories
import matplotlib.pyplot as plt                                             #For plotting Pareto fronts

def save_results(name, results):
    os.makedirs("results", exist_ok=True)                                   #Create the folder if it doesn't exist
    with open(f"results/{name}_results.json", "w") as f:                    #Define output file path
        json.dump(results, f, indent=2)                                     #Save results in a human-readable format

def save_pareto_plot(name, all_models, pareto_models, task_type):
    os.makedirs("plots", exist_ok=True)                                     #Create the "plots/" directory if needed

    ylabel = "MSE" if task_type == "regression" else "Accuracy"             #Set y-axis label depending on regression or classification task

    x_all = [ind.fitness.values[1] for ind in all_models]                   #x = complexity
    y_all = [ind.fitness.values[0] for ind in all_models]                   #y = error or -accuracy
    
    #Extract complexity and error for only Pareto front models
    x_pareto = [ind.fitness.values[1] for ind in pareto_models]
    y_pareto = [ind.fitness.values[0] for ind in pareto_models]

    plt.figure()                                                            #Start a new figure

    if len(x_all) > 1:                                                      #Plot all models (background GP model points)
        plt.plot(x_all, y_all, 'b--', label="GP Models", alpha=0.5)
        plt.scatter(x_all, y_all, c='blue')                                 #Blue dots for all GP models
    else:
        plt.scatter(x_all, y_all, c='blue', label="GP Models")

    if len(x_pareto) > 1:
        plt.plot(x_pareto, y_pareto, 'ro--', label="Pareto Front")          #Red dashed line with dots
    else:
        plt.scatter(x_pareto, y_pareto, c='red', label="Pareto Front")
    
    # Set axis labels and title
    plt.xlabel("Model Complexity (features used)")
    plt.ylabel(ylabel)
    plt.title(f"Pareto Front - {name}")
    # Add legend and grid
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{name}_pareto.png")                                 #Save plot as PNG image in the "plots/" directory                    
    plt.close()                                                             #Close the plot to avoid overlapping with future plots
