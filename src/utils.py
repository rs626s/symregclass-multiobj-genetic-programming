import os                                                       #For creating directories and handling file paths
import json                                                     #For saving results in JSON format

def count_nodes(individual):
    return len(individual)                                      #Each node in the tree corresponds to one operation, input, or constant            

def save_results(name, results):
    os.makedirs("results", exist_ok=True)                       #Create "results" folder if it doesn't exist
    with open(f"results/{name}_results.json", "w") as f:        #Open the target JSON file for writing    
        json.dump(results, f, indent=2)                         #Write results with indentation for readability
