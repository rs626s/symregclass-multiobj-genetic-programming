from deap import gp                                 #Import DEAP's genetic programming module

def count_features_used(individual):                #Function to count the number of distinct input features used in a GP individual
    feature_indices = set()                         #Use a set to store unique feature indices
    for node in individual:                         #Iterate through each node in the individual's tree (expression)    
        if isinstance(node, gp.Terminal) and isinstance(node.value, str) and node.value.startswith("ARG"):
            index = int(node.value[3:])             #Extract the feature index from the string "ARG0", "ARG1", etc.
            feature_indices.add(index)              #Add the index to the set
    return len(feature_indices)                     #Return the number of unique features used
