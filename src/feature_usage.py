from deap import gp

def count_features_used(individual):
    """Count number of distinct input features (x0, x1...) used."""
    feature_indices = set()
    for node in individual:
        if isinstance(node, gp.Terminal) and isinstance(node.value, str) and node.value.startswith("ARG"):
            index = int(node.value[3:])
            feature_indices.add(index)
    return len(feature_indices)
