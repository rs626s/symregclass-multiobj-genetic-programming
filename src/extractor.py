def extract_expression(individual, pset):
    """Convert tree to readable expression, showing renamed arguments like x0, x1."""
    expr = str(individual)
    for i, name in enumerate(pset.arguments):
        expr = expr.replace(f"ARG{i}", name)
    return expr
