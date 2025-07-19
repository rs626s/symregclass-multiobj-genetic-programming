import operator
import numpy as np
import random
import functools
from deap import base, creator, gp, tools, algorithms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from src.utils import count_nodes
from src.feature_usage import count_features_used

class GeneticProgrammingModel:
    def __init__(self, task_type="regression"):
        self.task_type = task_type
        self.toolbox = base.Toolbox()
        self.pset = gp.PrimitiveSet("MAIN", arity=0)
        self._setup_primitive_set()

    def _setup_primitive_set(self):
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(operator.neg, 1)
        self.pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -1, 1))
        self.pset.renameArguments(ARG0='x0')

    def run(self, X, y):
        self.pset = gp.PrimitiveSet("MAIN", X.shape[1])
        self._setup_primitive_set()
        if "FitnessMulti" not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        def eval_symb_reg(individual):
            func = self.toolbox.compile(expr=individual)
            try:
                if count_features_used(individual) == 0:
                    return float("inf"), float("inf")  # Penalize constant models

                y_pred = np.array([func(*x) for x in X])
                error = np.mean((y_pred - y) ** 2) if self.task_type == "regression" else \
                        -1 * np.mean(y_pred == y)
                complexity = count_nodes(individual)
                return error, complexity
            except:
                return float("inf"), float("inf")

        self.toolbox.register("evaluate", eval_symb_reg)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        pop = self.toolbox.population(n=50)
        hof = tools.ParetoFront()
        algorithms.eaMuPlusLambda(pop, self.toolbox, mu=50, lambda_=100, cxpb=0.5, mutpb=0.2, 
                                  ngen=10, stats=None, halloffame=hof, verbose=False)
        return hof
