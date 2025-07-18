import operator
import numpy as np
import random
from functools import partial
import warnings
import math
from deap import algorithms, base, creator, tools, gp
from sklearn.base import BaseEstimator
from loader import cfg
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
import logging

def protected_div(x, y):
    try:
        return x / y if abs(y) > 1e-6 else 1.0
    except:
        return 1.0

def population_diversity(population):
    return len(set(str(ind) for ind in population)) / len(population)

def early_stopping(fitness_log, patience=10, delta=1e-4):
    if len(fitness_log) <= patience:
        return False
    return max(fitness_log[-patience:]) - min(fitness_log[-patience:]) < delta

def simplify_individual(ind, pset):
    return gp.prune(ind, pset)

class GPModel(BaseEstimator):
    def __init__(self, mode='regression', **kwargs):
        self.mode = mode.lower()
        if self.mode not in ['regression', 'classification']:
            raise ValueError("Mode must be 'regression' or 'classification'")
        
        self.toolbox = base.Toolbox()
        self.stats = None
        self.feature_names = None
        self.pset = None
        self.best_individuals_ = None
        self.pareto_front = None

    @staticmethod
    def format_stat(stat_list):
        return "[" + ", ".join(f"{x:.4f}" for x in stat_list) + "]"

    def _setup_stats(self):
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_fit.register("avg", np.mean, axis=0)
        stats_fit.register("std", np.std, axis=0)
        stats_fit.register("min", np.min, axis=0)
        stats_fit.register("max", np.max, axis=0)
        return stats_fit

    def _setup_components(self):
        self.pset = gp.PrimitiveSet("MAIN", cfg["input_dim"])
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(protected_div, 2)
        self.pset.addEphemeralConstant("rand101", partial(random.uniform, -1, 1))
        
        for i in range(cfg["input_dim"]):
            self.pset.renameArguments(**{f'ARG{i}': f'x{i}'})

        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=3)
        self.toolbox.register("expr_mut", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.register("mutate_node", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.register("mutate_shrink", gp.mutShrink)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    def _evaluate(self, individual, X, y):
        func = self.toolbox.compile(expr=individual)

        try:
            y_pred = np.array([func(*x) for x in X])
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                logging.warning(f"Invalid predictions from individual: {individual}")
                return (1.0, 1e6, 1.0)

            if self.mode == "regression":
                error = mean_squared_error(y, y_pred)
                r2 = max(-1.0, r2_score(y, y_pred))
                return (error, -r2, len(individual))
            else:
                y_pred = 1 / (1 + np.exp(-y_pred))
                y_pred_class = np.round(y_pred)
                accuracy = accuracy_score(y, y_pred_class)
                f1 = f1_score(y, y_pred_class, average='weighted')
                return (1.0 - accuracy, 1.0 - f1, len(individual))
        except Exception as e:
            logging.warning(f"Exception evaluating individual: {individual} | Error: {e}")
            return (1.0, 1e6, 1.0)

    def fit(self, X, y, feature_names=None):
        self.feature_names = list(feature_names) if feature_names is not None else None
        cfg["input_dim"] = X.shape[1]
        logging.info(f"GP input_dim set to {cfg['input_dim']}")

        self._setup_components()
        self.stats = self._setup_stats()

        pop = self.toolbox.population(n=cfg["gp"]["population_size"])
        hof = tools.ParetoFront()
        self.toolbox.register("evaluate", lambda ind: self._evaluate(ind, X, y))
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

        fitness_log = []
        for gen in range(cfg["gp"]["generations"]):
            pop, log = algorithms.eaMuPlusLambda(
                population=pop,
                toolbox=self.toolbox,
                mu=cfg["gp"]["population_size"],
                lambda_=cfg["gp"]["population_size"],
                cxpb=cfg["gp"]["crossover_rate"],
                mutpb=cfg["gp"]["mutation_rate"],
                ngen=1,
                stats=self.stats,
                halloffame=hof,
                verbose=False
            )

            best = min(pop, key=lambda ind: ind.fitness.values[0])
            fitness_log.append(best.fitness.values[0])
            diversity = population_diversity(pop)
            print(f"Gen {gen}: Diversity = {diversity:.4f}")

            if early_stopping(fitness_log):
                print(f"Stopping early at generation {gen} due to fitness stagnation")
                break

        self.best_individuals_ = hof
        self.pareto_front = self.best_individuals_
        return self

    def predict(self, X):
        if not self.best_individuals_:
            raise ValueError("Model not trained.")
        func = self.toolbox.compile(expr=self.best_individuals_[0])
        return np.array([func(*x) for x in X])

    def export_expressions(self):
        return [str(ind) for ind in getattr(self, "best_individuals_", [])]

    def get_best_equation(self):
        if not self.best_individuals_:
            return "No equation found."
        expr = str(self.best_individuals_[0])
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                expr = expr.replace(f"x{i}", name)
        return expr

    def get_complexity_metrics(self):
        if not self.best_individuals_:
            return (None, "No individuals")
        best = self.best_individuals_[0]
        return (len(best), best.height)
