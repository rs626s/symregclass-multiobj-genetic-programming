import operator                                                     #Provides arithmetic operator functions (add, sub, etc.)
import numpy as np                                                  #For numerical computations
import random                                                       #For generating random constants
import functools                                                    #For creating partial functions (used in ephemeral constants)
from deap import base, creator, gp, tools, algorithms               #DEAP: Evolutionary computation framework
from sklearn.model_selection import KFold                           #For cross-validation
from src.utils import count_nodes                                   #Counts total nodes in a GP individual (used as a complexity measure)
from src.feature_usage import count_features_used                   #Counts how many features are used in the expression

class GeneticProgrammingModel:
    def __init__(self, task_type="regression"):                     
        self.task_type = task_type                                  #Task type: either "regression" or "classification"
        self.toolbox = base.Toolbox()                               #DEAP toolbox to hold genetic operations
        self.pset = gp.PrimitiveSet("MAIN", arity=0)                #Create an initial (empty) primitive set
        self._setup_primitive_set()                                 #Add operators and constants to the primitive set

    def _setup_primitive_set(self):
        self.pset.addPrimitive(operator.add, 2)                     #Add binary addition
        self.pset.addPrimitive(operator.sub, 2)                     #Add binary subtraction
        self.pset.addPrimitive(operator.mul, 2)                     #Add binary multiplication
        self.pset.addPrimitive(operator.neg, 1)                     #Add unary negation    
        self.pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -1, 1))         #Random constant in [-1, 1]
        self.pset.renameArguments(ARG0='x0')                        #Rename the first argument to 'x0'

    def run(self, X, y):
        self.pset = gp.PrimitiveSet("MAIN", X.shape[1])             #Set arity equal to number of input features
        self._setup_primitive_set()                                 #Re-add primitives and arguments
        
        #Create a multi-objective fitness (minimize error and complexity)
        if "FitnessMulti" not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)                #Expression generator
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)   #Create individual
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)            #Create population
        self.toolbox.register("compile", gp.compile, pset=self.pset)                                    #Compile expression tree into function

        def eval_symb_reg(individual):
            func = self.toolbox.compile(expr=individual)            #Convert GP individual to callable function    
            try:
                if count_features_used(individual) == 0:
                    return float("inf"), float("inf")               #Penalize models that use no input features

                y_pred = np.array([func(*x) for x in X])            #Predict outputs for all inputs    
                error = np.mean((y_pred - y) ** 2) if self.task_type == "regression" else \
                        -1 * np.mean(y_pred == y)                   #Accuracy as negative for minimization    
                complexity = count_nodes(individual)                #Count number of nodes (expression complexity)
                return error, complexity                            #Multi-objective fitness
            except:
                return float("inf"), float("inf")                   #If error occurs during evaluation, heavily penalize

        self.toolbox.register("evaluate", eval_symb_reg)            #Register evaluation function
        self.toolbox.register("select", tools.selNSGA2)             #Use NSGA-II selection
        self.toolbox.register("mate", gp.cxOnePoint)                #Use one-point crossover
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)              #Use uniform mutation
        
        #Apply height limit to avoid overly complex trees
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        pop = self.toolbox.population(n=50)                                             #Generate initial population of 50 individuals            
        hof = tools.ParetoFront()                                                       #Store the Pareto-optimal individuals
        
        #Run the evolutionary algorithm with NSGA-II (μ=50, λ=100, 10 generations)
        algorithms.eaMuPlusLambda(pop, self.toolbox, mu=50, lambda_=100, cxpb=0.5, mutpb=0.2, 
                                  ngen=10, stats=None, halloffame=hof, verbose=False)
        return hof                                                                      #Return the final Pareto front of evolved solutions
