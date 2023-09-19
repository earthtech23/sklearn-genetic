import numbers
import multiprocess
import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.metrics import check_scoring
from sklearn.feature_selection import SelectorMixin
from sklearn.utils._joblib import cpu_count
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# Creating the DEAP creators
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)

def _eaFunction(population, toolbox, cxpb, mutpb, ngen, ngen_no_change=None, stats=None,
                halloffame=None, verbose=0):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("The 'halloffame' parameter should not be None.")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    wait = 0
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Add the best back to population
        offspring.extend(halloffame.items)

        # Get the previous best individual before updating the hall of fame
        prev_best = halloffame[0]

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # If the new best individual is the same as the previous best individual,
        # increment a counter, otherwise reset the counter
        if halloffame[0] == prev_best:
            wait += 1
        else:
            wait = 0

        # If the counter reached the termination criteria, stop the optimization
        if ngen_no_change is not None and wait >= ngen_no_change:
            break

    return population, logbook

def _createIndividual(icls, n, max_features):
    n_features = np.random.randint(1, max_features + 1)
    genome = ([1] * n_features) + ([0] * (n - n_features))
    np.random.shuffle(genome)
    return icls(genome)

def _evalFunction(individual, estimator, X, y, groups, cv, scorer, fit_params, max_features,
                  caching, scores_cache={}):
    individual_sum = np.sum(individual, axis=0)
    if individual_sum == 0 or individual_sum > max_features:
        return -10000, individual_sum, 10000
    individual_tuple = tuple(individual)
    if caching and individual_tuple in scores_cache:
        return scores_cache[individual_tuple][0], individual_sum, scores_cache[individual_tuple][1]
    X_selected = X[:, np.array(individual, dtype=bool)]
    scores = cross_val_score(estimator=estimator, X=X_selected, y=y, groups=groups, scoring=scorer,
                             cv=cv, fit_params=fit_params)
    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    if caching:
        scores_cache[individual_tuple] = [scores_mean, scores_std]
    return scores_mean, individual_sum, scores_std

class GeneticSelectionCV(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    def __init__(self, estimator, cv=None, scoring=None, fit_params=None, max_features=None,
                 verbose=0, n_jobs=1, n_population=300, crossover_proba=0.5, mutation_proba=0.2,
                 n_generations=40, crossover_independent_proba=0.1,
                 mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=None,
                 caching=False, random_seed=None):  # Added random_seed argument
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.fit_params = fit_params
        self.max_features = max_features
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_population = n_population
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.n_generations = n_generations
        self.crossover_independent_proba = crossover_independent_proba
        self.mutation_independent_proba = mutation_independent_proba
        self.tournament_size = tournament_size
        self.n_gen_no_change = n_gen_no_change
        self.caching = caching
        self.scores_cache = {}
        self.random_seed = random_seed  # Added random_seed attribute

    def _initialize_random_seed(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def fit(self, X, y, groups=None):
        return self._fit(X, y, groups)

    def _fit(self, X, y, groups=None):
        X, y = check_X_y(X, y, "csr")
        self._initialize_random_seed()  # Initialize the random seed
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        if self.max_features is not None:
            if not isinstance(self.max_features, numbers.Integral):
                raise TypeError("'max_features' should be an integer between 1 and {} features."
                                " Got {!r} instead."
                                .format(n_features, self.max_features))
            elif self.max_features < 1 or self.max_features > n_features:
                raise ValueError("'max_features' should be between 1 and {} features."
                                 " Got {} instead."
                                 .format(n_features, self.max_features))
            max_features = self.max_features
        else:
            max_features = n_features

        if not isinstance(self.n_gen_no_change, (numbers.Integral, np.integer, type(None))):
            raise ValueError("'n_gen_no_change' should either be None or an integer."
                             " {} was passed."
                             .format(self.n_gen_no_change))

        estimator = clone(self.estimator)

        # Genetic Algorithm
        toolbox = base.Toolbox()

        toolbox.register("individual", _createIndividual, creator.Individual, n=n_features,
                         max_features=max_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", _evalFunction, estimator=estimator, X=X, y=y,
                         groups=groups, cv=cv, scorer=scorer, fit_params=self.fit_params,
                         max_features=max_features, caching=self.caching,
                         scores_cache=self.scores_cache)
        toolbox.register("mate", tools.cxUniform, indpb=self.crossover_independent_proba)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_independent_proba)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        if self.n_jobs == 0:
            raise ValueError("n_jobs == 0 has no meaning.")
        elif self.n_jobs > 1:
            pool = multiprocess.Pool(processes=self.n_jobs)
            toolbox.register("map", pool.map)
        elif self.n_jobs < 0:
            pool = multiprocess.Pool(processes=max(cpu_count() + 1 + self.n_jobs, 1))
            toolbox.register("map", pool.map)

        pop = toolbox.population(n=self.n_population)
        hof = tools.HallOfFame(1, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        if self.verbose > 0:
            print("Selecting features with genetic algorithm.")

        with np.printoptions(precision=6, suppress=True, sign=" "):
            _, log = _eaFunction(pop, toolbox, cxpb=self.crossover_proba,
                                 mutpb=self.mutation_proba, ngen=self.n_generations,
                                 ngen_no_change=self.n_gen_no_change,
                                 stats=stats, halloffame=hof, verbose=self.verbose)
        if self.n_jobs != 1:
            pool.close()
            pool.join()

        support_ = np.array(hof, dtype=bool)[0]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, support_], y)

        self.generation_scores_ = np.array([score for score, _, _ in log.select("max")])
        self.n_features_ = support_.sum()
        self.support_ = support_

        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        return self.support_

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))
