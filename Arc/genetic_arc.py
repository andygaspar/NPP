import copy
import itertools
import random
import time

import networkx as nx
import numpy as np

from Arc.ArcInstance.arc_instance import ArcInstance
from Arc.Arc_GA.arc_genetic_cpp import ArcGeneticCpp, ArcGeneticCppHeuristic
from Arc.Heuristic.arc_heuristic import run_arc_heuristic
from Arc.Heuristic.arc_heuristic2 import run_arc_heuristic2


# from Net.network_manager import NetworkManager


class GeneticArc:

    def __init__(self, population_size, npp: ArcInstance, offspring_rate=0.5, mutation_rate=0.02,
                 n_threads=None):
        self.adj_solution = None
        self.mat_solution = None
        self.solution = None
        self.time = None
        self.pop_size = population_size
        self.offs_size = int(self.pop_size * offspring_rate)
        self.total_pop_size = self.pop_size + self.offs_size

        self.n_tolls = npp.n_tolls
        self.npp = copy.deepcopy(npp)
        self.upper_bounds = np.array([p.N_p for p in npp.tolls])
        # self.lower_bounds = np.array([p.L_p for p in g.tolls])
        self.lower_bounds = np.zeros_like(self.upper_bounds)
        self.npp.edges = [p.idx for p in npp.tolls]
        self.toll_idxs_flat = np.array(self.npp.edges).T.flatten()
        self.origins = np.array([commodity.origin for commodity in self.npp.commodities])
        self.destinations = np.array([commodity.destination for commodity in self.npp.commodities])
        self.n_users = np.array([commodity.n_users for commodity in self.npp.commodities])

        self.population = np.zeros((self.total_pop_size, self.n_tolls))
        self.combs = list(itertools.combinations(range(self.pop_size), 2))
        self.idx_range = range(self.n_tolls)
        self.pop_idx = range(self.pop_size)
        self.fitness_fun = npp.compute_obj
        self.mutation_rate = mutation_rate
        self.best_val = None
        self.recombination_size = self.n_tolls // 2

        self.adj = npp.get_adj().copy()
        self.prices = np.zeros_like(npp.get_adj())

        self.genetic_cpp = None

        self.vals = None

    def get_mats(self, sol):
        prices = np.zeros_like(self.adj)
        for i in range(self.n_tolls):
            prices[self.npp.tolls[i].idx] = sol[i]
        adj = self.adj + prices

        return adj, prices

    def generation(self, initial_position=None):
        if initial_position is not None:
            self.population[:self.pop_size] = initial_position
            self.vals = np.array([self.fitness_fun(*self.get_mats(sol)) for sol in self.population])

        recombined = np.random.choice(range(len(self.combs)), size=self.offs_size, replace=False)
        recombined = [self.combs[i] for i in recombined]
        for i, comb in enumerate(recombined):
            recombination_idx = np.random.choice(self.idx_range, size=self.recombination_size, replace=False)
            self.population[self.pop_size + i] = self.population[comb[0]]
            self.population[self.pop_size + i][recombination_idx] = self.population[comb[1]][recombination_idx]
            mutation = np.array([1 if np.random.uniform() < self.mutation_rate else 0 for _ in range(self.n_tolls)])
            for idx in np.where(mutation == 1)[0]:
                self.population[self.pop_size + i][idx] = np.random.uniform(self.lower_bounds[idx], self.upper_bounds[idx])

        self.vals[self.pop_size: self.pop_size + self.offs_size] = (
            np.array([self.fitness_fun(*self.get_mats(sol))
                      for sol in self.population[self.pop_size: self.pop_size + self.offs_size]]))

        idxs = np.argsort(self.vals)
        self.vals = self.vals[idxs[::-1]]
        self.population = self.population[idxs[::-1]]
        self.best_val = max(self.vals)

    def init_values(self, restart=False):
        start = 1 if restart else 0
        population = np.zeros((self.pop_size - start, self.n_tolls))
        for i in range(start, self.n_tolls):
            population[:self.pop_size, i] = np.random.uniform(self.lower_bounds[i], self.upper_bounds[i], size=self.pop_size - start)
        return population

    def get_pop_sample(self, n):
        return self.population[np.random.choice(self.pop_size, size=n, replace=False)]

    def run(self, iterations, verbose=False, initial_position=None):
        self.time = time.time()
        initial_position = self.init_values() if initial_position is None else initial_position
        self.generation(initial_position)
        for i in range(1, iterations):
            self.generation()
            std = np.std(self.vals[:self.pop_size])
            if std < 1:
                self.population[1:self.pop_size] = self.init_values(restart=True)
            if verbose and i % 1 == 0:
                print(i, self.best_val, np.mean(self.vals[:self.pop_size]), std)
            if i % 10 == 0:
                for j in range(self.pop_size):
                    self.population[j], self.vals[j] = run_arc_heuristic(self.npp, *self.get_mats(self.population[j]))
        self.time = time.time() - self.time

    def run_cpp(self, iterations, verbose, n_threads, seed=None, initial_position=None):
        self.time = time.time()

        self.genetic_cpp = ArcGeneticCpp(self.upper_bounds, self.lower_bounds, self.adj, self.toll_idxs_flat, self.n_users, self.origins,
                                         self.destinations, self.npp.n_commodities,
                                         self.npp.n_tolls, self.pop_size, self.offs_size, self.mutation_rate, self.recombination_size,
                                         verbose, n_threads, seed)
        if initial_position is None:
            initial_position = np.ascontiguousarray(self.init_values())
        else:
            initial_position = np.ascontiguousarray(initial_position)
        self.best_val = self.genetic_cpp.run(initial_position, iterations)
        self.population, self.vals = self.genetic_cpp.get_results()
        self.solution = dict(zip(self.npp.edges, self.population[0]))
        self.adj_solution, self.prices = self.get_mats(self.population[0])
        for i, toll in enumerate(self.npp.edges):
            self.npp.g.edges[toll]['price'] = self.solution[toll]
            self.npp.g.edges[toll]['cost'] = self.npp.g.edges[toll]['weight'] + self.solution[toll]
        self.npp.assign_paths(self.adj_solution, self.prices)
        self.time = time.time() - self.time

    def re_run_h(self, initial_position, iterations):
        self.time = time.time()
        self.best_val = self.genetic_cpp.run(initial_position, iterations)
        self.population, self.vals = self.genetic_cpp.get_results()
        self.solution = self.population[0]
        self.adj_solution, self.prices = self.get_mats(self.solution)
        self.solution = dict(zip(self.npp.arc_tolls, self.population[0]))
        for toll in self.npp.arc_tolls:
            self.npp.g.edges[toll]['price'] = self.solution[toll]
            self.npp.g.edges[toll]['cost'] = self.npp.g.edges[toll]['weight'] + self.solution[toll]
        self.npp.assign_paths(self.adj_solution, self.prices)
        self.time = time.time() - self.time

    def run_cpp_heuristic(self, iterations, dijkstra_every, verbose, n_threads, seed=None, initial_position=None):
        self.time = time.time()

        self.genetic_cpp = ArcGeneticCppHeuristic(self.upper_bounds, self.lower_bounds, self.adj, self.toll_idxs_flat, self.n_users,
                                                  self.origins,
                                                  self.destinations, self.npp.n_commodities,
                                                  self.npp.n_tolls, self.pop_size, self.offs_size, self.mutation_rate,
                                                  self.recombination_size, dijkstra_every,
                                                  verbose, n_threads, seed)
        if initial_position is None:
            initial_position = self.init_values()
        else:
            initial_position = np.ascontiguousarray(initial_position)
        self.best_val = self.genetic_cpp.run(initial_position, iterations)
        self.population, self.vals = self.genetic_cpp.get_results()
        self.solution = self.population[0]
        self.adj_solution, self.prices = self.get_mats(self.solution)
        self.solution = dict(zip(self.npp.arc_tolls, self.population[0]))
        for toll in self.npp.arc_tolls:
            self.npp.g.edges[toll]['price'] = self.solution[toll]
            self.npp.g.edges[toll]['cost'] = self.npp.g.edges[toll]['weight'] + self.solution[toll]
        self.npp.assign_paths(self.adj_solution, self.prices)
        self.time = time.time() - self.time

    def save_population(self, folder):
        np.savetxt(folder + '/population.csv', self.population, fmt='%.18f')
