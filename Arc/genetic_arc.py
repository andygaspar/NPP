import copy
import itertools
import random
import time

import networkx as nx
import numpy as np

from Arc.ArcInstance.arc_instance import ArcInstance
from Arc.Arc_GA.arc_genetic_cpp import ArcGeneticCpp
from Arc.Heuristic.arc_heuristic import run_arc_heuristic


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
        # self.lower_bounds = np.array([p.L_p for p in npp.tolls])
        self.lower_bounds = np.zeros_like(self.upper_bounds)
        self.tolls_idxs = [p.idx for p in npp.tolls]
        self.toll_idxs_flat = np.array(self.tolls_idxs).T.flatten()
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
        for i in range(self.n_tolls):
            self.prices[self.npp.tolls[i].idx] = sol[i]
        adj = self.adj + self.prices

        return adj, self.prices

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
                # print(genetic.best_val, np.std(np.std(genetic.population, axis=0)))

                print(i, self.best_val, np.mean(self.vals[:self.pop_size]), std)
            if i % 10 == 0:
                for j in range(self.pop_size):
                    self.population[j], self.vals[j] = run_arc_heuristic(self.npp, *self.get_mats(self.population[j]))
            # for j in range(self.pop_size):
            #     print(self.population[j], self.vals[j])
            #
            # print('')
        self.time = time.time() - self.time

    def run_cpp(self, iterations, verbose, n_threads, seed=None):
        self.time = time.time()

        self.genetic_cpp = ArcGeneticCpp(self.upper_bounds, self.lower_bounds, self.adj, self.toll_idxs_flat, self.n_users, self.origins,
                                         self.destinations, self.npp.n_commodities,
                                         self.npp.n_tolls, self.pop_size, self.offs_size, self.mutation_rate, self.recombination_size,
                                         verbose, n_threads, seed)
        initial_position = self.init_values()
        self.best_val = self.genetic_cpp.run(initial_position, iterations)
        self.population, self.vals = self.genetic_cpp.get_results()
        self.solution = self.population[0]
        self.adj_solution, self.mat_solution = self.get_mats(self.solution)
        self.npp.npp = nx.from_numpy_array(self.adj_solution)
        for c in self.npp.commodities:
            c.solution_path = nx.shortest_path(self.npp.npp, c.origin, c.destination, weight='weight')
            c.solution_edges = [(c.solution_path[i], c.solution_path[i + 1]) for i in range(len(c.solution_path) - 1)]
        self.time = time.time() - self.time

        # (self, upper_bounds, lower_bounds, adj, tolls_idxs, n_users: np.array, origins, destinations,
        # n_commodities, n_tolls,
        # pop_size, offs_size, mutation_rate, recombination_size,
        # verbose, n_threads=None, seed=None)

    def generate_cpp(self):
        new_line = '};\n'

        code = 'short n_commodities = ' + str(self.npp.n_commodities) + ';\n'
        code += 'short n_tolls = ' + str(self.npp.n_tolls) + ';\n'

        code += 'double adj_[] = {'
        for i in range(self.adj.shape[0]):
            for j in range(self.adj.shape[0]):
                code += str(self.adj[i, j]) + ','

        code = code[:-1] + new_line

        code += 'int adj_size = ' + str(self.adj.shape[0]) + ':\n'

        code += 'double upper_bounds_[] = {'
        for i in range(self.upper_bounds.shape[0]):
            code += str(self.upper_bounds[i]) + ','
        code = code[:-1] + new_line

        toll_idxs = np.array(self.tolls_idxs).T.flatten()

        code += 'int toll_idxs_[]= {'
        for t in toll_idxs:
            code += str(t) + ','
        code = code[:-1] + new_line

        origins = np.array([commodity.origin for commodity in self.npp.commodities])
        code += 'int origins_[]= {'
        for t in origins:
            code += str(t) + ','
        code = code[:-1] + new_line

        destinations = np.array([commodity.destination for commodity in self.npp.commodities])
        code += 'int destinations_[]= {'
        for t in destinations:
            code += str(t) + ','
        code = code[:-1] + new_line

        n_users = np.array([commodity.n_users for commodity in self.npp.commodities])
        code += 'int n_users_[]= {'
        for t in n_users:
            code += str(t) + ','
        code = code[:-1] + new_line
        print(code)
    def run_cpp_h(self, iterations, verbose, n_threads, seed=None):
        self.time = time.time()


        self.population[:self.pop_size] = self.init_values()
        every = 300

        self.vals = np.zeros(self.total_pop_size)
        self.vals[:self.pop_size] = np.array([self.fitness_fun(*self.get_mats(sol)) for sol in self.population[:self.pop_size]])
        for i in range(iterations//every):
            self.genetic_cpp = ArcGeneticCpp(self.upper_bounds, self.lower_bounds, self.adj, self.toll_idxs_flat, self.n_users,
                                             self.origins,
                                             self.destinations, self.npp.n_commodities,
                                             self.npp.n_tolls, self.pop_size, self.offs_size, self.mutation_rate, self.recombination_size,
                                             verbose, n_threads, seed)
            self.best_val = self.genetic_cpp.run(self.population[: self.pop_size], every)
            self.population[:self.pop_size], self.vals[:self.pop_size] = self.genetic_cpp.get_results()
            self.solution = self.population[0]
            self.adj_solution, self.mat_solution = self.get_mats(self.solution)
            for j in [0] + random.choices(range(1, self.pop_size), k=5):
                self.population[j], self.vals[j] = run_arc_heuristic(self.npp, *self.get_mats(self.population[j]), 1e-16)
            idx = np.argsort(self.vals[:self.pop_size])[::-1]
            self.population[:self.pop_size] = self.population[idx]
            self.vals[:self.pop_size] = self.vals[idx]
            # curren_val = self.vals[0]
            # t = time.time()
            # for j in range(1, self.pop_size):
            #     if curren_val - 1e-9 <= self.vals[j] <= curren_val + 1e-9:
            #         new_individual = np.zeros(self.n_tolls)
            #         for k in range(self.n_tolls):
            #             new_individual[k] = np.random.uniform(self.lower_bounds[k], self.upper_bounds[k])
            #         self.population[j], self.vals[j] = run_arc_heuristic(self.npp, *self.get_mats(new_individual), 1e-16)
            #     else:
            #         curren_val = self.vals[j]
            print(max(self.vals[:self.pop_size]))
            # print(self.vals[:self.pop_size])
        self.solution = self.population[0]
        self.adj_solution, self.mat_solution = self.get_mats(self.solution)
        self.time = time.time() - self.time
