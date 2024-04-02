import copy
import itertools
import os as os
import random
import time

import numpy as np
import pandas as pd
from Instance.instance import Instance
# from Net.network_manager import NetworkManager
from Solver.solver import GlobalSolver
from Solver.pso_solver import PsoSolverNew
from Genetic.genetic_cpp import GeneticOperators


class GeneticOld:

    def __init__(self, population_size, pso_population, pso_selection, npp: Instance, offspring_rate, mutation_rate=0.02,
                 n_threads=None):
        self.time = None
        self.pop_size = population_size
        self.pso_population = pso_population
        self.offs_size = int(self.pop_size * offspring_rate)
        self.total_pop_size = self.pop_size + self.offs_size + pso_selection

        self.n_paths = npp.n_paths
        self.npp = npp
        self.upper_bounds = npp.upper_bounds

        self.population = np.zeros((self.total_pop_size, self.n_paths))
        self.combs = list(itertools.combinations(range(self.pop_size), 2))
        self.idx_range = range(self.n_paths)
        self.pop_idx = range(self.pop_size)
        self.fitness_fun = npp.compute_solution_value
        self.mutation_rate = mutation_rate
        self.best_val = None
        self.recombination_size = self.n_paths//2

        self.values = np.array([c.c_od for c in self.npp.commodities] +
                               [c.c_od - p for c in self.npp.commodities for p in c.c_p_vector])

        self.operator = GeneticOperators(self.upper_bounds, npp.commodities_tax_free,
                                         npp.n_users,
                                         npp.transfer_costs, npp.n_commodities,
                                         self.pop_size, self.offs_size, self.n_paths,
                                         self.mutation_rate, self.recombination_size, n_threads)
        self.vals = None



    def parallel_generation(self, initial_position=None, pso_particles=None):
        if initial_position is not None:
            self.population[:self.pop_size] = initial_position
            self.vals = np.array([self.fitness_fun(sol) for sol in self.population])
        if pso_particles is not None: # to fix....the eval should be taken from pso
            self.population[self.pop_size + self.offs_size:] = pso_particles

        self.population[self.pop_size: self.pop_size + self.offs_size] = self.operator.generation(
            self.population[:self.pop_size])

        if pso_particles is not None:
            self.vals[self.pop_size:] = (np.array([self.fitness_fun(sol) for sol in self.population[self.pop_size:]]))
        else:
            self.vals[self.pop_size: self.pop_size + self.offs_size] = self.operator.eval_fitness(
                self.population[self.pop_size: self.pop_size + self.offs_size])

        idxs = np.argsort(self.vals)
        self.vals = self.vals[idxs[::-1]]
        self.population = self.population[idxs[::-1]]
        self.best_val = max(self.vals)
        # print(self.vals)

    def generation(self, initial_position=None, pso_particles=None):
        if initial_position is not None:
            self.population[:self.pop_size] = initial_position
            self.vals = np.array([self.fitness_fun(sol) for sol in self.population])
        if pso_particles is not None:
            self.population[self.pop_size + self.offs_size:] = pso_particles

        recombined = np.random.choice(range(len(self.combs)), size=self.offs_size, replace=False)
        recombined = [self.combs[i] for i in recombined]
        for i, comb in enumerate(recombined):
            recombination_idx = np.random.choice(self.idx_range, size=self.recombination_size, replace=False)
            self.population[self.pop_size + i] = self.population[comb[0]]
            self.population[self.pop_size + i][recombination_idx] = self.population[comb[1]][recombination_idx]
            mutation = np.array([1 if np.random.uniform() < self.mutation_rate else 0 for _ in range(self.n_paths)])
            for idx in np.where(mutation == 1)[0]:
                self.population[self.pop_size + i][idx] = np.random.uniform(0, self.upper_bounds[idx])

        if pso_particles is not None:
            self.vals[self.pop_size:] = (np.array([self.fitness_fun(sol) for sol in self.population[self.pop_size:]]))
        else:
            self.vals[self.pop_size: self.pop_size + self.offs_size] = (
                np.array([self.fitness_fun(sol)
                          for sol in self.population[self.pop_size: self.pop_size + self.offs_size]]))

        idxs = np.argsort(self.vals)
        self.vals = self.vals[idxs[::-1]]
        self.population = self.population[idxs[::-1]]
        self.best_val = max(self.vals)

    def init_values(self):
        for i in range(self.n_paths):
            vals = self.values[self.npp.paths[i].L_p <= self.values]
            vals = vals[self.npp.paths[i].N_p >= vals]
            self.population[:self.pop_size, i] = (
                np.random.choice(vals, size=self.pop_size, replace=True))
        self.vals = np.array([self.fitness_fun(sol) for sol in self.population])
    def generation_es(self, initial_position=None):
        if initial_position is not None:
            self.population[:self.pop_size] = initial_position
            self.vals = np.array([self.fitness_fun(sol) for sol in self.population])

        for i in range(self.n_paths):
            self.population[self.pop_size: self.pop_size + self.offs_size, i] = (
                np.random.choice(self.population[:self.pop_size, i], size=self.offs_size, replace=True))
            mutation = np.array([1 if np.random.uniform() < self.mutation_rate else 0
                                 for _ in range((self.pop_size) * self.n_paths)])
            mutation = mutation.reshape((self.pop_size, self.n_paths))
            mutation = np.where(mutation == 1)
            for idx in range(len(mutation)):
                self.population[mutation[idx][0], mutation[idx][1]] = np.random.choice(self.values)

        self.vals[self.pop_size:] = np.array([self.fitness_fun(sol) for sol in self.population[self.pop_size:]])

        idxs = np.argsort(self.vals)
        self.vals = self.vals[idxs[::-1]]
        self.population = self.population[idxs[::-1]]
        self.best_val = max(self.vals)

    def get_pop_sample(self, n):
        return self.population[np.random.choice(self.pop_size, size=n, replace=False)]

    def random_init(self):
        return np.random.uniform(size=(self.pop_size, self.npp.n_paths)) * self.npp.upper_bounds

    def run(self, iterations, pso_n_particles, pso_every, pso_iterations, no_update_lim, pso_selection_size,
            pso_final_iteration, verbose=False, initial_position=None):
        self.time = time.time()
        initial_position = self.random_init() if initial_position is None else initial_position
        self.parallel_generation(initial_position)
        for i in range(1, iterations):

            if i % pso_every == 0:
                pso = PsoSolverNew(self.npp, pso_n_particles, n_iterations=pso_iterations, no_update_lim=no_update_lim)
                pso.run(self.get_pop_sample(pso_n_particles), verbose=False)
                new_particles = pso.get_best_n_particles(pso_selection_size)
                self.parallel_generation(pso_particles=new_particles)
            else:
                self.parallel_generation()
            if verbose and i % 100 == 0:
                # print(genetic.best_val, np.std(np.std(genetic.population, axis=0)))
                print(i, self.best_val)
            # for j in range(self.pop_size):
            #     print(self.population[j], self.vals[j])
            #
            # print('')
        pso = PsoSolverNew(self.npp, self.total_pop_size, pso_final_iteration, no_update_lim)
        pso.run(self.population, verbose=verbose)
        self.best_val = pso.best_val if pso.best_val > self.best_val else self.best_val
        self.time = time.time() - self.time
