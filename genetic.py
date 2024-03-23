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


class Genetic:

    def __init__(self, population_size, pso_population, npp: Instance, offspring_rate, fitness_fun, mutation_rate=0.02,
                 n_threads=None):
        self.pop_size = population_size

        self.pso_population = pso_population
        self.n_paths = npp.n_paths
        self.npp = npp
        self.upper_bounds = npp.upper_bounds
        self.offs_size = int(self.pop_size * offspring_rate)
        self.population = np.zeros((self.pop_size + self.offs_size + self.pso_population, self.n_paths))
        self.combs = list(itertools.combinations(range(self.pop_size), 2))
        self.idx_range = range(self.n_paths)
        self.pop_idx = range(self.pop_size)
        self.fitness_fun = fitness_fun
        self.mutation_rate = mutation_rate
        self.best_val = None
        self.recombination_size = self.n_paths//2

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

    def get_pop_sample(self, n):
        return self.population[np.random.choice(self.pop_size, size=n, replace=False)]
