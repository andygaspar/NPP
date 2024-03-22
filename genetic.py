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

    def __init__(self, population_size, pso_population, npp, offspring_rate, fitness_fun, mutation_rate=0.02):
        self.pop_size = population_size

        self.pso_population = pso_population
        self.n_paths = npp.n_paths
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

        self.operator = GeneticOperators(self.upper_bounds, np.array(self.combs).T.flatten(), npp.commodities_tax_free, npp.n_users,
                                         npp.transfer_costs, npp.n_commodities,
                                         self.pop_size, self.offs_size, self.n_paths,
                                         self.mutation_rate, self.recombination_size)

        self.vals = None

    def generation(self, initial_position=None, pso_particles=None):
        if initial_position is not None:
            self.population[:self.pop_size] = initial_position
            self.vals = np.array([self.fitness_fun(sol) for sol in self.population])
        if pso_particles is not None:
            self.population[self.pop_size + self.offs_size:] = pso_particles

        # t = time.time()
        # recombined = np.random.choice(range(len(self.combs)), size=self.offs_size, replace=False)
        # recombined = [self.combs[i] for i in recombined]
        # for i, comb in enumerate(recombined):
        #     recombination_idx = np.random.choice(self.idx_range, size=self.recombination_size, replace=False)
        #     self.population[self.pop_size + i] = self.population[comb[0]]
        #     self.population[self.pop_size + i][recombination_idx] = self.population[comb[1]][recombination_idx]
        #     mutation = np.array([1 if np.random.uniform() < self.mutation_rate else 0 for _ in range(self.n_paths)])
        #     for idx in np.where(mutation == 1)[0]:
        #         self.population[self.pop_size + i][idx] = np.random.uniform(0, self.upper_bounds[idx])
        # for i, comb in enumerate(recombined):
        #     self.operator.generate(self.population[comb[0]], self.population[comb[1]], self.population[self.pop_size + i], self.upper_bounds)
        self.operator.generation(self.population, self.vals[self.pop_size: self.pop_size + self.offs_size])
        print(self.vals[self.pop_size: self.pop_size + self.offs_size])
        print(np.array([self.fitness_fun(sol) for sol in self.population[self.pop_size: self.pop_size + self.offs_size]]))
        print(self.population[self.pop_size: self.pop_size + self.offs_size])

        idxs = np.argsort(self.vals)
        self.population = self.population[idxs[::-1]]
        self.best_val = max(self.vals)

    def get_pop_sample(self, n):
        return self.population[np.random.choice(self.pop_size, size=n, replace=False)]

os.system("PSO/install.sh")


columns = ['run', 'commodities', 'paths', 'obj_exact', 'obj_pso', 'gap', 'time_exact', 'time_pso', 'status', 'mip_gap']


n_iterations = 1000
n_particles = 20_000
no_update_lim = 1000

TIME_LIMIT = 10
VERBOSE = False
row = 0

n_commodities = 2
n_paths = 2
run = 0

print(n_commodities, n_paths, run)
random.seed(run)
np.random.seed(run)

npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=run)
solver = GlobalSolver(npp, verbose=True, time_limit=TIME_LIMIT)
solver.solve()
print("target val", solver.obj)

# solver.print_model()

POPULATION = 128
PARTICLES = 32
PSO_ITERATIONS = 1000
ADDITIONAL_PARTICLES = 8

t = time.time()
pso = PsoSolverNew(npp, POPULATION, 0, no_update_lim)
initial_position = pso.random_init()
npp.compute_solution_value(initial_position[0])

particles = np.copy(initial_position)

genetic = Genetic(population_size=initial_position.shape[0], pso_population=ADDITIONAL_PARTICLES, npp=npp,
                  fitness_fun=npp.compute_solution_value, offspring_rate=0.5)
genetic.generation(particles)
for i in range(1000):

    if i % 100 == 0 and i > 0:
        pso = PsoSolverNew(npp, PARTICLES, n_iterations=PSO_ITERATIONS, no_update_lim=no_update_lim)
        pso.run(genetic.get_pop_sample(PARTICLES), verbose=True)
        new_particles = pso.get_best_n_particles(ADDITIONAL_PARTICLES)
        genetic.generation(pso_particles=new_particles)
    else:
        genetic.generation()
    if i % 10 == 0 and i > 0:
        print(genetic.best_val, np.std(np.std(genetic.population, axis=0)))
pso = PsoSolverNew(npp, 12, n_iterations=PSO_ITERATIONS, no_update_lim=no_update_lim)
pso.run(genetic.get_pop_sample(128), verbose=True)

PARTICLES = 10000
pso = PsoSolverNew(npp, genetic.pop_size, 1000, no_update_lim)
initial_position = pso.random_init()
pso.run(genetic.population, verbose=True)

t = time.time() - t

print(pso.best_val, t, solver.obj, 1 - pso.best_val/solver.obj)