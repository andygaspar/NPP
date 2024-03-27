import os
import random
import time

import numpy as np

from Genetic.genetic_cpp import GeneticCpp
from Genetic.genetic_old import GeneticOld
from Instance.instance import Instance
from Solver.solver import GlobalSolver


class GeneticPso:
    def __init__(self, npp: Instance, pop_size, offs_size, mutation_rate, recombination_size,
                 pso_size, pso_selection_size, pso_every, pso_iterations, pso_final_iterations, pso_no_update_limit,
                 verbose=True, n_threads=None, seed=None):
        self.time = None
        self.pop_size = pop_size
        self.offs_size = offs_size
        self.mutation_rate = mutation_rate
        self.recombination_size = recombination_size
        self.pso_size = pso_size
        self.pso_selection_size = pso_selection_size
        self.pso_every = pso_every
        self.pso_iterations = pso_iterations
        self.pso_final_iterations = pso_final_iterations
        self.pso_no_update_limit = pso_no_update_limit
        self.verbose = verbose
        self.seed = seed
        self.num_threads = n_threads

        self.n_paths = npp.n_paths
        self.npp = npp
        self.upper_bounds = npp.upper_bounds
        self.mutation_rate = mutation_rate
        self.best_val = None
        self.time = None

        self.genetic = GeneticCpp(npp.upper_bounds, npp.commodities_tax_free,
                                   npp.n_users, npp.transfer_costs,
                                   npp.n_commodities, npp.n_paths,
                                   self.pop_size, self.offs_size,
                                   self.mutation_rate, self.recombination_size,
                                   self.pso_size, self.pso_selection_size,
                                   self.pso_every, self.pso_iterations,
                                   self.pso_final_iterations, self.pso_no_update_limit,
                                    self.verbose, self.num_threads, self.seed)

    def run(self, iterations):
        self.time = time.time()
        init_population = np.random.uniform(size=(self.pop_size, self.npp.n_paths)) * self.upper_bounds
        self.best_val = self.genetic.run(init_population, iterations)
        self.time = time.time() - self.time



n_paths = 20
n_commodities = 20

POPULATION = 256
off_size = int(POPULATION/2)
iterations = 1000
recombination_size = int(n_paths/2)
MUTATION_RATE = 0.02


pso_every = 150
PSO_SIZE = 32
pso_selection = 4
pso_iterations = 1000
pso_final_iterations = 10000
pso_no_update = 300
os.system("PSO/install.sh")

N_THREADS = None
seed = 0
VERBOSE = True

TIME_LIMIT = 3

np.random.seed(seed)
random.seed(seed)

npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=seed)
# npp.save_problem()
solver = GlobalSolver(npp, verbose=VERBOSE, time_limit=TIME_LIMIT)
solver.solve()


t = time.time()
genetic = GeneticPso(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, PSO_SIZE,
                     pso_selection, pso_every, pso_iterations, pso_final_iterations,
                     pso_no_update, verbose=VERBOSE, n_threads=N_THREADS, seed=-1)


genetic.run(iterations)

print(solver.obj)
print(time.time() - t)


POPULATION = 256
MUTATION_RATE = 0.02
N_THREADS = None
ITERATIONS = 1_000

PSO_RUN = 150
PARTICLES = 32

PSO_ITERATIONS = 1000
PSO_FINAL_ITERATIONS = 10000
ADDITIONAL_PARTICLES = 4
NO_UPDATE_LIM = 1000


genetic_op = GeneticOld(population_size=POPULATION, pso_population=PSO_SIZE, pso_selection= ADDITIONAL_PARTICLES, npp=npp, mutation_rate=MUTATION_RATE,
                              fitness_fun=npp.compute_solution_value, offspring_rate=0.5, n_threads=N_THREADS)

genetic_op.run(ITERATIONS, PARTICLES, PSO_RUN, PSO_ITERATIONS, NO_UPDATE_LIM, ADDITIONAL_PARTICLES, PSO_FINAL_ITERATIONS, VERBOSE)

print(genetic.time, genetic_op.time, solver.time, genetic.best_val, genetic_op.best_val, solver.obj, 1 - genetic.best_val/solver.obj,
      1 - genetic_op.best_val/solver.obj)
