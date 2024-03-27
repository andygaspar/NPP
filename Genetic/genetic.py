import os
import random
import time

import numpy as np

from Genetic.genetic_cpp import GeneticCpp
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
        init_population = np.random.uniform(size=(self.pop_size, self.npp.n_paths)) * 0
        self.genetic.run(init_population, iterations)



n_paths = 10
n_commodities = 10
pop_size = 256
off_size = int(pop_size/2)
iterations = 10000
recombination_size = int(n_paths/2)
mutation_rate = 0.02


pso_every = 100
pso_size = 32
pso_selection = 2
pso_iterations = 10000
pso_final_iterations = 10000
pso_no_update = 300
os.system("PSO/install.sh")

np.random.seed(0)
random.seed(0)

npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=0)
npp.save_problem()
solver = GlobalSolver(npp, verbose=False, time_limit=30)
solver.solve()


t = time.time()
genetic = GeneticPso(npp, pop_size, off_size, mutation_rate, recombination_size, pso_size,
                     pso_selection, pso_every, pso_iterations, pso_final_iterations,
                     pso_no_update, verbose=False, n_threads=12, seed=-1)


genetic.run(iterations)
print(solver.obj)
print(time.time() - t)

print(npp)