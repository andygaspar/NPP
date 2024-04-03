import os as os
import random

import numpy as np
import pandas as pd

from Genetic.genetic_ import GeneticPso
from Instance.instance import Instance
# from Net.network_manager import NetworkManager
from Solver.solver import GlobalSolver
from Genetic.genetic_old import GeneticOld

os.system("PSO/install.sh")


columns = ['run', 'commodities', 'paths', 'obj_exact', 'obj_alg', 'gap', 'time_exact', 'time_alg', 'status', 'mip_gap']


POPULATION = 256
off_size = int(POPULATION / 2)
ITERATIONS = 20000
MUTATION_RATE = 0.02

PSO_EVERY = 100000
PSO_SIZE = 4
PSO_SELECTION = 4
PSO_ITERATIONS = 20000
PSO_FINAL_ITERATIONS = 1
NO_UPDATE_LIM = 300

TIME_LIMIT = 60 * 60
VERBOSE = False
N_THREADS = None
row = 0
run = 0


df = pd.DataFrame(columns=columns)
for n_commodities in [20, 56, 90]:
    for n_paths in [20, 56, 90]:
        for run in range(10):
            print("\nProblem", n_commodities, n_paths, run)
            random.seed(run)
            np.random.seed(run)

            npp = Instance(n_paths=n_paths, n_commodities=n_commodities)
            solver = GlobalSolver(npp, verbose=VERBOSE, time_limit=TIME_LIMIT)
            solver.solve()
            print(npp.compute_solution_value(solver.solution_array), solver.obj, 'sol')

            recombination_size = int(n_paths / 2)

            genetic = GeneticPso(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, PSO_SIZE,
                                 PSO_SELECTION, PSO_EVERY, PSO_ITERATIONS, PSO_FINAL_ITERATIONS,
                                 NO_UPDATE_LIM, verbose=VERBOSE, n_threads=N_THREADS, seed=run)
            init_population = np.random.uniform(size=(POPULATION, npp.n_paths)) * npp.upper_bounds

            genetic.run(ITERATIONS, init_population)

            print(genetic.time, solver.time, genetic.best_val, solver.obj, 1 - genetic.best_val/solver.obj)
            gap = 1 - genetic.best_val / solver.obj
            df.loc[row] = [run, n_commodities, n_paths, solver.obj, genetic.best_val, gap, solver.time, genetic.time, solver.m.status,
                           solver.m.MIPGap]
            row += 1

df.to_csv('test.csv', index=False)
