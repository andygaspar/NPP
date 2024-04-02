import os as os
import random

import numpy as np
import pandas as pd
from Instance.instance import Instance
# from Net.network_manager import NetworkManager
from Solver.solver import GlobalSolver
from Genetic.genetic_old import GeneticOld

os.system("PSO/install.sh")


columns = ['run', 'commodities', 'paths', 'obj_exact', 'obj_alg', 'gap', 'time_exact', 'time_alg', 'status', 'mip_gap']


POPULATION = 256
MUTATION_RATE = 0.02
N_THREADS = 4
ITERATIONS = 1_000

PSO_RUN = 1500
PARTICLES = 32

PSO_ITERATIONS = 1000
PSO_FINAL_ITERATIONS = 1
ADDITIONAL_PARTICLES = 4
NO_UPDATE_LIM = 1000

TIME_LIMIT = 60 * 2
VERBOSE = False
row = 0
run = 0

df = pd.DataFrame(columns=columns)
for n_commodities in [20, 56, 90]:
    for n_paths in [20, 56, 90]:
        for run in range(5):
            print("\n", n_commodities, n_paths, run)
            random.seed(run)
            np.random.seed(run)

            npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=run)
            solver = GlobalSolver(npp, verbose=VERBOSE, time_limit=TIME_LIMIT)
            solver.solve()
            print("target val", solver.obj)

            genetic = GeneticOld(population_size=POPULATION, pso_population=ADDITIONAL_PARTICLES,
                                 pso_selection=0, npp=npp, mutation_rate=MUTATION_RATE,
                                    offspring_rate=0.5, n_threads=N_THREADS)

            genetic.run(ITERATIONS, PARTICLES, PSO_RUN, PSO_ITERATIONS, NO_UPDATE_LIM, ADDITIONAL_PARTICLES, PSO_FINAL_ITERATIONS, VERBOSE)

            print(genetic.time, solver.time, genetic.best_val, solver.obj, 1 - genetic.best_val/solver.obj)
            gap = 1 - genetic.best_val / solver.obj
            df.loc[row] = [run, n_commodities, n_paths, solver.obj, genetic.best_val, gap, solver.time, genetic.time, solver.m.status,
                           solver.m.MIPGap]
            row += 1

df.to_csv('test.csv', index=False)
