import os as os
import random

import numpy as np
import pandas as pd

from Solver.genetic_heuristic_solver import GeneticHeuristic
from Solver.genetic_pso_solver import GeneticPso
from Instance.instance import Instance
from Solver.genetic_solver import Genetic
# from Net.network_manager import NetworkManager
from Solver.solver import GlobalSolver

# os.system("CPP/install.sh")


# columns = ['run', 'commodities', 'paths',
#            'obj_exact', 'obj_gapso', 'obj_ga',
#            'GAP_gapso', 'GAP_ga', 'mip_GAP', 'status',
#            'time_exact', 'time_gapso', 'time_ga']


columns = ['run', 'commodities', 'paths', 'obj_gah', 'obj_ga', 'time_gah', 'time_ga']

POPULATION = 256
off_size = int(POPULATION / 2)
ITERATIONS = 2000
MUTATION_RATE = 0.02

PSO_EVERY = 100
PSO_SIZE = 4
PSO_SELECTION = 4
PSO_ITERATIONS = 20000
PSO_FINAL_ITERATIONS = 1000
NO_UPDATE_LIM = 300

H_EVERY = 10

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
            # solver = GlobalSolver(npp, verbose=VERBOSE, time_limit=TIME_LIMIT)
            # solver.solve()
            # print(npp.compute_solution_value(solver.solution_array), solver.obj, 'sol')

            recombination_size = int(n_paths / 2)

            genetic_h = GeneticHeuristic(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, heuristic_every=H_EVERY,
                                       verbose=VERBOSE,
                                       n_threads=N_THREADS, seed=run)

            genetic_h.run(ITERATIONS)

            g = Genetic(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, verbose=VERBOSE, n_threads=N_THREADS, seed=run)

            g.run(ITERATIONS)
            print(g.time, g.best_val)

            # gap_g_pso = 1 - genetic_h.best_val / solver.obj
            # gap_g = 1 - g.best_val / solver.obj

            # print(genetic_h.time, solver.time, genetic_h.best_val, solver.obj, 1 - genetic_h.best_val / solver.obj)
            # gap_g_pso = 1 - genetic_h.best_val / solver.obj
            # gap_g = 1 - g.best_val / solver.obj
            # df.loc[row] = [run, n_commodities, n_paths,
            #                solver.obj, genetic_h.best_val, g.best_val,
            #                gap_g_pso, gap_g, solver.m.MIPGap, solver.m.status,
            #                solver.time, genetic_h.time, g.time]

            print(genetic_h.time, g.time, genetic_h.best_val, g.best_val)

            df.loc[row] = [run, n_commodities, n_paths, genetic_h.best_val, g.best_val,
                           genetic_h.time, g.time]
            row += 1

df.to_csv('results.csv', index=False)
