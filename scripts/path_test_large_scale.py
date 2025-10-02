import os
import random
import os
import numpy as np
import pandas as pd

from Path.Solver.genetic_heuristic_solver import GeneticHeuristic
from Path.Instance.instance import Instance
from Path.Solver.genetic_solver import Genetic
from Path.Solver.solver import GlobalSolver

# from Net.network_manager import NetworkManager

# os.system("Path/CPP/install.sh")

columns = ['run', 'commodities', 'paths',
           'obj_exact', 'obj_h',
           'GAP_h', 'mip_GAP', 'status',
           'time_exact', 'time_h', 'h_iter', 'partial']

# columns = ['run', 'commodities', 'paths', 'obj_gah', 'obj_ga', 'time_gah', 'time_ga']

POPULATION = 256
off_size = int(POPULATION / 2)
ITERATIONS = 20000
MUTATION_RATE = 0.02

H_EVERY = 100

TIME_LIMIT = 3600
VERBOSE = False
N_THREADS = None
row = 0
run = 0

n_commodities = 180
n_paths = 180

file_name = 'Results/PathResults/path_results_large.csv'

df = pd.DataFrame(columns=columns)
for partial in [True, False]:

    for run in range(10):
        print("\nProblem ", 'Partial' if partial else 'Complete', n_commodities, n_paths, run)
        random.seed(run)
        np.random.seed(run)

        npp = Instance(n_paths=n_paths, n_commodities=n_commodities, partial=partial, seed=run)
        solver = GlobalSolver(npp, verbose=VERBOSE, time_limit=TIME_LIMIT)
        solver.solve()
        # solver.time = 0
        # solver.status = 2
        # solver.obj = 1

        recombination_size = int(n_paths / 2)

        genetic_h = GeneticHeuristic(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, heuristic_every=H_EVERY,
                                     verbose=VERBOSE,
                                     n_threads=N_THREADS, seed=run)

        genetic_h.run(ITERATIONS)
        print(genetic_h.time)
        # g = Genetic(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, verbose=VERBOSE, n_threads=N_THREADS, seed=run)
        #
        # g.run(ITERATIONS)
        # print(g.time, g.best_val)

        gap_g_h = 1 - genetic_h.best_val / solver.obj
        # gap_g = 1 - g.best_val / solver.obj
        print(partial, n_commodities, n_paths, run, genetic_h.heuristic_iterations)
        print(genetic_h.time, solver.time, genetic_h.best_val, solver.obj, 1 - genetic_h.best_val / solver.obj, '\n')
        df.loc[row] = [run, n_commodities, n_paths,
                       solver.obj, genetic_h.best_val,
                       gap_g_h, solver.final_gap, solver.m.status,
                       solver.time, genetic_h.time,  genetic_h.heuristic_iterations, partial]

        # print(genetic_h.time, g.time, genetic_h.best_val, g.best_val)
        #
        # df.loc[row] = [run, n_commodities, n_paths, genetic_h.best_val, g.best_val,
        #                genetic_h.time, g.time]
        row += 1

    df.to_csv(file_name, index=False)
