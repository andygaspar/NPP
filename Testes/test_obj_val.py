import os
import random
import time

import numpy as np
from Instance.instance import Instance
from Solver.genetic_heuristic_solver import GeneticHeuristic
from Solver.genetic_pso_solver import GeneticPso
from Solver.genetic_solver import Genetic
from Solver.solver import GlobalSolver
from Heuristic.heuristic import improve_solution_3

n_paths = 90
n_commodities = 90
VERBOSE = True
random.seed(0)
np.random.seed(0)
tol = 1e-9

os.system("CPP/install.sh")

npp = Instance(n_paths=n_paths, n_commodities=n_commodities)
# solver = GlobalSolver(npp, verbose=VERBOSE)
# solver.solve()
#
# print(solver.obj)

# print(npp.compute_solution_value_with_tol(solver.solution_array, tol=tol))
# print(npp.compute_solution_value(solver.solution_array))


# t = solver.solution_array
#
# costs = np.zeros((n_commodities, n_paths + 1))
# idxs = np.zeros(n_commodities, dtype=int)
# for i, commodity in enumerate(npp.commodities):
#     p = np.array([solver.p[path, commodity].x for path in npp.paths])
#     idx = np.argmax(p)
#     costs[i, :-1] = t + commodity.c_p_vector
#     costs[i, -1] = commodity.c_od
#     idxs[i] = idx


POPULATION = 256
off_size = int(POPULATION / 2)
ITERATIONS = 2000
MUTATION_RATE = 0.02

PSO_EVERY = 100
PSO_SIZE = 36
PSO_SELECTION = 4
PSO_ITERATIONS = 1000
PSO_FINAL_ITERATIONS = 2000
NO_UPDATE_LIM = 300

TIME_LIMIT = 60 * 60
VERBOSE = False
N_THREADS = 1
row = 0
run = 0

recombination_size = int(n_paths / 2)

# genetic = GeneticPso(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, PSO_SIZE,
#                      PSO_SELECTION, PSO_EVERY, PSO_ITERATIONS, PSO_FINAL_ITERATIONS,
#                      NO_UPDATE_LIM, verbose=VERBOSE, n_threads=N_THREADS, seed=run)
#
# genetic.run(ITERATIONS + 1000)

g = Genetic(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, verbose=VERBOSE, n_threads=N_THREADS, seed=run)

g.run(ITERATIONS*4)
print(g.time, g.best_val)
# new_pop = np.zeros_like(g.final_population)
# v = np.zeros(g.pop_size)
# for i, p in enumerate(g.final_population):
#     new_pop[i], v[i] = improve_solution_3(npp, p, tol=tol)
# print(max(v))
#
# t = time.time()
# gg = Genetic(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, verbose=VERBOSE, n_threads=N_THREADS, seed=run)
#
# gg.run(1000)
# v = np.zeros(g.pop_size)
# for i, p in enumerate(g.final_population):
#     new_pop[i], v[i] = improve_solution_3(npp, p, tol=tol)
# gg.run(1000, new_pop)
# print(time.time() - t, gg.best_val)

h_every = 10
ggg = GeneticHeuristic(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, heuristic_every=h_every, verbose=VERBOSE,
                       n_threads=N_THREADS, seed=run)

ggg.run(ITERATIONS*2)
print(ggg.time, ggg.best_val)
