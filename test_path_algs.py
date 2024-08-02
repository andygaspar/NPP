import os
import random
import numpy as np
from Path.Solver.genetic_heuristic_solver import GeneticHeuristic
from Path.Instance.instance import Instance
from Path.Solver.genetic_pso_solver import GeneticPso
from Path.Solver.genetic_solver import Genetic
from Path.Solver.solver import GlobalSolver

os.system("Path/CPP/install.sh")


TIME_LIMIT = 60 * 60
VERBOSE = False
N_THREADS = None
row = 0
run = 0

n_commodities = 20
n_paths = 20
recombination_size = int(n_paths / 2)

random.seed(run)
np.random.seed(run)

npp = Instance(n_paths=n_paths, n_commodities=n_commodities)
solver = GlobalSolver(npp, verbose=VERBOSE, time_limit=TIME_LIMIT)
solver.solve()


POPULATION = 256
off_size = int(POPULATION / 2)
ITERATIONS = 200000000
MUTATION_RATE = 0.02

g = Genetic(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, verbose=VERBOSE, n_threads=N_THREADS, seed=run)
g.run(ITERATIONS)


H_EVERY = 10

genetic_h = GeneticHeuristic(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, heuristic_every=H_EVERY,
                             verbose=VERBOSE, n_threads=N_THREADS, seed=run)
genetic_h.run(ITERATIONS)


PSO_EVERY = 100
PSO_SIZE = 4
PSO_SELECTION = 4
PSO_ITERATIONS = 20000
PSO_FINAL_ITERATIONS = 1000
NO_UPDATE_LIM = 300

g_pso = GeneticPso(npp, POPULATION, off_size, MUTATION_RATE, recombination_size,
                   PSO_SIZE, PSO_SELECTION, PSO_EVERY, PSO_ITERATIONS, PSO_FINAL_ITERATIONS, NO_UPDATE_LIM, verbose=VERBOSE, n_threads=N_THREADS, seed=run)
g_pso.run(ITERATIONS)

print(g.time, genetic_h.time, g_pso.time, solver.time)
print(g.best_val, genetic_h.best_val, g_pso.best_val, solver.obj)

