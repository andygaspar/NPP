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
from genetic import Genetic

os.system("PSO/install.sh")


columns = ['run', 'commodities', 'paths', 'obj_exact', 'obj_pso', 'gap', 'time_exact', 'time_pso', 'status', 'mip_gap']


n_iterations = 1000
n_particles = 20_000
no_update_lim = 1000

TIME_LIMIT = 30
VERBOSE = False
row = 0

n_commodities = 20
n_paths = 56
run = 0

print(n_commodities, n_paths, run)
random.seed(run)
np.random.seed(run)

npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=run)
solver = GlobalSolver(npp, verbose=True, time_limit=TIME_LIMIT)
solver.solve()
print("target val", solver.obj)

# solver.print_model()

POPULATION = 256
PARTICLES = 32
PSO_ITERATIONS = 1000
ADDITIONAL_PARTICLES = 4
MUTATION_RATE = 0.02
N_THREADS = None

t = time.time()
pso = PsoSolverNew(npp, POPULATION, 0, no_update_lim)
initial_position = pso.random_init()
npp.compute_solution_value(initial_position[0])
# 7506.375328431998
# 7569.735330211166
particles = np.copy(initial_position)

genetic = Genetic(population_size=initial_position.shape[0], pso_population=ADDITIONAL_PARTICLES, npp=npp,
                  fitness_fun=npp.compute_solution_value, offspring_rate=0.5, n_threads=N_THREADS, mutation_rate=MUTATION_RATE)
genetic.parallel_generation(particles)
for i in range(10_000):

    if i % 100 == 0 and i > 0:
        pso = PsoSolverNew(npp, PARTICLES, n_iterations=PSO_ITERATIONS, no_update_lim=no_update_lim)
        pso.run(genetic.get_pop_sample(PARTICLES), verbose=False)
        new_particles = pso.get_best_n_particles(ADDITIONAL_PARTICLES)
        genetic.parallel_generation(pso_particles=new_particles)
    else:
        genetic.parallel_generation()
    if i % 150 == 0 and i > 0:
        # print(genetic.best_val, np.std(np.std(genetic.population, axis=0)))
        print(genetic.best_val)

pso = PsoSolverNew(npp, 128, n_iterations=PSO_ITERATIONS, no_update_lim=no_update_lim)
pso.run(genetic.population[:128], verbose=True)

PARTICLES = 10000
pso = PsoSolverNew(npp, genetic.pop_size, 1000, no_update_lim)
initial_position = pso.random_init()
pso.run(genetic.population, verbose=True)

t = time.time() - t

print(t, solver.time, pso.best_val, solver.obj, 1 - pso.best_val/solver.obj)
