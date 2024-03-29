import os
import random
import time

import numpy as np

from Genetic.genetic_cpp import GeneticCpp
from Genetic.genetic_old import GeneticOld
from Instance.instance import Instance
from Solver.pso_solver import PsoSolverNew
from Solver.solver import GlobalSolver
from heuristic import improve_solution


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

        self.final_population = None
        self.final_vals = None

        self.genetic = GeneticCpp(npp.upper_bounds, npp.lower_bounds,
                                  npp.commodities_tax_free,
                                  npp.n_users, npp.transfer_costs,
                                  npp.n_commodities, npp.n_paths,
                                  self.pop_size, self.offs_size,
                                  self.mutation_rate, self.recombination_size,
                                  self.pso_size, self.pso_selection_size,
                                  self.pso_every, self.pso_iterations,
                                  self.pso_final_iterations, self.pso_no_update_limit,
                                  self.verbose, self.num_threads, self.seed)

    def run(self, iterations, init_population=None):
        self.time = time.time()
        init_population = np.random.uniform(
            size=(self.pop_size, self.npp.n_paths)) * self.upper_bounds if init_population is None else init_population
        self.best_val = self.genetic.run(init_population, iterations)
        self.time = time.time() - self.time
        self.final_population, self.final_vals = self.genetic.get_results()


n_paths = 90
n_commodities = 90

POPULATION = 256
off_size = int(POPULATION / 2)
ITERATIONS = 20000
recombination_size = int(n_paths / 2)
MUTATION_RATE = 0.02

PSO_EVERY = 10000
PSO_SIZE = 4
PSO_SELECTION = 4
PSO_ITERATIONS = 20000
PSO_FINAL_ITERATIONS = 1
NO_UPDATE_LIM = 300
os.system("PSO/install.sh")

N_THREADS = None
seed = 6
VERBOSE = True

np.set_printoptions(edgeitems=30, linewidth=100000)


TIME_LIMIT = 100

np.random.seed(seed)
random.seed(seed)

solutions = np.zeros((2, n_paths))
solver_solution = 1

npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=seed)

solver = GlobalSolver(npp, verbose=True, time_limit=TIME_LIMIT)
solver.solve()
solver_solution = solver.obj

genetic = GeneticPso(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, PSO_SIZE,
                     PSO_SELECTION, PSO_EVERY, PSO_ITERATIONS, PSO_FINAL_ITERATIONS,
                     NO_UPDATE_LIM, verbose=VERBOSE, n_threads=N_THREADS, seed=-1)
init_population = np.random.uniform(size=(POPULATION, npp.n_paths)) * npp.upper_bounds

genetic.run(ITERATIONS, init_population)
solutions[0] = genetic.final_population[0]
#
t = time.time()
island_size = 32
start = np.zeros((POPULATION, npp.n_paths))
vals = np.zeros(POPULATION)
for i in range(island_size):
    genetic_op = GeneticPso(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, PSO_SIZE,
                            PSO_SELECTION, 15000, PSO_ITERATIONS, 0,
                            NO_UPDATE_LIM, verbose=False, n_threads=N_THREADS, seed=-1)

    genetic_op.run(500)

    start[i] = improve_solution(npp, genetic_op.final_population[0], genetic.best_val)
    vals[i] = genetic_op.best_val

start[island_size:] = np.random.uniform(size=(POPULATION - island_size, npp.n_paths)) * npp.upper_bounds

genetic_op = GeneticPso(npp, start.shape[0], start.shape[0] // 2, MUTATION_RATE, recombination_size, PSO_SIZE,
                        PSO_SELECTION, 15000, PSO_ITERATIONS, 0,
                        NO_UPDATE_LIM, verbose=True, n_threads=N_THREADS, seed=-1)
print('final')
genetic_op.run(ITERATIONS, start)

solutions[1] = genetic_op.final_population[0]

t = time.time() - t
# print(max(vals))
print(genetic.time, t, genetic.best_val, genetic_op.best_val, 1 - genetic.best_val / genetic_op.best_val)



# print(genetic.time, genetic_op.time, solver.time, genetic.best_val, genetic_op.best_val, 1 - genetic.best_val / genetic_op.best_val)

# sol = solutions[0].copy()
# sol = np.array([20.66011135, 21.84417224, 20.85962104, 16.64011465, 16.59655319, 18.58551733, 17.82162057, 21.77570462, 18.0644112 , 20.53840836, 14.99435891, 18.84067586, 18.98875125, 16.97326997, 18.04566825, 18.89767318, 21.51951767, 21.47526179, 15.68106649, 15.50911928, 20.14371097, 24.03953721, 22.64490523, 19.57367535, 16.28541261, 15.74296516, 17.42590343, 19.07841297, 20.76669389, 17.19400715, 22.00515044, 21.78049731, 17.73584207, 18.45822701, 17.90289863, 22.79013006, 18.9026711 , 17.77541157, 16.41388888, 16.45744616, 15.09697211, 20.84267093, 16.54744751, 22.98301433, 17.62772858, 21.55578872, 22.93361653, 22.79491045, 20.25835115, 19.54940771, 19.52487206, 17.53526016, 18.35639023, 20.60424729, 21.29767399, 18.608402  ])

# sol = np.loadtxt('test_solution.csv')
new_sol = improve_solution(npp, solutions[0], genetic.best_val)
new_sol_1 = improve_solution(npp, solutions[1], genetic_op.best_val)
tol_new_sol = improve_solution(npp, solutions[0], genetic.best_val, tol=0)
tol_new_sol_1 = improve_solution(npp, solutions[1], genetic_op.best_val, tol=0)

print(genetic.best_val, genetic_op.best_val)
print(npp.compute_solution_value(new_sol), npp.compute_solution_value(new_sol_1))
print(npp.compute_solution_value(tol_new_sol), npp.compute_solution_value(tol_new_sol_1))
print(solver_solution, solver.time, solver.status)

#
# new_start = []
# for i, sol in enumerate(genetic.final_population):
#     val = npp.compute_solution_value(sol)
#     new_sol = improve_solution(npp, sol, val)
#     new_start.append(new_sol)
#
# 4.492595672607422 7.581286907196045 4625.292164058913 4689.776235794727 0.013749925048372025
# 4689.776235794727 4693.104103094211 4630.38159611955
#
#
# genetic.run(ITERATIONS, np.array(new_start))