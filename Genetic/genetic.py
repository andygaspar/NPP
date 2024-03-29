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

        self.genetic = GeneticCpp(npp.upper_bounds, npp.commodities_tax_free,
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


n_paths = 56
n_commodities = 56

POPULATION = 128
off_size = int(POPULATION / 2)
ITERATIONS = 2000
recombination_size = int(n_paths / 2)
MUTATION_RATE = 0.02

PSO_EVERY = 10000
PSO_SIZE = 4
PSO_SELECTION = 4
PSO_ITERATIONS = 20000
PSO_FINAL_ITERATIONS = 1
NO_UPDATE_LIM = 300
os.system("PSO/install.sh")

N_THREADS = 1
seed = 0
VERBOSE = True

np.set_printoptions(edgeitems=30, linewidth=100000)

TIME_LIMIT = 5

np.random.seed(seed)
random.seed(seed)

solutions = np.zeros((2, n_paths))

npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=seed)

npp.save_problem()
solver = GlobalSolver(npp, verbose=True, time_limit=TIME_LIMIT)
solver.solve()

genetic = GeneticPso(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, PSO_SIZE,
                     PSO_SELECTION, PSO_EVERY, PSO_ITERATIONS, PSO_FINAL_ITERATIONS,
                     NO_UPDATE_LIM, verbose=VERBOSE, n_threads=N_THREADS, seed=-1)
init_population = np.random.uniform(size=(POPULATION, npp.n_paths)) * npp.upper_bounds

genetic.run(ITERATIONS * 16, init_population)
solutions[0] = genetic.final_population[0]
#
# t = time.time()
# island_size = 16
# start = np.zeros((POPULATION, npp.n_paths))
# vals = np.zeros(POPULATION)
# for i in range(island_size):
#     genetic_op = GeneticPso(npp, POPULATION, off_size, MUTATION_RATE, recombination_size, PSO_SIZE,
#                             PSO_SELECTION, 15000, PSO_ITERATIONS, 0,
#                             NO_UPDATE_LIM, verbose=False, n_threads=N_THREADS, seed=-1)
#
#     genetic_op.run(ITERATIONS)
#
#     start[i] = genetic_op.final_population[0]
#     vals[i] = genetic_op.best_val
#
# start[island_size:] = np.random.uniform(size=(POPULATION - island_size, npp.n_paths)) * npp.upper_bounds
#
# genetic_op = GeneticPso(npp, start.shape[0], start.shape[0] // 2, MUTATION_RATE, recombination_size, PSO_SIZE,
#                         PSO_SELECTION, 15000, PSO_ITERATIONS, 0,
#                         NO_UPDATE_LIM, verbose=True, n_threads=N_THREADS, seed=-1)
# print('final')
# genetic_op.run(ITERATIONS, start)
#
# solutions[1] = genetic_op.final_population[0]

# t = time.time() - t
# print(max(vals))
# print(genetic.time, t, genetic.best_val, genetic_op.best_val, 1 - genetic.best_val / genetic_op.best_val)



# print(genetic.time, genetic_op.time, solver.time, genetic.best_val, genetic_op.best_val, 1 - genetic.best_val / genetic_op.best_val)



sol = solutions[0]
npp.compute_solution_value(sol)
new_sol = improve_solution(npp, sol, 0.01)
new_sol = improve_solution(npp, new_sol, 0.01)
print(npp.compute_solution_value(new_sol), npp.compute_solution_value(sol))

sol = solutions[0]
npp.compute_solution_value(sol)

res = []
res2 = []

n_usr = []

new_sol = sol.copy()
new_sol[18] += 0.0001 # 0.0347932262337396

w = None
# npp.compute_solution_value(sol)
total_profit = 0
for commodity in npp.commodities:
    costs = new_sol + commodity.c_p_vector
    idxs = np.argsort(np.append(costs, commodity.c_od))
    s = np.append(new_sol, [0])
    # if idxs[0] == 18:
    #     print(commodity, np.append(costs, commodity.c_od))
    cost = 0 if min(costs) > commodity.c_od + 1e-5 else new_sol[np.argmin(costs)]
    # if commodity.name == '$U_{49}$':
    #     print(cost, np.argmin(costs), idxs[0], commodity.c_od, min(costs), np.append(costs, commodity.c_od))
    #     print(idxs)
    c = np.sort(np.append(costs, commodity.c_od))[:2]
    # print(c, c[0]==c[1] )
    print(c[c==c[0]])
    res2.append([idxs[0], cost])
    total_profit += cost * commodity.n_users
    n_usr.append(commodity.n_users)
print(total_profit)
print(list(sol))


w[0] == w[1]


r = np.array(res)
r2 = np.array(res2)
jj = np.concatenate([r, r2], axis=1)
total_profit = 1
for j in range(56):
    if jj[j, 0] != jj[j, 1]:
        print('k')
print('mandi')
