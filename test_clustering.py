

import os as os
import random
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from Instance.instance import Instance
# from Net.network_manager import NetworkManager
from Solver.solver import GlobalSolver
from Solver.pso_solver import PsoSolverNew

os.system("PSO/install.sh")

columns = ['run', 'commodities', 'paths', 'obj_exact', 'obj_pso', 'gap', 'time_exact', 'time_pso', 'status', 'mip_gap']

df = pd.DataFrame(columns=columns)

print("mandi")

n_iterations = 1000
n_particles = 20_000
no_update_lim = 1000

TIME_LIMIT = 1
VERBOSE = False
row = 0

n_commodities = 20
n_paths = 90
run = 0

print(n_commodities, n_paths, run)
random.seed(run)
np.random.seed(run)

npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=run)
solver = GlobalSolver(npp, verbose=True, time_limit=TIME_LIMIT)
solver.solve()
print("target val", solver.obj)

# solver.print_model()

pso = PsoSolverNew(npp, 1, 0, no_update_lim)
initial_position = pso.random_init()
npp.compute_solution_value(initial_position[0])
pso.run(initial_position)

t = time.time()
print('cluster time', time.time() - t)
pso.run(init_pos=initial_position, stats=False, verbose=True, seed=run)
print('time global ', solver.time, 'time pso ', pso.time)
gap = 1 - pso.best_val / solver.obj
print("obj val global", solver.obj, "  obj pso", pso.best_val, '    gap', 1 - pso.best_val / solver.obj,
      ' iter', pso.final_iterations)




initial_position = pso.random_init()
n_clusters = 12
t = time.time()
k_means = KMeans(n_clusters=n_clusters, max_iter=3).fit(initial_position)
init_clusters = []
for i in range(n_clusters):
    idxs = np.where(k_means.labels_ == i)
    init_clusters.append(initial_position[idxs])

b_val = 0
new_population = []
for i in range(n_clusters):
    print(i)
    pso = PsoSolverNew(npp, init_clusters[i].shape[0], 100, no_update_lim)
    pso.run(init_pos=init_clusters[i], stats=False, verbose=False, seed=run)
    new_population.append(pso.best)
    if pso.best_val > b_val:
        b_val = pso.best_val
        print(pso.best_val)

new_population = np.array(new_population)
pso = PsoSolverNew(npp, new_population.shape[0], 0, no_update_lim)
pso.run(init_pos=new_population, stats=False, verbose=True, seed=run)

for sol in new_population:
    print(npp.compute_solution_value(sol))



print("computed val ex", npp.compute_solution_value(solver.solution_array),
      "computed val pso", npp.compute_solution_value(pso.best))


row += 1

