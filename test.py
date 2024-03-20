import os as os
import random
import time

import numpy as np
import pandas as pd

from Instance.instance import Instance
# from Net.network_manager import NetworkManager
from Solver.solver import GlobalSolver
from Solver.pso_solver import PsoSolverNew

os.system("PSO/install.sh")

columns = ['run', 'commodities', 'paths', 'obj_exact', 'obj_pso', 'gap', 'time_exact', 'time_pso', 'status', 'mip_gap']

df = pd.DataFrame(columns=columns)

print("mandi")

n_iterations = 500
n_particles = 20_000
no_update_lim = 1000

VERBOSE = False
row = 0

for n_commodities in [20]:
    for n_paths in [20]:
# for n_commodities in [56]:
#     for n_paths in [20, 56, 90]:
        for run in range(10):
            print(n_commodities, n_paths, run)
            random.seed(run)
            np.random.seed(run)

            npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=run)
            # npp.show_original()
            # npp.show()
            # npp.save_problem()
            # npp.show()

            t = time.time()
            global_solver = GlobalSolver(npp, verbose=True, time_limit=10)
            global_solver.solve()
            print("target val", global_solver.obj)
            time_solver = time.time() - t

            # global_solver.print_model()

            #
            # path_costs = np.random.uniform(size=npp.n_paths*n_particles)
            # init_norm_costs = np.random.uniform(size=npp.n_paths*n_particles)
            #
            #
            t = time.time()
            pso = PsoSolverNew(npp, n_particles, n_iterations, no_update_lim)
            initial_position = pso.random_init()
            # initial_position = pso.compute_latin_hypercube_init(dimensions=5)
            pso.run(init_pos=initial_position, stats=False, verbose=True, seed=run)
            pso_time = time.time() - t
            print('time global ', time_solver, 'time pso ', pso_time)
            gap = 1 - pso.best_val / global_solver.obj
            print("obj val global", global_solver.obj, "  obj pso", pso.best_val, '    gap', 1 - pso.best_val / global_solver.obj,
                  ' iter', pso.final_iterations)

            initial_position = pso.compute_latin_hypercube_init(dimensions=5)
            pso.run(init_pos=initial_position, stats=False, verbose=True, seed=run)
            print('pso latin', pso.best_val)

            print("computed val ex", npp.compute_solution_value(global_solver.solution_array),
                  "computed val pso", npp.compute_solution_value(pso.best))

            # df = df.({'run': run, 'commodities': n_commodities, 'paths': n_paths, 'obj_exact': global_solver.obj, 'obj_pso': pso.best_val,
                            # 'gap': gap, 'time_exact': time_solver, 'time_pso': pso_time}, ignore_index=True)
            df.loc[row] = [run, n_commodities, n_paths, global_solver.obj, pso.best_val,  gap, time_solver, pso_time, global_solver.m.status, global_solver.m.MIPGap]

            row += 1

    # print(global_solver.solution_array)
    # print(pso.best)
    # print(global_solver.get_prices())
    # print(npp.n_toll_paths)
    # stats = pso.get_stats()
df.to_csv('test.csv', index=False)
'''
iter 0  best_val: 355.979    avg vel: 0
iter 100  best_val: 1125.06    avg vel: 0
iter 200  best_val: 1142.53    avg vel: 0
iter 300  best_val: 1175.73    avg vel: 0
iter 400  best_val: 1185.21    avg vel: 0
iter 500  best_val: 1191.62    avg vel: 0
'''
