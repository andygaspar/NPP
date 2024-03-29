import os as os
import random
import time

import numpy as np
import pandas as pd

from Instance.instance import Instance
# from Net.network_manager import NetworkManager
from Solver.solver import GlobalSolver
from Solver.pso_solver import PsoSolverNew

os.system("../PSO/install.sh")

columns = ['run', 'commodities', 'paths', 'obj_exact', 'obj_pso', 'gap', 'time_exact', 'time_pso', 'status', 'mip_gap']

df = pd.DataFrame(columns=columns)

print("mandi")

n_iterations = 1000
n_particles = 20_000
no_update_lim = 1000

TIME_LIMIT = 100
VERBOSE = False
row = 0

for n_commodities in [20]:
    for n_paths in [90]:
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

            solver = GlobalSolver(npp, verbose=True, time_limit=TIME_LIMIT)
            solver.solve()
            print("target val", solver.obj)

            # solver.print_model()

            pso = PsoSolverNew(npp, n_particles, n_iterations, no_update_lim)
            initial_position = pso.random_init()
            pso.run(init_pos=initial_position, stats=False, verbose=True, seed=run)
            print('time global ', solver.time, 'time pso ', pso.time)
            gap = 1 - pso.best_val / solver.obj
            print("obj val global", solver.obj, "  obj pso", pso.best_val, '    gap', 1 - pso.best_val / solver.obj,
                  ' iter', pso.final_iterations)

            initial_position = pso.compute_latin_hypercube_init(dimensions=10)
            pso.run(init_pos=initial_position, stats=False, verbose=True, seed=run)
            print('pso latin', pso.best_val)

            print("computed val ex", npp.compute_solution_value(solver.solution_array),
                  "computed val pso", npp.compute_solution_value(pso.best))

            # df = df.({'run': run, 'commodities': n_commodities, 'paths': n_paths, 'obj_exact': solver.obj, 'obj_pso': pso.best_val,
                            # 'gap': gap, 'time_exact': time_solver, 'time_pso': pso_time}, ignore_index=True)
            df.loc[row] = [run, n_commodities, n_paths, solver.obj, pso.best_val,  gap, solver.time, pso.time, solver.m.status, solver.m.MIPGap]

            row += 1

    # print(solver.solution_array)
    # print(pso.best)
    # print(solver.get_prices())
    # print(npp.n_toll_paths)
    # stats = pso.get_stats()
# df.to_csv('test.csv', index=False)
'''
iter 0  best_val: 355.979    avg vel: 0
iter 100  best_val: 1125.06    avg vel: 0
iter 200  best_val: 1142.53    avg vel: 0
iter 300  best_val: 1175.73    avg vel: 0
iter 400  best_val: 1185.21    avg vel: 0
iter 500  best_val: 1191.62    avg vel: 0
'''
