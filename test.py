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

n_iterations = 2_000
n_particles = 20_000
no_update_lim = 1000

VERBOSE = False
row = 0

for n_commodities in [20, 56, 90]:
    for n_paths in [20, 56, 90]:
# for n_commodities in [56]:
#     for n_paths in [20, 56, 90]:
        for run in range(10):
            print(n_commodities, n_paths, run)
            random.seed(run)
            np.random.seed(run)

            npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=False)
            # npp.show_original()
            # npp.show()
            # npp.save_problem()
            # npp.show()

            t = time.time()
            global_solver = GlobalSolver(npp, verbose=VERBOSE)
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
            k = pso.random_init()
            # latin_hyper = pso.compute_latin_hypercube_init(dimensions=5)
            pso.run(init_pos=k, stats=False, verbose=True, seed=run)
            pso_time = time.time() - t
            print('time global ', time_solver, 'time pso ', pso_time)
            gap = 1 - pso.best_val / global_solver.obj
            print("obj val global", global_solver.obj, "  obj pso", pso.best_val, '    gap', 1 - pso.best_val / global_solver.obj,
                  ' iter', pso.final_iterations)

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
time global  0.10101079940795898 time pso  2.761152982711792
obj val global 1274.6312511954852   obj pso 1273.70652438468     gap 0.0007254857512224433
computed val 1273.70652438468
seed set to 1

time global  0.0958106517791748 time pso  2.3473119735717773
obj val global 805.3006068177644   obj pso 803.3259568497367     gap 0.0024520656650574013
computed val 803.3259568497367
'''
