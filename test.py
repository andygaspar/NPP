import os as os
import time

import numpy as np
from importlib.metadata import version
from Instance.instance import Instance
# from Net.network_manager import NetworkManager
from Solver.global_ import GlobalSolver
from Solver.pso_solver import PsoSolver

os.system("PSO_/install.sh")

n_locations = 10
n_commodities = 2
n_tolls = 3

# n_locations = 50
# n_commodities = 10
# n_tolls = 8




n_locations = 10
n_commodities = 8
n_tolls = 15

for _ in range(1):

    npp = Instance(n_locations=n_locations, n_tolls=n_tolls, n_commodities=n_commodities, seeds=True)
    # npp.save_problem()
    # npp.show()

    t = time.time()
    global_solver = GlobalSolver(npp)
    global_solver.solve()
    global_solver.print_model()
    print('time global ', time.time() - t)

    # print("obj val global", global_solver.m.objVal)

    n_iterations = 1_0
    n_particles = 1

    N_PARTS = n_particles // 5
    n_cut = 3
    N_DIV = 4
    n_u_l = 1000
    #
    # path_costs = np.random.uniform(size=npp.n_paths*n_particles)
    # init_norm_costs = np.random.uniform(size=npp.n_paths*n_particles)
    #
    #
    t = time.time()
    s = PsoSolver(npp, None, n_particles, n_iterations, N_PARTS, n_cut, N_DIV, n_u_l, normalised=False, verbose=False)#, time_limit=1, init_sol_num=5)
    s.run()
    # k = s.random_init()
    #
    # latin_hyper = s.compute_latin_hypercube_init(dimensions=5)

    print('time pso ', time.time() - t)
    print(s.best_val)





