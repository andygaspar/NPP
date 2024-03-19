import os as os
import time

import numpy as np
from Instance.instance import Instance
# from Net.network_manager import NetworkManager
from Solver.global_ import GlobalSolver
from Solver.pso_solver_ import PsoSolverNew

os.system("PSO_/install.sh")

np.random.seed(0)
VERBOSE = False
n_commodities = 20
n_paths = 20

for _ in range(2):

    npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seeds=False)
    # npp.show_original()
    # npp.show()
    # npp.save_problem()
    # npp.show()

    t = time.time()
    global_solver = GlobalSolver(npp, verbose=True)
    global_solver.solve()
    print('time global ', time.time() - t)
    print("obj val global", global_solver.obj)

    n_iterations = 10000
    n_particles = 128
    no_update_lim = 1000
    #
    # path_costs = np.random.uniform(size=npp.n_paths*n_particles)
    # init_norm_costs = np.random.uniform(size=npp.n_paths*n_particles)
    #
    #
    t = time.time()
    pso = PsoSolverNew(npp, n_particles, n_iterations, no_update_lim)
    k = pso.random_init()

    latin_hyper = pso.compute_latin_hypercube_init(dimensions=5)
    pso.run(init_pos=k, stats=False, verbose=True)
    print('time pso ', time.time() - t)
    print(pso.best_val)

    print('gap', 1 - pso.best_val/global_solver.obj)

    # print(global_solver.get_prices())
    # print(npp.n_toll_paths)
    # stats = pso.get_stats()
    print()






