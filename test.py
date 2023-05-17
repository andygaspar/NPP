import os as os
import time

import numpy as np
from importlib.metadata import version
from Instance.instance import Instance
# from Net.network_manager import NetworkManager
from Solver.global_ import GlobalSolver
from Solver.pso_solver import PsoSolver
from Solver.pso_solver_ import PsoSolverNew

os.system("PSO_/install.sh")

# n_locations = 10
# n_commodities = 2
# n_tolls = 3

# n_locations = 50
# n_commodities = 10
# n_tolls = 8

np.random.seed(0)

n_locations = 10
n_commodities = 8
n_tolls = 15

for _ in range(2):

    npp = Instance(n_locations=n_locations, n_tolls=n_tolls, n_commodities=n_commodities, seeds=False)
    # npp.save_problem()
    # npp.show()

    t = time.time()
    global_solver = GlobalSolver(npp)
    global_solver.solve()
    # global_solver.print_model()
    print('time global ', time.time() - t)

    print("obj val global", global_solver.m.objVal)

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
    pso.run(init_pos=latin_hyper, stats=False, verbose=True)
    print('time pso ', time.time() - t)
    print(pso.best_val)

    print()

    # print(global_solver.get_prices())
    # print(npp.n_toll_paths)
    # stats = pso.get_stats()






