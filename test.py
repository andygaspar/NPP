import os as os
import time

import numpy as np

from Instance.instance import Instance
from Solver.global_ import GlobalSolver
from Solver.pso_solver import PsoSolver

"""

IMPLEMENTARE BOUND PARTICLE


"""
os.system("PSO/install.sh")

# n_locations = 10
# n_commodities = 2
# n_tolls = 3

# n_locations = 50
# n_commodities = 10
# n_tolls = 8


n_locations = 15
n_commodities = 8
n_tolls = 4

npp = Instance(n_locations=n_locations, n_tolls=n_tolls, n_commodities=n_commodities, seeds=True)
npp.save_problem()
npp.show()

t = time.time()
global_solver = GlobalSolver(npp)
global_solver.solve()
# global_solver.print_model()
print('time global ', time.time() - t)

print("obj val global", global_solver.m.objVal)

n_iterations = 1_000_000
n_particles = 96
#
# path_costs = np.random.uniform(size=npp.n_paths*n_particles)
# init_norm_costs = np.random.uniform(size=npp.n_paths*n_particles)
#
#
t = time.time()
s = PsoSolver(npp, None, n_particles, n_iterations)#, time_limit=1, init_sol_num=5)
s.run()
print('time pso ', time.time() - t)
print(s.best_val)





