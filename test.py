import os as os
import time

import numpy as np

from Instance.instance import Instance
from PSO.swarm import Swarm
from Solver.global_ import GlobalSolver
from Solver.lower import LowerSolver
from Solver.lower_aggregated import LowerSolverAggregated
from Solver.pso_solver import PsoSolver
from Solver.pso_solver_parallel import PsoSolverParallel

"""

IMPLEMENTARE BOUND PARTICLE


"""
os.system("PSO/install.sh")

# n_locations = 10
# n_commodities = 10
# n_tolls = 3

n_locations = 10
n_commodities = 15
n_tolls = 8



npp = Instance(n_locations=n_locations, n_tolls=n_tolls, n_commodities=n_commodities, seeds=True)

npp.show()

t = time.time()
global_solver = GlobalSolver(npp)
global_solver.solve()

print('time global ', time.time() - t)

print("obj val global", global_solver.m.objVal)

n_iterations = 100
n_particles = 100

path_costs = np.random.uniform(size=npp.n_paths*n_particles)
init_norm_costs = np.random.uniform(size=npp.n_paths*n_particles)


s = PsoSolverParallel(npp, init_norm_costs, path_costs, n_particles, n_tolls, n_iterations)
t = time.time()
s.run()
t = time.time() - t
print('time ', t)
# s.print_swarm()

# s.test_io(10)
