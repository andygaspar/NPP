import os as os

import numpy as np

from Instance.instance import Instance
from PSO.swarm import Swarm
from Solver.global_ import GlobalSolver
from Solver.lower import LowerSolver
from Solver.lower_aggregated import LowerSolverAggregated
from Solver.pso_solver import PsoSolver


"""

IMPLEMENTARE BOUND PARTICLE


"""
os.system("PSO/install.sh")

n_tolls = 3
n_particles = 200

npp = Instance(n_locations=5, n_tolls=n_tolls, n_commodities=10, seeds=True)

npp.show()

global_solver = GlobalSolver(npp)
global_solver.solve()

print("obj val global", global_solver.m.objVal)

pippo = [npp.npp.edges[p]['weight'] for p in npp.toll_paths]
max_total_val = max([max(list(com.M_p.values())) for com in npp.commodities])
n_iterations = 100

path_costs = np.random.uniform(size=n_tolls*n_particles)
init_norm_costs = np.random.uniform(size=n_tolls*n_particles)


s = PsoSolver(npp, init_norm_costs, path_costs, n_particles, n_tolls, n_iterations)
s.run()
# s.print_swarm()

# s.test_io(10)
