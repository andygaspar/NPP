import os

import numpy as np

from Instance.instance import Instance
from PSO.swarm import Swarm
from Solver.global_ import GlobalSolver
from Solver.lower import LowerSolver
from Solver.lower_aggregated import LowerSolverAggregated


os.system("PSO/install.sh")

n_tolls = 3
n_particles = 5

npp = Instance(n_locations=5, n_tolls=n_tolls, n_commodities=10, seeds=True)

npp.show()

# global_solver = GlobalSolver(npp)
# global_solver.solve()

max_total_val = max([com.c_od for com in npp.commodities] + [com.c_p[val] for com in npp.commodities for val in com.c_p.keys()])

path_costs = np.array(n_particles*[npp.npp.edges[p]['weight'] for p in npp.p])/max_total_val
lower_solver = LowerSolverAggregated(npp, n_particles)
lower_solver.set_up()

s = Swarm(path_costs, n_particles, n_tolls)
for iteration in range(100):
    personal_run_results = lower_solver.solve(path_costs)
    run_best = max(personal_run_results)
    if run_best > s.get_best():
        print(s.get_best())
        s.update_best(np.argmax(personal_run_results), run_best)
        print(s.get_best())

    s.update_swarm(iteration, personal_run_results)
# s.print_swarm()

# s.test_io(10)
