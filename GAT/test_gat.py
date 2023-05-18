import numpy as np
import torch_geometric
from Instance.instance import Instance
from Instance.instance2 import Instance2
from Solver.global_new import GlobalSolverNew
from Solver.pso_solver_ import PsoSolverNew

np.random.seed(0)
n_locations = 10
n_commodities = 8
n_tolls = 15

npp = Instance2(n_tolls=n_tolls, n_commodities=n_commodities, seeds=False)

global_solver = GlobalSolverNew(npp)
global_solver.solve()
print("obj val global", global_solver.m.objVal)


n_iterations = 10000
n_particles = 128
no_update_lim = 1000
#
# path_costs = np.random.uniform(size=npp.n_paths*n_particles)
# init_norm_costs = np.random.uniform(size=npp.n_paths*n_particles)
#
#

pso = PsoSolverNew(npp, n_particles, n_iterations, no_update_lim)
k = pso.random_init()

latin_hyper = pso.compute_latin_hypercube_init(dimensions=5)
pso.run(init_pos=latin_hyper, stats=False, verbose=True)

npp.show()

data_set = npp.make_torch_graph(solution=global_solver.get_prices())

print(data_set)


