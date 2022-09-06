from Instance.instance import Instance
from Solver.global_ import GlobalSolver

npp = Instance(n_locations=5, n_tolls=3, n_commodities=10, seeds=True)

npp.show()

global_solver = GlobalSolver(npp)
global_solver.solve()
