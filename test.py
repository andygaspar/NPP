from Instance.instance import Instance
from Solver.global_ import GlobalSolver
from Solver.lower import LowerSolver
from Solver.lower_aggregated import LowerSolverAggregated

npp = Instance(n_locations=5, n_tolls=3, n_commodities=10, seeds=True)

npp.show()

global_solver = GlobalSolver(npp)
global_solver.solve()
lower_solver = LowerSolverAggregated(npp, 1)
lower_solver.set_up()
path_costs = [npp.edges[p]['weight'] for p in npp.p]
lower_solver.solve(path_costs)
