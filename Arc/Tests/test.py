from Arc.ArcInstance.delunay_instance import DelaunayInstance
from Arc.ArcSolver.arc_solver import ArcSolver

n_arcs = 104
dim_grid = (5, 12)
n_locations = dim_grid[0] * dim_grid[1]
# toll_proportion = 10
toll_proportion = [5, 10, 15, 20]
# n_commodities = 10
n_commodities = [40, 50, 60]

instance = DelaunayInstance(n_locations, n_arcs, toll_proportion[3], n_commodities[2])
# instance.show()

global_solve = ArcSolver(instance=instance)
f_o_global, best_bound_global = global_solve.solve(time_limit=1800, verbose=True)  # int(pso.time))