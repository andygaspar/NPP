import os
import random

import networkx as nx
import numpy as np

from Arc.ArcInstance.delunay_instance import DelaunayInstance
from Arc.ArcInstance.grid_instance import GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
from Arc.Arc_GA.arc_genetic_cpp import ArcGeneticCpp
from Arc.genetic_arc import GeneticArc

random.seed(0)
np.random.seed(0)

os.system("Arc/Arc_GA/install_arc.sh")

n_arcs = 104
dim_grid = (5, 12)
# 5 *12
# dim_grid = (3, 4)
n_locations = dim_grid[0] * dim_grid[1]
# toll_proportion = 10
toll_proportion = [5, 10, 15, 20]
# n_commodities = 10
n_commodities = [40, 50, 60]

# instance = DelaunayInstance(n_locations, n_arcs, toll_proportion[0], n_commodities[0])
instance = GridInstance(n_locations, dim_grid, toll_proportion[2], n_commodities[2])

instance.show()


solver = ArcSolver(instance=instance, symmetric_costs=False)
solver.solve(time_limit=60, verbose=True)  # int(pso.time))

print(solver.time)

# adj, prices = solver.get_adj_solution()
# instance.dijkstra(adj, prices, instance.commodities[0])
# print(instance.compute_obj(adj, prices))

# print(solver.get_tolls())

g = GeneticArc(64, instance)
# g.run(3, verbose=True)

g.run_cpp(1000, verbose=True, n_threads=16, seed=0)

pop, vals = g.genetic_cpp.get_results()
print(solver.time, g.time, solver.obj, g.best_val)

print(solver.solution)
print(g.solution)

print(instance.compute_obj(solver.adj_solution, solver.mat_solution))
print(instance.compute_obj(g.adj_solution, g.mat_solution))


