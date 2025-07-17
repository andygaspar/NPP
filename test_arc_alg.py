import os
import random

import networkx as nx
import numpy as np
import pandas as pd

from Arc.ArcInstance.grid_instance import GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
from Arc.genetic_arc import GeneticArc

random.seed(0)
np.random.seed(0)

os.system("Arc/Arc_GA/install_arc.sh")

n_arcs = 3*4
# 5 *12100

n = 3
n_arcs = n ** 2
dim_grid = (n, n)
n_locations = dim_grid[0] * dim_grid[1]

toll_proportion = 15
n_commodities = 30


columns = ['run', 'graphs', 'toll_pr', 'n_com', 'obj', 'time', 'gap', 'status']
row = 0
df = pd.DataFrame(columns=columns)

run = 0
random.seed(run)
np.random.seed(run)
instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion, n_commodities, seed=run)
instance.show()

solver = ArcSolver(instance=instance, symmetric_costs=False)
solver.solve(time_limit=5, verbose=True)  # int(pso.time))

ITERATIONS = 10000

g = GeneticArc(1024, instance, offspring_rate=0.2)
g.run_cpp_heuristic(ITERATIONS, dijkstra_every=50, verbose=True, n_threads=16, seed=0)

print(g.time, solver.time)
print(g.best_val, solver.obj)

print(solver.incumbent.times)
print(solver.incumbent.sol_list)
