import os
import random

import networkx as nx
import numpy as np
import pandas as pd

from Arc.ArcInstance.delunay_instance import DelaunayInstance
from Arc.ArcInstance.grid_instance import GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
from Arc.Arc_GA.arc_genetic_cpp import ArcGeneticCpp
from Arc.Heuristic.arc_heuristic import run_arc_heuristic
from Arc.Heuristic.arc_heuristic2 import run_arc_heuristic2
from Arc.genetic_arc import GeneticArc

random.seed(0)
np.random.seed(0)

os.system("Arc/Arc_GA/install_arc.sh")

n_arcs = 3*4
# 5 *12
dim_grid = (3, 4)
n_locations = dim_grid[0] * dim_grid[1]

toll_proportion = 5
n_commodities = 10


columns = ['run', 'graphs', 'toll_pr', 'n_com', 'obj', 'time', 'gap', 'status']
row = 0
df = pd.DataFrame(columns=columns)

run = 0
random.seed(run)
np.random.seed(run)
instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion, n_commodities, seed=run)
instance.show()

solver = ArcSolver(instance=instance, symmetric_costs=False)
solver.solve(time_limit=3600, verbose=True)  # int(pso.time))

ITERATIONS = 100

g = GeneticArc(64, instance, offspring_rate=0.2)
g.run_cpp(ITERATIONS, verbose=True, n_threads=16, seed=0)

print(g.time, solver.time)
print(g.best_val, solver.obj)
