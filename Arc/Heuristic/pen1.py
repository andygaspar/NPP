import random

import numpy as np

from Arc.ArcInstance.grid_instance import GridInstance
import gurobipy as gb
from gurobipy import GRB

dim_grid = (3, 4)
n_locations = dim_grid[0] * dim_grid[1]

toll_proportion = 5
n_commodities = 10
n_arcs = 104

run = 0
random.seed(run)
np.random.seed(run)
instance = GridInstance(n_locations, dim_grid, toll_proportion, n_commodities, seed=run)
# instance.show()

a1, a2 = instance.get_mats()

a1

a2

m = gb.Model('CVRP')
m.modelSense = GRB.MAXIMIZE
status_dict = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED',
                    6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT'}
status = None

eps = 1e-2

x = m.addVars([(a.idx, k) for a in instance.tolls for k in instance.commodities],
                        vtype=GRB.BINARY)



x = 1105.99

y = 1105.99 *30 /100
x + y