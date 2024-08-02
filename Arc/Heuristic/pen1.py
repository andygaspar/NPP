import os
import random

import numpy as np
import gurobipy as gb
from gurobipy import GRB

from Arc.ArcInstance.grid_instance import GridInstance

random.seed(0)
np.random.seed(0)

# os.system("Arc/Arc_GA/install_arc.sh")

n_arcs = 104
dim_grid = (5, 12)
# 5 *12
# dim_grid = (3, 4)
n_locations = dim_grid[0] * dim_grid[1]
# toll_proportion = 10
toll_proportion = [5, 10, 15, 20]
# n_commodities = 10
n_commodities = [10, 50, 60]

# instance = DelaunayInstance(n_locations, n_arcs, dim_grid, toll_proportion[0], n_commodities[0])
# instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion[2], n_commodities[2])

# instance.show()
row = 0

instance = GridInstance(n_locations, n_arcs, dim_grid, 5, 10, seed=0)

instance.show()
a1, a2 = instance.get_adj()

A1 = np.zeros((instance.n_nodes, len(instance.edges_idx)), dtype=int)
A2 = np.zeros((instance.n_nodes, len(instance.edges_idx)), dtype=int)

for edge in instance.edges_idx.keys():
    if edge in instance.toll_arcs:
        A1[edge[0], instance.edges_idx[edge]] = 1
        A1[edge[1], instance.edges_idx[edge]] = -1
    else:
        A2[edge[0], instance.edges_idx[edge]] = 1
        A2[edge[1], instance.edges_idx[edge]] = -1

b = np.zeros(instance.n_nodes)
for com in instance.commodities:
    b[com.origin] -= com.n_users
    b[com.destination] += com.n_users

c = np.zeros(len(instance.edges_idx))
d = np.zeros(len(instance.edges_idx))

for edge in instance.edges_idx.keys():
    if edge in instance.toll_arcs:
        c[instance.edges_idx[edge]] = instance.npp.edges[edge]['weight']
    else:
        d[instance.edges_idx[edge]] = instance.npp.edges[edge]['weight']

pen1 = gb.Model()

x = pen1.addMVar(len(instance.edges_idx), name='x', vtype=GRB.CONTINUOUS)
y = pen1.addMVar(len(instance.edges_idx), name='y', vtype=GRB.CONTINUOUS)
# T = pen1.addMVar(len(instance.edges_idx), name='T')
T = np.random.uniform(1, 10, len(instance.edges_idx))
lam = pen1.addMVar(instance.n_nodes, name='lam', lb=-np.inf, ub=np.inf)


pen1.setObjective((c + T) @ x, GRB.MAXIMIZE)

pen1.addConstr(A1 @ x + A2 @ y == b)
pen1.addConstr(lam @ A1 <= c + T)
pen1.addConstr(lam @ A2 <= d)
#
pen1.addConstr((c + T) @ x + d @ y - lam @ b == 0)

pen1.optimize()

print(np.round(y.x))


print("hello")