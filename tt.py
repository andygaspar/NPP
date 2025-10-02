import copy
import time

import pandas as pd

from Arc.ArcInstance.arc_instance import DelaunayInstance, GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
import random
import numpy as np
from gurobipy import Model, GRB

from Arc.genetic_arc import GeneticArc

seed = 9

random.seed(seed)
np.random.seed(seed)


N = 144
COMMODITIES = 11
TOLL_PROPORTION = 0.2

TIME_LIMIT = 10


# grid.draw()

adj = np.array([[]])


time_np = time.time()
adj_arc_node_bool = np.zeros((4, 4))

edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
arc_tolls = [(0, 1), (0, 2)]
arc_free = [(1, 3), (1, 3)]

for i, e in enumerate(edges):
    adj_arc_node_bool[e[0], i] = -1
    adj_arc_node_bool[e[1], i] = 1

b = np.zeros((1, 4))

b[0,  0] = -1
b[0, 3] = 1


m = Model('')
m.Params.timelimit = TIME_LIMIT
x = m.addMVar((1, 4), vtype=GRB.BINARY)

c_at = np.array([1 for e in edges if e in arc_tolls])
c_af = np.array([2 for e in edges if e in arc_free])
la = m.addMVar((1, 4), lb=-1000000)
T = m.addMVar(2)
t = m.addMVar((1, 2))
A_1_bool = np.zeros((4, 2))
A_2_bool = np.zeros((4, 2))

i, j = 0, 0
for e in edges:
    if e in arc_tolls:
        A_1_bool[e[0], i] = 1
        A_1_bool[e[1], i] = -1
        i += 1
    if e in arc_free:
        A_2_bool[e[0], j] = 1
        A_2_bool[e[1], j] = -1
        j += 1

x_idxs = np.array([0, 1])
y_idxs = np.array([2, 3])

N = 1000000



m.addConstr(adj_arc_node_bool @ x[0] == b[0])
m.addConstr(A_1_bool.T @ la[0] <= c_at + T)
m.addConstr(A_2_bool.T @ la[0] <= c_af)

m.addConstr((c_at * x[0, x_idxs]).sum() + t[0].sum() + (c_af * x[0, y_idxs]).sum() ==
                la[0, 0] - la[0, 3])

M = 10000000
m.addConstr(t[0] <= M * x[0, x_idxs])
m.addConstr(T - t[0] <= N * (1 - x[0, x_idxs]))
m.addConstr(t[0] <= T)


m.setObjective( t.sum(), sense=GRB.MAXIMIZE)
print('**********', seed)
m.optimize()

