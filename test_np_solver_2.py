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
COMMODITIES = 30
TOLL_PROPORTION = 0.2

TIME_LIMIT = 60

grid = GridInstance(COMMODITIES, TOLL_PROPORTION, N)
grid.draw()

ITERATIONS = 10000
g2 = GeneticArc(128, grid, mutation_rate=0.02)
g2.run_cpp_heuristic(ITERATIONS, dijkstra_every=500, verbose=True, n_threads=16, seed=0)

time_np = time.time()
adj_arc_node_bool = np.zeros((grid.n_nodes, grid.n_edges))

for i, e in enumerate(grid.edges):
    adj_arc_node_bool[e[0], i] = -1
    adj_arc_node_bool[e[1], i] = 1

b = np.zeros((grid.n_commodities, grid.n_nodes))
for i, k in enumerate(grid.commodities):
    b[i,  k.origin] = -1
    b[i, k.destination] = 1


m = Model('')
m.Params.timelimit = TIME_LIMIT
x = m.addMVar((grid.n_commodities, grid.n_edges), vtype=GRB.BINARY)

c_at = np.array([grid.adj[e[0], e[1]] for e in grid.edges if e in grid.arc_tolls])
c_af = np.array([grid.adj[e[0], e[1]] for e in grid.edges if e in grid.arc_free])
la = m.addMVar((grid.n_commodities, grid.n_nodes), lb=-1e5)
T = m.addMVar(grid.n_tolls)
t = m.addMVar((grid.n_commodities, grid.n_tolls))
A_1_bool = np.zeros((grid.n_nodes, grid.n_tolls))
A_2_bool = np.zeros((grid.n_nodes, grid.n_free))

i, j = 0, 0
for e in grid.edges:
    if e in grid.arc_tolls:
        A_1_bool[e[0], i] = 1
        A_1_bool[e[1], i] = -1
        i += 1
    if e in grid.arc_free:
        A_2_bool[e[0], j] = 1
        A_2_bool[e[1], j] = -1
        j += 1

x_idxs = np.array([i for i in range(grid.n_edges) if grid.edges[i] in grid.arc_tolls])
y_idxs = np.array([i for i in range(grid.n_edges) if grid.edges[i] in grid.arc_free])

N = np.array([toll.N_p for toll in grid.tolls])


for k, comm in enumerate(grid.commodities):
    m.addConstr(adj_arc_node_bool @ x[k] == b[k])
    m.addConstr(A_1_bool.T @ la[k] <= c_at + T)
    m.addConstr(A_2_bool.T @ la[k] <= c_af)

    m.addConstr((c_at * x[k, x_idxs]).sum() + t[k].sum() + (c_af * x[k, y_idxs]).sum() ==
                    la[k, comm.origin] - la[k, comm.destination])
#
    M = np.array([comm.M_p[e] for e in grid.edges if e in grid.arc_tolls])
    m.addConstr(t[k] <= M * x[k, x_idxs])
    m.addConstr(T - t[k] <= N * (1 - x[k, x_idxs]))
    m.addConstr(t[k] <= T)

# m.addConstr(T[1:4] == 100000)




# max_profit = sum([k.n_users * sum(k.M_p.values()) for k in grid.commodities])
n_k = np.array([[k.n_users for e in grid.arc_tolls] for k in grid.commodities])

m.setObjective((n_k * t).sum(), sense=GRB.MAXIMIZE)
print('**********', seed)
m.optimize()


time_np = time.time() - time_np
print(m.objval)
problem = ArcSolver(grid)
problem.solve(verbose=True, time_limit=TIME_LIMIT)

print(m.objVal, problem.obj)
print(time_np, problem.time)



pass

# print(g2.best_val/problem.obj)
# grid.compute_obj(g2.adj_solution, g2.prices)
# grid.compute_obj(problem.adj_solution, problem.prices)
