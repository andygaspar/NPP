import os
import random

import networkx as nx
import numpy as np
import gurobipy as gb
from gurobipy import GRB

from Arc.ArcInstance.grid_instance import GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
from Arc.genetic_arc import GeneticArc

random.seed(0)
np.random.seed(0)

# os.system("Arc/Arc_GA/install_arc.sh")
200/20
n_arcs = 104
# dim_grid = (5, 12)
# 5 *12
dim_grid = (20, 10)
n_locations = dim_grid[0] * dim_grid[1]
# toll_proportion = 10
toll_proportion = [5, 10, 15, 20]
# n_commodities = 10
n_commodities = [10, 50, 60]

# instance = DelaunayInstance(n_locations, n_arcs, dim_grid, toll_proportion[0], n_commodities[0])
# instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion[2], n_commodities[2])

# instance.show()
row = 0



instance = GridInstance(n_locations, n_arcs, dim_grid, 20, 60, seed=0)
print(instance.n_edges)

solver = ArcSolver(instance=instance, symmetric_costs=False)
solver.solve(time_limit=30, verbose=True)  # int(pso.time))

# instance.show()

POPULATION_SIZE = 128
ITERATIONS = 1000
g = GeneticArc(population_size=POPULATION_SIZE, npp=instance, offspring_rate=0.5, mutation_rate=0.02)
g.run_cpp(ITERATIONS, verbose=True, n_threads=16, seed=0)

used = []
for com in g.npp.commodities:
    for path in com.solution_edges:
        if path not in used and path in g.npp.toll_arcs:
            used.append(path)



a1, a2 = instance.get_bool_mats()

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

print([i.idx for i in instance.tolls])
for edge in instance.edges_idx.keys():
    if edge in instance.toll_arcs:
        print(instance.edges_idx[edge])
        c[instance.edges_idx[edge]] = instance.npp.edges[edge]['weight']
    else:
        d[instance.edges_idx[edge]] = instance.npp.edges[edge]['weight']





np.round(1.20385479e+02)

M = 1

# quad = gb.Model()
# x = quad.addMVar(len(instance.edges_idx), name='x', lb=0, ub=np.inf)
# y = quad.addMVar(len(instance.edges_idx), name='y', lb=0, ub=np.inf)
#
# lam = quad.addMVar(instance.n_nodes, name='lam', lb=-np.inf, ub=np.inf)
# quad.setObjective((lam @ A1 - c) @ x - M * (lam @ A1 @ x + d @ y - lam @ b), GRB.MAXIMIZE)
# quad.addConstr( A1 @ x + A2 @ y == b)
# quad.addConstr(lam @ A2 <= d)
# # quad.addConstr(lam @ A1 - c >= 0)
# quad.optimize()



for com in instance.commodities:
    print(com.origin, com.destination)

d
np.nonzero(A1)
toll_ = []
for edge in instance.edges_idx.keys():
    if edge in instance.toll_arcs:
        toll_.append(instance.edges_idx[edge])
toll_

np.nonzero(A2)


quad1 = gb.Model()
quad1.setParam('NonConvex', 2)
x = quad1.addMVar(len(instance.edges_idx), name='x', lb=0, ub=np.inf)
y = quad1.addMVar(len(instance.edges_idx), name='y', lb=0, ub=np.inf)
T = quad1.addMVar(len(instance.edges_idx), name='y', lb=0, ub=np.inf)
lam = quad1.addMVar(instance.n_nodes, name='lam', lb=-np.inf, ub=np.inf)
quad1.setObjective(T @ x - 100 * ((c + T) @ x + d @ y - lam @ b), GRB.MAXIMIZE)
quad1.addConstr(A1 @ x + A2 @ y == b)
quad1.addConstr(lam @ A2 <= d)
quad1.addConstr(lam @ A1 <= c + T)

quad1.optimize()
T.x
solver.solution
lam.x
(T.x * x.x).sum()
x.x
b
POPULATION_SIZE = 64
ITERATIONS = 10000
g = GeneticArc(population_size=POPULATION_SIZE, npp=instance, offspring_rate=0.5, mutation_rate=0.1)
g.run_cpp(ITERATIONS, verbose=True, n_threads=16, seed=0)

g.best_val
solver.obj
x = np.zeros(len(instance.edges_idx))
y = np.zeros(len(instance.edges_idx))
for com in g.npp.commodities:
    for i in range(len(com.solution_path) - 1):
        if (com.solution_path[i], com.solution_path[i + 1]) in instance.toll_arcs:
            print(instance.edges_idx[(com.solution_path[i], com.solution_path[i + 1])])
            x[instance.edges_idx[(com.solution_path[i], com.solution_path[i + 1])]] += com.n_users
        else:
            y[instance.edges_idx[(com.solution_path[i], com.solution_path[i + 1])]] += com.n_users


quad1 = gb.Model()
quad1.setParam('NonConvex', 2)
T = quad1.addMVar(len(instance.edges_idx), name='y', lb=0, ub=np.inf)
lam = quad1.addMVar(instance.n_nodes, name='lam', lb=-np.inf, ub=np.inf)
quad1.setObjective(T @ x - 1 * ((c + T) @ x + np.dot(d,  y) - lam @ b), GRB.MAXIMIZE)
quad1.addConstr(lam @ A2 <= d)
quad1.addConstr(lam @ A1 <= c + T)
quad1.optimize()
T.x

g.solution
solver.solution

pen1 = gb.Model()
lam_max = 1000
lam = pen1.addMVar(instance.n_nodes, name='lam', lb=0, ub=lam_max)
M = 100

pen1.setObjective((lam @ A1 - c) @ x - M * ( lam @ A1 @ x + np.dot(d, y) - lam @ b), GRB.MAXIMIZE)

pen1.addConstr(lam @ A2 <= d)
# pen1.addConstr(lam @ A1 - c >= 0)

pen1.optimize()

T = np.dot(lam.x, A1) - c
print(T)
print(g.solution)
a, p = g.get_mats(g.solution)

c

g.best_val
solver.obj

prices = np.zeros_like(instance.adj)
for toll in instance.edges_idx.keys():
    prices[toll[0], toll[1]] = T[instance.edges_idx[toll]]
instance.compute_obj_from_price(prices)

l = np.zeros_like(lam.x)
l[-1] = 8
np.dot(l, A1)
np.dot(l, A2) <= d
np.dot(l, A2)

solver.solution
g.solution