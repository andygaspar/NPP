import os
import random
import time

import networkx as nx
import numpy as np
import gurobipy as gb
from gurobipy import GRB

from Arc.ArcInstance.arc_instance import ArcInstance
from Arc.ArcInstance.delunay_instance import DelaunayInstance
from Arc.ArcInstance.grid_instance import GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
from Arc.Heuristic.arc_heuristic import run_arc_heuristic
from Arc.Heuristic.arc_heuristic2 import run_arc_heuristic2
from Arc.Heuristic.arc_heuristic3 import run_arc_heuristic3, Com, HeuristicNew
from Arc.genetic_arc import GeneticArc
random.seed(0)
np.random.seed(0)

# os.system("Arc/Arc_GA/install_arc.sh")
n_arcs = 104
dim_grid = (10, 24)
# dim_grid = (4, 3)
# dim_grid = (5, 8)
# 5 *12

# dim_grid = (20, 10)
n_locations = dim_grid[0] * dim_grid[1]
# toll_proportion = 10
toll_proportion = [5, 10, 15, 20]
# n_commodities = 10
n_commodities = [10, 50, 60]

# instance = DelaunayInstance(n_locations, n_arcs, dim_grid, toll_proportion[0], n_commodities[0])
# instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion[2], n_commodities[2])

# instance.show()
# 7554.263537727229 20 60
# 2681 10 60
row = 0

tp = 10
nc = 40
ins = 'g'


if ins == 'g':
    instance = GridInstance(n_locations, n_arcs, dim_grid, tp, nc, seed=0)
else:
    instance = DelaunayInstance(n_locations, n_arcs, dim_grid, tp, nc, seed=0)
instance.show()
# for c in instance.commodities:
#     print(c)
# instance.save_problem('test_' + ins + '_' + str(tp) + '_' + str(nc))
solver = ArcSolver(instance=instance, symmetric_costs=False)
solver.solve(time_limit=20, verbose=True)  # int(pso.time))
# print(solver.obj, solver.status)
os.getcwd()
solver.obj = 2681
h = HeuristicNew(instance)
# h.run()
# print('h', h.obj)

g = GeneticArc(128, instance, offspring_rate=0.2, mutation_rate=0.1)
# tot_time = 0
# for i in range(100):
#     g.run_cpp(10, verbose=False, n_threads=16)
#     t = time.time()
#     h.run(g.prices, g.best_val)
#     tot_time += time.time() - t
#     # print(g.best_val, h.obj, tot_time)
# print(tot_time, 'time')
g.run_cpp(10000, verbose=True, n_threads=16)
h.run(g.prices, g.best_val)
print(g.best_val, h.obj, g.time(), (1 - g.best_val/solver.obj) * 100, (1 - h.obj/solver.obj) * 100)


7451/7469
7451/7492
(1 - 7451/7554)*100
(1 - 7492/7554)*100

(1 - 8265/8409)*100

# print(solver.time)
#
# adj, prices = solver.get_adj_solution()
# instance.dijkstra(adj, prices, instance.commodities[0])
# print(instance.compute_obj(adj, prices))
#
# print(solver.get_tolls())

# g = GeneticArc(64, instance, offspring_rate=0.2)
# g.run_cpp_h(10000, verbose=True, n_threads=16, seed=0)
# print("\n")
# g.run_cpp_h(10000, verbose=True, n_threads=16, seed=0, old=False)
# print("\n")
# g.run_cpp(10000, verbose=True, n_threads=16, seed=0)

# pop, vals = g.genetic_cpp.get_results()
# print(solver.time, g.time, solver.obj, g.best_val)

# print(solver.solution)
# print(g.solution)

# print(instance.compute_obj(solver.adj_solution, solver.mat_solution))
# print(instance.compute_obj(g.adj_solution, g.mat_solution))
# # print(1 - g.best_val / solver.obj)
# print(run_arc_heuristic3(instance, g.adj_solution.copy(), g.mat_solution.copy())[1])
# print()


# run_arc_heuristic(instance, g.adj_solution, g.mat_solution)
#
#
#
# def compute_val(inst, sol, tol=1e-9):
#     return inst.compute_obj(*g.get_mats(sol), tol)
#
#
# # a = solver.adj_solution
# # for c in instance.commodities:
# #     print(c.origin, c.destination)
# # aa = g.adj_solution
# #
# # sol = g.solution.copy()
# # best = g.best_val
# # for i in range(g.n_tolls):
# #     improving = True
# #     while improving:
# #         s = sol.copy()
# #         s[i] += 0.01
# #         new_val = compute_val(instance, s)
# #         if new_val > best:
#             print(new_val, i,  s[i], instance.tolls[i].N_p)
#             sol = s
#             best = new_val
#         else:
#             improving = False

