import os
import random

import networkx as nx
import numpy as np
import pandas as pd

from Arc.ArcInstance.del_test import DelaunayInstanceTest
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

n_arcs = 104
dim_grid = (5, 12)
# dim_grid = (3, 4)
n_locations = dim_grid[0] * dim_grid[1]
# toll_proportion = 10
toll_proportion = [5, 10, 15, 20]
# n_commodities = 10
n_commodities = [10, 50, 60]
graphs = [DelaunayInstance, GridInstance]

# instance = DelaunayInstance(n_locations, n_arcs, dim_grid, toll_proportion[0], n_commodities[0])
# instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion[2], n_commodities[2])

# instance.show()
columns = ['run', 'graphs', 'toll_pr', 'n_com', 'g_obj', 'g_time', 'exact_obj', 'exact_time', 'MIP_gap', 'status']
row = 0
df = pd.DataFrame(columns=columns)
for graph in graphs:
    for n_c in n_commodities:
        for t_p in toll_proportion:
            for run in range(10):
                random.seed(run)
                np.random.seed(run)
                instance = graph(n_locations, n_arcs, dim_grid, t_p, n_c, seed=run)
                print("\nProblem ", instance.name, n_c, t_p, run)
                # instance.show()

                solver = ArcSolver(instance=instance, symmetric_costs=False)
                solver.solve(time_limit=3, verbose=False)  # int(pso.time))

                ITERATIONS = 1000
                POPULATION_SIZE = 128

                g = GeneticArc(population_size=POPULATION_SIZE, npp=instance, offspring_rate=0.2)
                g.run_cpp(ITERATIONS, verbose=False, n_threads=16, seed=0)

                print(g.time, solver.time, g.best_val, solver.obj, 1 - g.best_val / solver.obj)

                df.loc[row] = [run, instance.name, t_p, n_c, g.best_val, g.time, solver.obj, solver.time, solver.m.MIPGap, solver.m.status]
                row += 1

            df.to_csv('arc_test_solver.csv')
# solver.obj = 3966.
# print(solver.time)

# adj, prices = solver.get_adj_solution()
# instance.dijkstra(adj, prices, instance.commodities[0])
# print(instance.compute_obj(adj, prices))

# print(solver.get_tolls())
#
# g = GeneticArc(64, instance, offspring_rate=0.2)
# g.run_cpp_h(100, verbose=True, n_threads=16, seed=0)
#
# g.run_cpp(0, verbose=True, n_threads=16, seed=0)
#
# # pop, vals = g.genetic_cpp.get_results()
# print(solver.time, g.time, solver.obj, g.best_val)
#
# # print(solver.solution)
# # print(g.solution)
#
# # print(instance.compute_obj(solver.adj_solution, solver.mat_solution))
# print(instance.compute_obj(g.adj_solution, g.mat_solution))
# print(1 - g.best_val / solver.obj)
# print(run_arc_heuristic(instance, g.adj_solution.copy(), g.mat_solution.copy())[1])
# print()
#
#
# run_arc_heuristic2(instance, g.adj_solution, g.mat_solution)
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


