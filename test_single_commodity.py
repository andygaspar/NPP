import copy
import os
import time

import pandas as pd
from numpy.array_api import nonzero

from Arc.ArcInstance.arc_instance import DelaunayInstance, GridInstance, VoronoiInstance
from Arc.ArcSolver.arc_solver import ArcSolver
import random
import numpy as np
from gurobipy import Model, GRB

from Arc.ArcSolver.arc_solver_np import ArcSolverNp
from Arc.genetic_arc import GeneticArc


def get_path(xx, yy, arc_tolls, arc_free):
    return [arc_tolls[i] for i in np.nonzero(xx)[0]] + [arc_free[i] for i in np.nonzero(yy)[0]]
os.system("Arc/Arc_GA/install_arc.sh")



seed = 9

random.seed(seed)
np.random.seed(seed)


N = 12**2
COMMODITIES = 30
TOLL_PROPORTION = 0.2

TIME_LIMIT = 60

tt = time.time()
grid = GridInstance(COMMODITIES, TOLL_PROPORTION, N)
# grid.draw()
print('instance time', time.time() - tt)

gh = GeneticArc(population_size=1024, npp=grid, offspring_rate=0.5, mutation_rate=0.02)
gh.run_cpp_heuristic(10000, 500, verbose=True, n_threads=16, seed=seed)

#
# single_commodity = ArcSolverNp(grid)
# val = 0
# T = np.zeros((COMMODITIES, grid.n_tolls))
# for k in range(COMMODITIES):
#     obj, t = single_commodity.solve_single_commodity(k)
#     t[t == 0] = 10000
#
#     test_problem = copy.deepcopy(grid)
#     test_problem.assign_paths(*test_problem.get_mats_from_prices(t))
#     # print(test_problem.commodities[k].solution_tolls)
#
#     T[k, :] = t
#     # print(obj)
#     val += obj
#
# # grid.save_cpp_problem('test_dijkstra')
# print(val, 'single commodity total')
#
# final_T = T.min(axis=0)
# TT = T.copy()
# TT[TT==10000] = 0
# top_T = TT.max(axis=0)
#
# print('obj test', grid.compute_obj_from_T(final_T))
# print('obj test', grid.compute_obj_from_T(top_T))
#
# sub_problem = ArcSolverNp(grid)
# sub_problem.solve_sub_problem(final_T)
#
#
# nonzero = TT.T.nonzero()
#
# values = {}
# for i, v in enumerate(nonzero[0]):
#     if v not in values:
#         values[v] = [nonzero[1][i]]
#     else:
#         values[v].append(nonzero[1][i])
#
#
# best = 0
# T_best = None
# for i in range(1000):
#     T_test = np.array([toll.N_p for toll in grid.tolls])
#     for v in values:
#         t_val = [TT[vv, v] for vv in values[v]]
#         t_val.append(grid.tolls[v].N_p)
#         T_test[v] = np.random.choice(t_val)
#     current = grid.compute_obj_from_T(T_test)
#     if current > best:
#         best = current
#         T_best = T_test
#         # print('test', best)
#
# test_problem = copy.deepcopy(grid)
# test_problem.assign_paths(*test_problem.get_mats_from_prices(T_best))

problem_np = ArcSolverNp(grid)
problem_np.solve(verbose=True, time_limit=TIME_LIMIT)
print(problem_np.obj)

grid.compute_obj_from_T(problem_np.T.x)

problem_np.T.x

problem_np.x.x
# toll_idx = dict(zip(test_problem.arc_tolls, range(test_problem.n_tolls)))
# problem_np.assign_solution()
# for c in problem_np.instance.commodities:
#     print(sum([T_best[toll_idx[toll]]*c.n_users for toll in c.solution_tolls]), c, c.solution_tolls)
#
# for c in test_problem.commodities:
#     print(sum([T_best[toll_idx[toll]]*c.n_users for toll in c.solution_tolls]), c, c.solution_tolls)
