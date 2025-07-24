import os
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd

from Arc.ArcInstance.grid_instance import GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
from Arc.genetic_arc import GeneticArc
from test_np_solver import ArcNewInstance
from test_solver import ArcSolverNew

random.seed(0)
np.random.seed(0)

# os.system("Arc/Arc_GA/install_arc.sh")

n_arcs = 3*4
# 5 *12100

n = 6
# n_arcs = 5 ** 2
dim_grid = (n, n)
n_locations = dim_grid[0] * dim_grid[1]

toll_proportion = 20
n_commodities = 30


columns = ['run', 'graphs', 'toll_pr', 'n_com', 'obj', 'time', 'gap', 'status']
row = 0
df = pd.DataFrame(columns=columns)

run = 0
random.seed(run)
np.random.seed(run)
instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion, n_commodities, seed=run)
instance.save_instance('test_graph.pkl')
instance.show(edge_weight=True)

# solver = ArcSolver(instance=instance, symmetric_costs=False)
# solver.solve(time_limit=5, verbose=True)  # int(pso.time))

ITERATIONS = 10000

# g = GeneticArc(128, instance, offspring_rate=0.2)
# g.run_cpp_heuristic(ITERATIONS, dijkstra_every=100, verbose=True, n_threads=16, seed=0)
#
g = GeneticArc(128, instance, offspring_rate=0.2)
g.run_cpp(ITERATIONS, verbose=True, n_threads=16, seed=0)

# print(g.time, solver.time)
# print(g.best_val, solver.obj)

# print(solver.incumbent.times)
# print(solver.incumbent.sol_list)


with open("test_graph.pkl", 'rb') as f:  # notice the r instead of w
    graph = pickle.load(f)


inst_new = ArcNewInstance(len(graph._commodities), graph._toll_proportion, len(graph.nodes), graph, preset_graph=True)
inst_new.draw(show_cost=True)
gg = GeneticArc(128, inst_new, offspring_rate=0.2)
gg.run_cpp(ITERATIONS, verbose=True, n_threads=16, seed=0)


# for k in instance.commodities:
#     print(k.M_p)
#
# for k in inst_new.commodities:
#     print(k.M_p)
#
#
# for p in instance.tolls:
#     print(p.N_p)
#
# for p in inst_new.tolls:
#     print(p.N_p)



p = ArcSolverNew(instance)
p.solve(time_limit=60, verbose=True)

pp = ArcSolverNew(inst_new)
pp.solve(time_limit=60, verbose=True)
print(g.best_val, gg.best_val)
print(p.obj, pp.obj)




