import os
import random

import networkx as nx
import numpy as np
import pandas as pd

# from Arc.ArcInstance.del_test import DelaunayInstanceTest
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
# 5 *12
# dim_grid = (3, 4)
n_locations = (dim_grid[0] - 1) * (dim_grid[1] - 2)
# toll_proportion = 10
toll_proportion = [10, 15, 20] #
# n_commodities = 10
n_commodities = [40, 60, 80]
graphs = [DelaunayInstance, GridInstance]

free_path_distribution = []
TIMELIMIT = 3600

ITERATIONS = 10000
POPULATION_SIZE = 128

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
                instance = graph(n_locations, n_arcs, dim_grid, t_p, n_c,  seed=run)
                print("\nProblem ", instance.name, n_c, t_p, run, len(instance.npp.edges))
                # instance.show()

                solver = ArcSolver(instance=instance, symmetric_costs=False)
                solver.solve(time_limit=TIMELIMIT, verbose=False)  # int(pso.time))

                g = GeneticArc(population_size=POPULATION_SIZE, npp=instance, offspring_rate=0.2)
                g.run_cpp(ITERATIONS, verbose=False, n_threads=16, seed=run)

                print(g.time, solver.time, g.best_val, solver.obj, g.best_val / solver.obj)

                # ae, ae_obj = run_arc_heuristic2(instance, *g.get_mats(g.solution))

                # print(g.time, solver.time, g.best_val, solver.obj, ae_obj, (1 - g.best_val / ae_obj)*100, g.best_val / solver.obj)

                df.loc[row] = [run, instance.name, t_p, n_c, g.best_val, g.time, solver.obj, solver.time, solver.m.MIPGap, solver.m.status]
                row += 1

            df.to_csv('Results/arc_results.csv', index=False)

import pandas as pd

df = pd.read_csv('Results/arc_results.csv')
df['gap'] = df.g_obj/df.exact_obj


df.gap.mean()

