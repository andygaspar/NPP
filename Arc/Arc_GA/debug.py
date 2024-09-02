import os
print(os.getpid())
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


#os.system("Arc/Arc_GA/install_arc.sh")

n_arcs = 104
dim_grid = (5, 12)
# 5 *12
dim_grid = (3, 4)
n_locations = (dim_grid[0] - 1) * (dim_grid[1] - 2)
# toll_proportion = 10

graphs = [DelaunayInstance, GridInstance]

free_path_distribution = []
TIMELIMIT = 3600

ITERATIONS = 10000
POPULATION_SIZE = 128

# instance = DelaunayInstance(n_locations, n_arcs, dim_grid, toll_proportion[0], n_commodities[0])
# instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion[2], n_commodities[2])

# instance.show()

instance = graphs[0](n_locations, n_arcs, dim_grid, 5, 3,  seed=0)
print("\nProblem ", instance.name, 3, 5, 0, len(instance.npp.edges))
# instance.show()



g = GeneticArc(population_size=POPULATION_SIZE, npp=instance, offspring_rate=0.2, n_threads=1)
g.run_cpp_heuristic(ITERATIONS, verbose=False, n_threads=16, seed=0)

print(g.time, g.best_val)



