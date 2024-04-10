import os
import random
import time

import numpy as np

from Instance.instance import Instance
from Solver.genetic_solver import Genetic
from Solver.solver import GlobalSolver
# os.system("CPP/install.sh")
n_paths = 20
n_commodities = 20
VERBOSE = False
TIME_LIMIT = 30

run = 0

random.seed(run)
np.random.seed(run)

npp = Instance(n_paths=n_paths, n_commodities=n_commodities)
solver = GlobalSolver(npp, verbose=VERBOSE, time_limit=TIME_LIMIT)
solver.solve()

g = Genetic(npp, 256, 256//2, 0.02, int(n_paths / 2), verbose=VERBOSE, n_threads=None,
            seed=run)

g.run(10000)
print(g.time, g.best_val)


print(solver.obj, solver.time)

tol = 1e-18

print(npp.compute_solution_value(solver.solution_array), npp.compute_solution_value_with_tol(solver.solution_array, tol=tol))
print(npp.compute_solution_value(g.solution), npp.compute_solution_value_with_tol(g.solution, tol=tol))