import copy
import time

import pandas as pd

from Arc.ArcInstance.arc_instance import DelaunayInstance, GridInstance, VoronoiInstance
from Arc.ArcSolver.arc_solver import ArcSolver
import random
import numpy as np
from gurobipy import Model, GRB

from Arc.ArcSolver.arc_solver_np import ArcSolverNp
from Arc.genetic_arc import GeneticArc

def check(obj_, objs_):
    for o in objs_:
        if o - 1e-9 <= obj_ <= o + 1e-9:
            return True
    else:
        return False

def get_path(xx, yy, arc_tolls, arc_free):
    return [arc_tolls[i] for i in np.nonzero(xx)[0]] + [arc_free[i] for i in np.nonzero(yy)[0]]

def get_x_y(instance, T, toll_idx_, free_idx_):
    solution = dict(zip(instance.arc_tolls, T))
    com_path_1 = {k: instance.dijkstra(*instance.get_mats_from_prices(solution), k)[3] for k in instance.commodities}

    x = np.zeros((instance.n_commodities, instance.n_tolls))
    y = np.zeros((instance.n_commodities, instance.n_free))
    for j, k in enumerate(instance.commodities):
        for p in range(len(com_path_1[k]) - 1):
            e = (com_path_1[k][p], com_path_1[k][p + 1])
            if e in instance.arc_tolls:
                x[j, toll_idx_[e]] = 1
            else:
                y[j, free_idx_[e]] = 1
    return x, y


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

POP_SIZE = 16

# grid.save_cpp_problem('test_dijkstra')
upper_bounds = np.array([p.N_p for p in grid.tolls])
T = np.zeros((POP_SIZE * 2, grid.n_tolls))
vals = np.zeros(POP_SIZE * 2)
for i in range(grid.n_tolls):
    T[:POP_SIZE, i] = np.random.uniform(0, upper_bounds[i], size=POP_SIZE)

for i in range(POP_SIZE):
    solution = dict(zip(grid.arc_tolls, T[i]))
    vals[i] = grid.compute_obj(*grid.get_mats_from_prices(solution))


partial = ArcSolverNp(grid)

toll_idx = dict(zip(grid.arc_tolls, range(len(grid.arc_tolls))))
free_idx = dict(zip(grid.arc_free, range(len(grid.arc_free))))

best = 0
for iteration in range(100):

    # idxs = np.random.choice(range(POP_SIZE), 5, replace=False).tolist() + list(range(5))
    for i in range(POP_SIZE):

        parents = random.choices(range(POP_SIZE), k=2)
        T[i + POP_SIZE] = T[parents[0]]
        non_zero = np.nonzero(T[parents[1]])[0]
        for k in random.choices(non_zero, k=len(non_zero)//2):
            T[i + POP_SIZE][k] = T[parents[1]][k]

        if iteration % 20 == 0 and iteration > 0:
            x, y = get_x_y(grid, T[i + POP_SIZE], toll_idx, free_idx)
            T[i + POP_SIZE], vals[i + POP_SIZE] = partial.solve_max_price_2(x, y)

        else:
            solution = dict(zip(grid.arc_tolls, T[i]))
            vals[i + POP_SIZE] = grid.compute_obj(*grid.get_mats_from_prices(solution))

        if check(vals[i + POP_SIZE], vals[:i + POP_SIZE].tolist() + vals[i + POP_SIZE + 1:].tolist()):
            non_zero = np.nonzero(T[i + POP_SIZE])[0]
            idx = random.choices(non_zero, k=4)
            for k in idx:
                T[i + POP_SIZE][k] = np.random.uniform(0, upper_bounds[k])

    idxs = np.argsort(vals)[::-1]
    T = T[idxs]
    vals = vals[idxs]
    print('iter', iteration, vals[0])


problem_np = ArcSolverNp(grid)
problem_np.solve(verbose=True, time_limit=TIME_LIMIT)
#
#
# problem = ArcSolver(grid)
# problem.solve(verbose=True, time_limit=TIME_LIMIT)
#
# print('time constr ', problem_np.time_constr,  problem.time_constr)
# print(problem_np.obj, problem.obj)
# print(problem_np.time, problem.time)




