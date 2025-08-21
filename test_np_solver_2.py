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

# grid.save_cpp_problem('test_dijkstra')

ITERATIONS = 400
POP_SIZE = 1024
g2 = GeneticArc(POP_SIZE, grid, mutation_rate=0.02)
g2.run_cpp_heuristic(ITERATIONS // 4, dijkstra_every=100, verbose=True, n_threads=16, seed=0)

partial = ArcSolverNp(grid)
# for i in range(5):
#     x, y = partial.solve_x(g2.population[i])
#     g2.population[i], obj = partial.solve_max_price(x, y)
#     print(obj, g2.vals[i])

toll_idx = dict(zip(grid.arc_tolls, range(len(grid.arc_tolls))))
free_idx = dict(zip(grid.arc_free, range(len(grid.arc_free))))

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


best = 0
for _ in range(8):
    print('iter', _)
    # idxs = np.random.choice(range(POP_SIZE), 5, replace=False).tolist() + list(range(5))
    idxs = range(20)
    objs = []
    for i in idxs:
        x, y = get_x_y(grid, g2.population[i], toll_idx, free_idx)

        g2.population[i], obj = partial.solve_max_price_2(x, y)

        if check(obj, objs):
            # print(np.nonzero(x))
            idx = random.choice(np.nonzero(x.sum(axis=0))[0])
            x, y = get_x_y(grid, g2.population[i], toll_idx, free_idx)
            _, obj_2 = partial.solve_max_price_2(x, y)
            g2.population[i][idx] -= 1
            x, y = get_x_y(grid, g2.population[i], toll_idx, free_idx)
            _, obj_3 = partial.solve_max_price_2(x, y)
            print('improve', obj, obj_2, obj_3)
        objs.append(obj)
        if best < obj:
            best = obj
            print(obj, g2.vals[i])


        # ooo = grid.compute_obj(*partial.get_mats())
        # print(ooo, obj, g2.vals[i], test_1)

    g2.genetic_cpp.run(np.ascontiguousarray(g2.population), ITERATIONS)
    g2.population, g2.vals = g2.genetic_cpp.get_results()
# g3 = GeneticArc(128, grid, mutation_rate=0.02)
# g3.run_cpp_heuristic(5000, dijkstra_every=100, verbose=True, n_threads=16, seed=0, initial_position=g2.population)
# g3.run_cpp(ITERATIONS*20, verbose=True, n_threads=16, seed=0, initial_position=g2.population)

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




