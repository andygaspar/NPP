import copy
import os
import time
from typing import List

import gurobipy as gb
import pandas as pd

from Arc.ArcInstance.arc_instance import DelaunayInstance, GridInstance, VoronoiInstance, ArcCommodity, ArcInstance
from Arc.ArcSolver.arc_solver import ArcSolver
import random
import numpy as np
from gurobipy import Model, GRB

from Arc.ArcSolver.arc_solver_np import ArcSolverNp
from Arc.genetic_arc import GeneticArc


class Path:

    def __init__(self, instance: ArcInstance, path, T):

        self.path = path
        self.tolls = [toll for toll in self.path if toll in instance.arc_tolls]
        self.toll_bool = np.array([1 if toll in self.tolls else 0 for toll in instance.arc_tolls])
        self.fixed_cost = sum([instance.adj[arc[0], arc[1]] for arc in self.path])
        self.profit = (T*self.toll_bool).sum()
        self.cost = self.fixed_cost + self.profit

    def __eq__(self, other):
        for i, arc in enumerate(self.path):
            if arc != path[i]:
                return False
        else:
            return True

    def __lt__(self, other):
        if self.cost < other.cost:
            return True
        if self.cost == other.cost:
            if self.profit < other.profit:
                return True
            else:
                return False
        else:
            return False

    def __str__(self):
        return str(self.cost) + ' ' + str(self.profit)

    def __repr__(self):
        return str(self)



class HCommodity:
    def __init__(self, commodity: ArcCommodity):
        self.commodity = commodity
        self.paths = []

    def __str__(self):
        return str(self.commodity) + ' ' + str(len(self.paths))

    def __repr__(self):
        return str(self)

    def add_path(self, instance, path: np.array, T):
        if path not in self.paths:
            self.paths.append(Path(instance, path, T))


seed = 9

random.seed(seed)
np.random.seed(seed)

N = 6 ** 2
COMMODITIES = 5
TOLL_PROPORTION = 0.2

TIME_LIMIT = 60

tt = time.time()
grid = GridInstance(COMMODITIES, TOLL_PROPORTION, N)
grid.draw()
print('instance time', time.time() - tt)

commodities: List[HCommodity] = []

for c in grid.commodities:
    commodities.append(HCommodity(c))
    _, _, path = grid.get_opt_path(np.zeros(grid.n_tolls), commodities[-1].commodity)
    commodities[-1].add_path(grid, path, np.zeros(grid.n_tolls))
    _, _, path = grid.get_opt_path(np.ones(grid.n_tolls)*10000, commodities[-1].commodity)
    commodities[-1].add_path(grid, path, np.ones(grid.n_tolls)*10000)

for i in range(100):
    T = np.array([np.random.uniform(0, toll.N_p) for toll in grid.tolls])
    for c in commodities:
        _, _, path = grid.get_opt_path(T, commodities[-1].commodity)
        commodities[-1].add_path(grid, path, T)




model = gb.Model()

ub = np.array([toll.N_p for toll in grid.tolls])
T_var = model.addMVar(grid.n_tolls, ub=np.array([toll.N_p for toll in grid.tolls]))

for commodity in commodities[:1]:
    commodity.paths = sorted(commodity.paths)
    for i in range(len(commodity.paths) - 1):

        model.addConstr(commodity.paths[i].fixed_cost + (T_var * commodity.paths[i].toll_bool).sum() <=
                      commodity.paths[i + 1].fixed_cost + (T_var * commodity.paths[i + 1].toll_bool).sum())


model.setObjective(T_var.sum(), GRB.MAXIMIZE)
model.optimize()
T_sol = T_var.x

res = grid.compute_obj_from_T(T_sol)

problem_np = ArcSolverNp(grid)
problem_np.solve(verbose=True, time_limit=TIME_LIMIT)
print(problem_np.obj)

grid.compute_obj_from_T(problem_np.T.x)

problem_np.T.x

for commodity in commodities:
    print(grid.get_opt_path(problem_np.T.x, commodity.commodity)[2])
    print(grid.get_opt_path(T_sol, commodity.commodity)[2],'\n')
