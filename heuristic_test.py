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
        self.toll_free = True if self.toll_bool.sum() == 0 else False

    def set_cost(self, T):
        self.profit = (T * self.toll_bool).sum()
        self.cost = self.fixed_cost + self.profit

    def __eq__(self, other):
        return other == self.path

    def __lt__(self, other):
        if self.cost < other.cost:
            return True
        if self.cost == other.cost:
            if self.profit > other.profit:
                return True
            else:
                return False
        else:
            return False

    def __str__(self):
        return str(self.path) +  ' ' + str(self.cost) + ' ' + str(self.profit) + ' ' + str(self.toll_free)

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

    def set_cost(self, T):
        for path in self.paths:
            path.set_cost(T)
        self.paths = sorted(self.paths)

os.environ['GUROBI_HOME'] = '/opt/gurobi1203/linux64'
os.environ['LD_LIBRARY_PATH'] = '/opt/gurobi1203/linux64/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.system("Arc/Arc_GA/install_arc.sh")

seed = 9

random.seed(seed)
np.random.seed(seed)

N = 12 ** 2
COMMODITIES = 40
TOLL_PROPORTION = 0.2

TIME_LIMIT = 60

tt = time.time()
grid = GridInstance(COMMODITIES, TOLL_PROPORTION, N)
# grid.save_cpp_problem('new_test')

g = GeneticArc(516, grid, offspring_rate=0.5, mutation_rate=0.02)
g.run_cpp_heuristic(3000, dijkstra_every=100, verbose=True, n_threads=16, seed=0)

# grid.draw()
print('instance time', time.time() - tt)

commodities: List[HCommodity] = []

N_p = np.array([toll.N_p for toll in grid.tolls])
toll_idx = dict(zip(grid.arc_tolls, range(grid.n_tolls)))

for c in grid.commodities:
    commodities.append(HCommodity(c))
    _, _, path = grid.get_opt_path(np.zeros(grid.n_tolls), commodities[-1].commodity)
    commodities[-1].add_path(grid, path, np.zeros(grid.n_tolls))
    _, _, path = grid.get_opt_path(np.ones(grid.n_tolls)*10000, commodities[-1].commodity)
    commodities[-1].add_path(grid, path, np.ones(grid.n_tolls)*10000)
    for toll in commodities[-1].paths[0].tolls:
        T = np.zeros(grid.n_tolls)
        T[toll_idx[toll]] = 10000
        _, _, path = grid.get_opt_path(T, commodities[-1].commodity)
        commodities[-1].add_path(grid, path, T)


g = GeneticArc(256, grid, offspring_rate=0.5, mutation_rate=0.02)
g.run_cpp_heuristic(10000, dijkstra_every=100, verbose=True, n_threads=16, seed=0)



best = 0
for iter in range(100):
    new_population = g.population.copy()
    for i in random.choices(range(g.pop_size), k= 16):
        model = gb.Model()
        model.setParam('OutputFlag', 0)

        ub = np.array([toll.N_p for toll in grid.tolls])
        T_var = model.addMVar(grid.n_tolls, ub=N_p)

        total_bool = np.zeros(grid.n_tolls, dtype=int)
        for commodity in commodities:
            commodity.set_cost(g.population[i])
            total_bool += commodity.paths[0].toll_bool

            for j in range(1, len(commodity.paths)):
                model.addConstr(commodity.paths[0].fixed_cost + (T_var * commodity.paths[0].toll_bool).sum() <=
                                commodity.paths[j].fixed_cost + (T_var * commodity.paths[j].toll_bool).sum())
                total_bool += commodity.paths[j].toll_bool

        model.setObjective(gb.quicksum((T_var * commodity.paths[0].toll_bool * commodity.commodity.n_users).sum() for commodity in commodities),
                           GRB.MAXIMIZE)

        model.optimize()
        total_bool = total_bool.astype(bool)

        T_sol = T_var.x * total_bool + g.population[i] * (1 - total_bool)
        res = grid.compute_obj_from_T(T_sol)
        profit_before = sum([c.paths[0].profit * c.commodity.n_users for c in commodities])

        for commodity in commodities:
            _, _, path = grid.get_opt_path(T_sol, commodity.commodity)
            commodity.add_path(grid, path, T_sol)
            _, _, path = grid.get_opt_path(g.population[i], commodity.commodity)
            commodity.add_path(grid, path, g.population[i])
        if res > g.vals[i]:
            new_population[i] = T_sol

        print(res, g.vals[i], grid.compute_obj_from_T(g.population[i]), 'obj', model.objVal, sum([len(c.paths) for c in commodities]))
        if res > best:
            best = res
            T_best = T_sol
    g.re_run_h(new_population, iterations = 100)
    print('final out', g.vals[0], grid.compute_obj_from_T(g.population[0]))






T_best = None
best = 0
POP_SIZE = 16
OFF_SIZE = 8
T = np.zeros((POP_SIZE + OFF_SIZE, grid.n_tolls))
for i in range(POP_SIZE):
    T[i] = np.array([np.random.uniform(0, toll.N_p) for toll in grid.tolls])
mask = np.random.random(len(T)) > 0.5

g = GeneticArc(256, grid, offspring_rate=0.5)
g.run_cpp_heuristic(10000, dijkstra_every=100, verbose=True, n_threads=16, seed=0)



num_paths = sum([len(c.paths) for c in commodities])
print('num paths', num_paths)

for i in range(g.pop_size):
    for c in commodities:
        _, _, path = grid.get_opt_path(g.population[i], c.commodity)
        c.add_path(grid, path, g.population[i])

num_paths = sum([len(c.paths) for c in commodities])
print('num paths', num_paths)
for _ in range(2):
    print('*********** iter', _)
    for i in range(g.pop_size):
        model = gb.Model()
        model.setParam('OutputFlag', 0)



        ub = np.array([toll.N_p for toll in grid.tolls])
        T_var = model.addMVar(grid.n_tolls, ub=np.array([toll.N_p for toll in grid.tolls]))

        model.setObjective(T_var.sum() + gb.quicksum((T_var * commodity.paths[0].toll_bool).sum() for commodity in commodities), GRB.MAXIMIZE)
        for commodity in commodities:
            commodity.set_cost(g.population[i])

            for j in range(1, len(commodity.paths)):

                model.addConstr(commodity.paths[0].fixed_cost + (T_var * commodity.paths[0].toll_bool).sum() <=
                                commodity.paths[j].fixed_cost + (T_var * commodity.paths[j].toll_bool).sum())

        model.optimize()
        T_sol = T_var.x
        res = grid.compute_obj_from_T(T_sol)
        profit_before = sum([c.paths[0].profit * c.commodity.n_users for c in commodities])
        if res < g.vals[i]:
            for commodity in commodities:
                _, _, path = grid.get_opt_path(T_sol, commodity.commodity)
                commodity.add_path(grid, path, T)

        if res > best:
            best = res
            T_best = T_sol

        res_1 = grid.compute_obj_from_T(g.population[i])
        num_paths = sum([len(c.paths) for c in commodities])
        print('hhhh        ', res, g.vals[i], res_1, profit_before, num_paths)

problem_np = ArcSolverNp(grid)
problem_np.solve(verbose=True, time_limit=TIME_LIMIT)
print(problem_np.obj)

grid.compute_obj_from_T(problem_np.T.x)

problem_np.T.x

RED = '\033[91m'
RESET = '\033[0m'


ppp = 0
for commodity in commodities:
    commodity.set_cost(T_best)
    print(commodity)
    for i, item in enumerate(grid.get_opt_path(problem_np.T.x, commodity.commodity)[2]):
        if item in grid.arc_tolls:
            print(f"{RED}{item}{RESET}", end=" ")
        else:
            print(item, end=" ")
    print()
    cost, prof0, paths = grid.get_opt_path(g.population[0], commodity.commodity)
    for i, item in enumerate(grid.get_opt_path(g.population[0], commodity.commodity)[2]):
        if item in grid.arc_tolls:
            print(f"{RED}{item}{RESET}", end=" ")
        else:
            print(item, end=" ")
    print(prof0)
    cost, prof1, paths = grid.get_opt_path(T_best, commodity.commodity)

    ppp += prof1
    for item in paths:
        if item in grid.arc_tolls:
            print(f"{RED}{item}{RESET}", end=" ")
        else:
            print(item, end=" ")
    print(prof1)
    print()
