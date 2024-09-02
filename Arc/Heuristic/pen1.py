import os
import random

import networkx as nx
import numpy as np
import gurobipy as gb
from gurobipy import GRB

from Arc.ArcInstance.arc_instance import ArcInstance
from Arc.ArcInstance.grid_instance import GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
from Arc.genetic_arc import GeneticArc


class Pen:
    def __init__(self, instance: ArcInstance):
        self.instance = instance
        self.a1, self.a2 = self.instance.get_bool_mats()

        self.A1 = np.zeros((self.instance.n_nodes, len(self.instance.edges_idx)), dtype=int)
        self.A2 = np.zeros((self.instance.n_nodes, len(self.instance.edges_idx)), dtype=int)

        for edge in self.instance.edges_idx.keys():
            if edge in self.instance.toll_arcs:
                self.A1[edge[0], self.instance.edges_idx[edge]] = 1
                self.A1[edge[1], self.instance.edges_idx[edge]] = -1
            else:
                self.A2[edge[0], self.instance.edges_idx[edge]] = 1
                self.A2[edge[1], self.instance.edges_idx[edge]] = -1

        self.b = np.zeros(self.instance.n_nodes)
        for com in self.instance.commodities:
            self.b[com.origin] -= com.n_users
            self.b[com.destination] += com.n_users

        self.c = np.zeros(len(self.instance.edges_idx))
        self.d = np.zeros(len(self.instance.edges_idx))

        # print([i.idx for i in self.instance.tolls])
        for edge in self.instance.edges_idx.keys():
            if edge in self.instance.toll_arcs:
                # print(self.instance.edges_idx[edge])
                self.c[self.instance.edges_idx[edge]] = self.instance.npp.edges[edge]['weight']
            else:
                self.d[self.instance.edges_idx[edge]] = self.instance.npp.edges[edge]['weight']

        self.obj = None

    def solve_0(self):
        quad = gb.Model()
        x = quad.addMVar(len(instance.edges_idx), name='x', lb=0, ub=np.inf)
        y = quad.addMVar(len(instance.edges_idx), name='y', lb=0, ub=np.inf)
        T = quad.addMVar(len(instance.edges_idx), name='T', lb=0, ub=np.inf)
        lam = quad.addMVar(instance.n_nodes, name='lam', lb=-np.inf, ub=np.inf)
        quad.setObjective(T @ x , GRB.MAXIMIZE)
        quad.addConstr( self.A1 @ x + self.A2 @ y == self.b)
        quad.addConstr(lam @ self.A1 <= self.c + T)
        # quad.addConstr(lam @ self.A2 <= self.d)
        # quad.addConstr(lam @ self.A1 - self.c >a= 0)
        quad.addConstr((self.c + T) @ x  + self.d @ y - lam @ self.b == 0)
        quad.optimize()
        return T.x

    def solve_1(self, M = 1):
        quad = gb.Model()
        x = quad.addMVar(len(instance.edges_idx), name='x', lb=0, ub=np.inf)
        y = quad.addMVar(len(instance.edges_idx), name='y', lb=0, ub=np.inf)

        lam = quad.addMVar(instance.n_nodes, name='lam', lb=-np.inf, ub=np.inf)
        quad.setObjective((lam @ self.A1 - self.c) @ x - M * (lam @ self.A1 @ x + self.d @ y - lam @ self.b), GRB.MAXIMIZE)
        quad.addConstr( self.A1 @ x + self.A2 @ y == self.b)
        quad.addConstr(lam @ self.A2 <= self.d)
        # quad.addConstr(lam @ self.A1 - self.c >= 0)
        quad.optimize()


    def solve_2(self, g):
        x = np.zeros(len(instance.edges_idx))
        y = np.zeros(len(instance.edges_idx))
        for com in g.npp.commodities:
            for i in range(len(com.solution_path) - 1):
                if (com.solution_path[i], com.solution_path[i + 1]) in instance.toll_arcs:
                    # print(instance.edges_idx[(com.solution_path[i], com.solution_path[i + 1])])
                    x[instance.edges_idx[(com.solution_path[i], com.solution_path[i + 1])]] += com.n_users
                else:
                    y[instance.edges_idx[(com.solution_path[i], com.solution_path[i + 1])]] += com.n_users

        quad1 = gb.Model()
        quad1.setParam('NonConvex', 2)
        T = quad1.addMVar(len(instance.edges_idx), name='y', lb=0, ub=np.inf)
        lam = quad1.addMVar(instance.n_nodes, name='lam', lb=-np.inf, ub=np.inf)
        quad1.setObjective(T @ x - 1 * ((self.c + T) @ x + np.dot(self.d, y) - lam @ self.b), GRB.MAXIMIZE)
        quad1.addConstr(lam @ self.A2 <= self.d)
        quad1.addConstr(lam @ self.A1 <= self.c + T)
        quad1.optimize()

    def solve_3(self, g):

        x = np.zeros(len(instance.edges_idx))
        y = np.zeros(len(instance.edges_idx))
        for com in g.npp.commodities:
            for i in range(len(com.solution_path) - 1):
                if (com.solution_path[i], com.solution_path[i + 1]) in instance.toll_arcs:
                    # print(instance.edges_idx[(com.solution_path[i], com.solution_path[i + 1])])
                    x[instance.edges_idx[(com.solution_path[i], com.solution_path[i + 1])]] += com.n_users
                else:
                    y[instance.edges_idx[(com.solution_path[i], com.solution_path[i + 1])]] += com.n_users

        pen1 = gb.Model()
        lam_max = 1000
        lam = pen1.addMVar(instance.n_nodes, name='lam', lb=0, ub=lam_max)
        M = 100

        pen1.setObjective((lam @ self.A1 - self.c) @ x - M * (lam @ self.A1 @ x + np.dot(self.d, y) - lam @ self.b), GRB.MAXIMIZE)

        pen1.addConstr(lam @ self.A2 <= self.d)
        # pen1.addConstr(lam @ A1 - c >= 0)

        pen1.optimize()
        self.obj = pen1.objVal

        T = np.dot(lam.x, self.A1) - self.c
        # print(T)
        # print(g.solution)




random.seed(0)
np.random.seed(0)

# os.system("Arc/Arc_GA/install_arc.sh")
n_arcs = 104
dim_grid = (5, 12)
dim_grid = (4, 3)
# 5 *12
# dim_grid = (20, 10)
n_locations = dim_grid[0] * dim_grid[1]
# toll_proportion = 10
toll_proportion = [5, 10, 15, 20]
# n_commodities = 10
n_commodities = [10, 50, 60]

# instance = DelaunayInstance(n_locations, n_arcs, dim_grid, toll_proportion[0], n_commodities[0])
# instance = GridInstance(n_locations, n_arcs, dim_grid, toll_proportion[2], n_commodities[2])

# instance.show()
row = 0

instance = GridInstance(n_locations, n_arcs, dim_grid, 20, 60, seed=0)
print(instance.n_edges)

solver = ArcSolver(instance=instance, symmetric_costs=False)
solver.solve(time_limit=5, verbose=True)  # int(pso.time))

# instance.show()

POPULATION_SIZE = 128
ITERATIONS = 1000
gen = GeneticArc(population_size=POPULATION_SIZE, npp=instance, offspring_rate=0.5, mutation_rate=0.02)
gen.run_cpp(ITERATIONS, verbose=True, n_threads=16, seed=0)

pen = Pen(instance=instance)
T = pen.solve_0()
print(pen.obj)
print(gen.best_val/solver.obj)
gen.solution
np.round(T)


ss = nx.to_numpy_array(instance.npp)
instance.edges_idx
sss = pen.d

