# import numpy as np
import copy
import os
import random
import time

import networkx as nx
import numpy as np

from Arc.ArcInstance.arc_instance import ArcInstance, ArcCommodity

from gurobipy import Model, GRB, quicksum  # , Env
from functools import partial


class Incumbent:
    def __init__(self):
        self.times = []
        self.sol_list = []


def add_current_sol(model: Model, where, incumbent_obj):
    if where == GRB.Callback.MIPSOL:
        incumbent_obj.sol_list.append(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        incumbent_obj.times.append(time.time() - model._start_time)


class ArcSolver:

    def __init__(self, instance: ArcInstance, symmetric_costs=False):
        self.solution = None
        self.obj = None
        self.time = None
        self.time_constr = time.time()

        self.best_bound = None
        self.gap = 0
        self.status = 0
        self.symmetric_costs = symmetric_costs
        self.instance = copy.deepcopy(instance)
        self.adj = self.instance.get_adj().copy()
        self.adj_solution = None
        self.prices = np.zeros_like(self.instance.get_adj())
        self.m = Model('CVRP')
        self.m.modelSense = GRB.MAXIMIZE
        self.status_dict = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED',
                            6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT'}

        self.tol = 1e-9

        self.x = self.m.addVars([(a.idx, k) for a in self.instance.tolls for k in self.instance.commodities],
                                vtype=GRB.BINARY)
        self.y = self.m.addVars([(a.idx, k) for a in self.instance.free for k in self.instance.commodities])
        self.t = self.m.addVars([(a.idx, k) for a in self.instance.tolls for k in self.instance.commodities], ub=10000)
        self.T = self.m.addVars([a.idx for a in self.instance.tolls], ub=10000)
        self.la = self.m.addVars([(i, k) for i in self.instance.g.nodes for k in self.instance.commodities])
        self.m.setParam("OptimalityTol", self.tol)
        self.m.setParam("FeasibilityTol", self.tol)

        self.incumbent = Incumbent()

    def set_obj(self):
        k: ArcCommodity
        self.m.setObjective(quicksum(k.n_users * self.t[a.idx, k]
                                     for a in self.instance.tolls for k in self.instance.commodities), sense=GRB.MAXIMIZE)

    def set_constraints(self):
        # 1.9b
        self.time_constr = time.time()
        for k in self.instance.commodities:
            for i in self.instance.g.nodes:
                in_tolls, out_tolls = self.incident_edges(i, self.instance.tolls)
                in_free, out_free = self.incident_edges(i, self.instance.free)
                b = -1 if i == k.origin else (1 if i == k.destination else 0)
                self.m.addConstr(
                    (quicksum([self.x[a.idx, k] for a in in_tolls])
                     + quicksum(self.y[a.idx, k] for a in in_free)) -  # i+
                    (quicksum([self.x[a.idx, k] for a in out_tolls])
                     + quicksum(self.y[a.idx, k] for a in out_free))  # i-
                    == b
                    , name='flow ' + str(k) + ' ' + str(i))
        # 1.9c
        for k in self.instance.commodities:
            for a in self.instance.tolls:
                self.m.addConstr(
                    self.la[a.idx[1], k] - self.la[a.idx[0], k] <= a.c_a + self.T[a.idx]
                , name='lambda T c ' + str(k) + ' ' + str(a.idx))
            # 1.9d
            for a in self.instance.free:
                self.m.addConstr(
                    self.la[a.idx[1], k] - self.la[a.idx[0], k] <= a.c_a
                , name='lambda c ' + str(k) + ' ' + str(a.idx))
        # 1.9e
        for k in self.instance.commodities:
            self.m.addConstr(
                quicksum([(a.c_a * self.x[a.idx, k]) + self.t[a.idx, k] for a in
                          self.instance.tolls]) +
                quicksum([a.c_a * self.y[a.idx, k] for a in self.instance.free])
                == self.la[k.destination, k] - self.la[k.origin, k]
            , name='lambda od ' + str(k))

            # self.m.addConstr(self.la[k.origin, k] == 0)

        # 1.9f, g, h
        for k in self.instance.commodities:
            for a in self.instance.tolls:
                self.m.addConstr(
                    self.t[a.idx, k] <= k.M_p[a.idx] * self.x[a.idx, k]
                , name='M ' + str(k) + ' ' + str(a.idx))
                self.m.addConstr(
                    self.T[a.idx] - self.t[a.idx, k] <= a.N_p * (1 - self.x[a.idx, k])
                , name='Np ' + str(k) + ' ' + str(a.idx))
                self.m.addConstr(
                    self.t[a.idx, k] <= self.T[a.idx]
                , name='t<T ' + str(k) + ' ' + str(a.idx))

        self.time_constr = time.time() - self.time_constr

        # for toll in self.instance.tolls:
        #     self.m.addConstr(self.T[toll.idx] == self.T[(toll.idx[1], toll.idx[0])])

    @staticmethod
    def incident_edges(i, edges):
        in_edges, out_edges = [], []
        for a in edges:
            if a.idx[0] == i:  # (i, .. )  i-
                out_edges.append(a)
            if a.idx[1] == i:  # (i, .. )  i-
                in_edges.append(a)
        return in_edges, out_edges

    def solve(self, time_limit=None, verbose=False):
        self.time = time.time()
        self.set_obj()

        self.set_constraints()

        if not verbose:
            self.m.setParam("OutputFlag", 0)
        if time_limit is not None:
            self.m.Params.timelimit = time_limit

        callback = partial(add_current_sol, incumbent_obj=self.incumbent)
        self.m._start_time = time.time()
        self.m.optimize(callback)
        self.time = time.time() - self.time

        self.status = self.status_dict[self.m.status]
        # print(self.status)
        self.obj = self.m.objval
        self.best_bound = self.m.getAttr('ObjBound')
        self.gap = self.m.MIPGap

        self.assign_solution()

    def solve_max_price(self):
        self.time = time.time()
        self.set_obj()
        self.set_constraints()
        for k in self.instance.commodities:
            for a in self.instance.tolls:
                if a.idx in k.solution_edges:
                    self.m.addConstr(self.x[a.idx, k] == 1)
                else:
                    self.m.addConstr(self.x[a.idx, k] == 0)
            for a in self.instance.free:
                if a.idx in k.solution_edges:
                    self.m.addConstr(self.y[a.idx, k] == 1)
                else:
                    self.m.addConstr(self.y[a.idx, k] == 0)
        # for a in sol:
        #     self.m.addConstr(self.T[a] == sol[a])
        self.m.optimize()
        self.m.computeIIS()
        for c in self.m.getConstrs():
            if c.IISConstr:
                print(f'\t{c.constrname}: {self.m.getRow(c)} {c.Sense} {c.RHS}')

        self.obj = self.m.objval
        self.time = time.time() - self.time
        self.assign_solution()
        self.best_bound = self.m.getAttr('ObjBound')

    def get_mats(self):
        price_solution = np.zeros((len(self.instance.g.nodes), len(self.instance.g.nodes)))
        for a in self.instance.tolls:
            price_solution[a.idx] = self.T[a.idx].x
        adj_solution = self.instance.get_adj() + price_solution
        return adj_solution, price_solution

    def assign_solution(self):
        self.adj_solution, self.prices = self.get_mats()
        self.solution = {a: self.T[a.idx].x for a in self.instance.tolls}
        self.instance.assign_paths(self.adj_solution, self.prices)

    def solution_debug(self):
        def get_edge(node, sol_edg):
            for e in sol_edg:
                if e.idx[0] == node:
                    return e

        for k in self.instance.commodities:
            k.solution_edges, solution_edges = [], []
            for p in self.instance.tolls:
                if self.x[p.idx, k].x > 0.9:
                    solution_edges.append(p)
            for p in self.instance.free:
                if self.y[p.idx, k].x > 0.9:
                    solution_edges.append(p)
            node = k.origin
            while node != k.destination:
                k.solution_edges.append(get_edge(node, solution_edges))
                node = k.solution_edges[-1].idx[1]

        profit = 0
        for k in self.instance.commodities:
            for e in k.solution_edges:
                if e in self.instance.tolls:
                    profit += self.T[e.idx].x * k.n_users
        print(profit)
