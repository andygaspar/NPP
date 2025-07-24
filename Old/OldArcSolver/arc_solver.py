# import numpy as np
import copy
import time

import networkx as nx
import numpy as np

from Arc.ArcInstance.arc_commodity import ArcCommodity
from Arc.ArcInstance.arc_instance import ArcInstance

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


class ArcSolverOld:

    def __init__(self, instance: ArcInstance, symmetric_costs = False):
        self.solution = None
        self.obj = None
        self.time = None
        self.mat_solution = None
        self.adj_solution = None
        self.best_bound = None
        self.gap = 0
        self.status = 0
        self.symmetric_costs = symmetric_costs
        self.instance = copy.deepcopy(instance)
        self.adj = self.instance.get_adj().copy()
        self.prices = np.zeros_like(self.instance.get_adj())
        self.m = Model('CVRP')
        self.m.modelSense = GRB.MAXIMIZE
        self.status_dict = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED',
                            6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT'}

        self.tol = 1e-9

        self.x = self.m.addVars([(a.idx, k) for a in self.instance.tolls for k in self.instance.commodities],
                                vtype=GRB.BINARY)
        self.y = self.m.addVars([(a.idx, k) for a in self.instance.free for k in self.instance.commodities])
        self.t = self.m.addVars([(a.idx, k) for a in self.instance.tolls for k in self.instance.commodities])
        self.T = self.m.addVars([a.idx for a in self.instance.tolls])
        self.la = self.m.addVars([(i, k) for i in self.instance.npp.nodes for k in self.instance.commodities])
        self.m.setParam("OptimalityTol", self.tol)
        self.m.setParam("FeasibilityTol", self.tol)

        self.incumbent = Incumbent()

    def set_obj(self):
        k: ArcCommodity
        self.m.setObjective(quicksum(k.n_users * self.t[a.idx, k]
                                     for a in self.instance.tolls for k in self.instance.commodities))

    def set_constraints(self):
        # 1.9b
        for k in self.instance.commodities:
            for j, i in enumerate(self.instance.npp.nodes):
                exiting_toll, entering_toll = self.iterations_on_arc(i, self.instance.tolls)
                exiting_free, entering_free = self.iterations_on_arc(i, self.instance.free)
                self.m.addConstr(
                    (quicksum([self.x[a.idx, k] for a in entering_toll]) + quicksum(
                        self.y[a.idx, k] for a in entering_free)) -  # i+
                    (quicksum([self.x[a.idx, k] for a in exiting_toll]) + quicksum(
                        self.y[a.idx, k] for a in exiting_free))  # i-
                    == k.b[j]
                )
        # 1.9c
        for k in self.instance.commodities:
            for a in self.instance.tolls:
                self.m.addConstr(
                    self.la[a.idx[1], k] - self.la[a.idx[0], k] <= a.c_a + self.T[a.idx]
                )
            # 1.9d
            for a in self.instance.free:
                self.m.addConstr(
                    self.la[a.idx[1], k] - self.la[a.idx[0], k] <= a.c_a
                )
        # 1.9e
        for k in self.instance.commodities:
            self.m.addConstr(
                quicksum([(a.c_a * self.x[a.idx, k]) + self.t[a.idx, k] for a in
                          self.instance.tolls]) +
                quicksum([a.c_a * self.y[a.idx, k] for a in self.instance.free])
                == self.la[k.destination, k] - self.la[k.origin, k]
            )

            # self.m.addConstr(self.la[k.origin, k] == 0)

        # 1.9f, g, h
        for k in self.instance.commodities:
            for a in self.instance.tolls:
                self.m.addConstr(
                    self.t[a.idx, k] <= k.M_p[a.idx] * self.x[a.idx, k]
                )
                self.m.addConstr(
                    self.T[a.idx] - self.t[a.idx, k] <= self.instance.N_p[a.idx] * (1 - self.x[a.idx, k])
                )
                self.m.addConstr(
                    self.t[a.idx, k] <= self.T[a.idx]
                )

        if self.symmetric_costs:
            for a in self.instance.toll_arcs_undirected:
                self.m.addConstr(self.T[a] == self.T[(a[1], a[0])])

    @staticmethod
    def iterations_on_arc(i, archi):
        exiting = []  # (i, .. )  i-
        entering = []  # ( .., i)  i+
        for a in archi:
            if a.idx[0] == i:  # (i, .. )  i-
                exiting.append(a)
            if a.idx[1] == i:  # ( .., i)  i+
                entering.append(a)
        return exiting, entering

    def solve(self, time_limit=None, verbose=False, set_bounds=True):
        self.time = time.time()
        self.set_obj()
        print('bound start')
        if set_bounds:
            self.set_bounds()
        print('bound end')

        self.set_constraints()
        print('constraint')

        if not verbose:
            self.m.setParam("OutputFlag", 0)
        if time_limit is not None:
            self.m.Params.timelimit = time_limit

        callback = partial(add_current_sol, incumbent_obj=self.incumbent)
        self.m._start_time = time.time()
        self.m.optimize(callback)
        self.time = time.time() - self.time

        self.status = self.status_dict[self.m.status]
        self.obj = self.m.objval
        self.best_bound = self.m.getAttr('ObjBound')
        self.adj_solution, self.mat_solution = self.get_adj_solution()
        self.solution = list(self.T[a.idx].x for a in self.instance.tolls)
        self.adj_solution, self.mat_solution = self.get_mats(self.solution)
        self.instance.npp = nx.from_numpy_array(self.adj_solution)
        self.gap = self.m.MIPGap
        for c in self.instance.commodities:
            c.solution_path = nx.shortest_path(self.instance.npp, c.origin, c.destination, weight='weight')
            c.solution_edges = [(c.solution_path[i], c.solution_path[i + 1]) for i in range(len(c.solution_path) - 1)]
        return self.m.objval, self.best_bound

    def get_tolls(self):
        tolls = list(self.T[a.idx].x for a in self.instance.tolls)
        return tolls

    def get_adj_solution(self):
        price_solution = np.zeros((len(self.instance.npp.nodes), len(self.instance.npp.nodes)))
        for a in self.instance.tolls:
            price_solution[a.idx] = self.T[a.idx].x
        adj_solution = self.instance.get_adj() + price_solution
        return adj_solution, price_solution

    def print_(self):
        for k in self.instance.commodities:
            print(k)
            for p in self.instance.toll_arcs:
                if self.x[p, k].x > 0.9:
                    print('x:', p)
                    print(self.t[p, k].x)
            for p in self.instance.free_arcs:
                if self.y[p, k].x > 0.9:
                    print('y:', p)

    def set_bounds(self):
        adj = self.instance.get_adj()
        adj_inf = copy.deepcopy(adj)
        for a in self.instance.tolls:
            adj_inf[a.idx[0], a.idx[1]] = 1000000

        for c in self.instance.commodities:
            dist_0 = self.instance.regular_dijkstra(adj, c.origin)
            dist_inf = self.instance.regular_dijkstra(adj_inf, c.origin)
            for a in self.instance.tolls:
                gamma_t_h_inf = self.instance.regular_dijkstra(adj_inf, a.idx[0])[a.idx[1]]
                gamma_t_d_inf = self.instance.regular_dijkstra(adj_inf, a.idx[0])[c.destination]
                gamma_h_d_0 = self.instance.regular_dijkstra(adj, a.idx[1])[c.destination]
                vals = [gamma_t_h_inf - a.c_a,
                        dist_inf[a.idx[1]] - dist_0[a.idx[0]] - a.c_a,
                        gamma_t_d_inf - gamma_h_d_0 - a.c_a,
                        dist_inf[c.destination] - dist_0[a.idx[0]] - gamma_h_d_0 - a.c_a]
                c.M_p[a.idx] = max(0, min(vals))

        self.instance.N_p = {a.idx: max([k.M_p[a.idx] for k in self.instance.commodities]) for a in self.instance.tolls}

    def get_mats(self, sol):
        for i in range(self.instance.n_tolls):
            self.prices[self.instance.tolls[i].idx] = sol[i]
        adj = self.adj + self.prices

        return adj, self.prices



