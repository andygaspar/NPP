# import numpy as np
import numpy as np

from Arc.ArcInstance.arc_commodity import ArcCommodity
from Arc.ArcInstance.arc_instance import ArcInstance

from gurobipy import Model, GRB, quicksum  # , Env


class ArcSolver:

    def __init__(self, instance: ArcInstance):
        self.price_solution = None
        self.adj_solution = None
        self.best_bound = None
        self.instance = instance
        self.m = Model('CVRP')
        self.m.modelSense = GRB.MAXIMIZE
        self.status_dict = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED',
                            6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT'}
        self.status = None

        self.eps = 1e-2

        self.x = self.m.addVars([(a, k) for a in self.instance.toll_arcs for k in self.instance.commodities],
                                vtype=GRB.BINARY)
        self.y = self.m.addVars([(a, k) for a in self.instance.free_arcs for k in self.instance.commodities])
        self.t = self.m.addVars([(a, k) for a in self.instance.toll_arcs for k in self.instance.commodities])
        self.T = self.m.addVars([a for a in self.instance.toll_arcs])
        self.la = self.m.addVars([(i, k) for i in self.instance.npp.nodes for k in self.instance.commodities])

    def set_obj(self):
        k: ArcCommodity
        self.m.setObjective(quicksum(k.n_users * self.t[a, k]
                                     for a in self.instance.toll_arcs for k in self.instance.commodities))

    def set_constraints(self):
        # 1.9b
        for k in self.instance.commodities:
            for j, i in enumerate(self.instance.npp.nodes):
                exiting_toll, entering_toll = self.iterations_on_arc(i, self.instance.toll_arcs)
                exiting_free, entering_free = self.iterations_on_arc(i, self.instance.free_arcs)
                self.m.addConstr(
                    (quicksum([self.x[a, k] for a in entering_toll]) + quicksum(
                        self.y[a, k] for a in entering_free)) -  # i+
                    (quicksum([self.x[a, k] for a in exiting_toll]) + quicksum(
                        self.y[a, k] for a in exiting_free))  # i-
                    == k.b[j]
                )
        # 1.9c
        for k in self.instance.commodities:
            for a in self.instance.toll_arcs:
                self.m.addConstr(
                    self.la[a[1], k] - self.la[a[0], k] <= self.instance.npp.edges[a]['weight'] + self.T[a]
                )
            # 1.9d
            for a in self.instance.free_arcs:
                self.m.addConstr(
                    self.la[a[1], k] - self.la[a[0], k] <= self.instance.npp.edges[a]['weight']
                )
        # 1.9e
        for k in self.instance.commodities:
            self.m.addConstr(
                quicksum([(self.instance.npp.edges[a]['weight'] * self.x[a, k]) + self.t[a, k] for a in
                          self.instance.toll_arcs]) +
                quicksum([self.instance.npp.edges[a]['weight'] * self.y[a, k] for a in self.instance.free_arcs])
                == self.la[k.destination, k] - self.la[k.origin, k]
            )

            # self.m.addConstr(self.la[k.origin, k] == 0)

        # 1.9f, g, h
        for k in self.instance.commodities:
            for a in self.instance.toll_arcs:
                self.m.addConstr(
                    self.t[a, k] <= k.M_p[a] * self.x[a, k]
                )
                self.m.addConstr(
                    self.T[a] - self.t[a, k] <= self.instance.N_p[a] * (1 - self.x[a, k])
                )
                self.m.addConstr(
                    self.t[a, k] <= self.T[a]
                )

    @staticmethod
    def iterations_on_arc(i, archi):
        exiting = []  # (i, .. )  i-
        entering = []  # ( .., i)  i+
        for a in archi:
            if a[0] == i:  # (i, .. )  i-
                exiting.append(a)
            if a[1] == i:  # ( .., i)  i+
                entering.append(a)
        return exiting, entering

    def solve(self, time_limit=None, verbose=False):
        self.set_obj()
        self.set_constraints()
        if not verbose:
            self.m.setParam("OutputFlag", 0)
        if time_limit is not None:
            self.m.Params.timelimit = time_limit
        self.m.optimize()
        print('Global: ', self.m.objval)

        self.status = self.status_dict[self.m.status]
        print('status', self.status)
        self.best_bound = self.m.getAttr('ObjBound')
        print(' bound***** ', self.best_bound)
        self.adj_solution, self.price_solution = self.get_adj_solution()
        return self.m.objval, self.best_bound

    def get_tolls(self):
        tolls = list(self.T[a].x for a in self.instance.toll_arcs)
        return tolls

    def get_adj_solution(self):
        price_solution = np.zeros((len(self.instance.npp.nodes), len(self.instance.npp.nodes)))
        for toll in self.instance.toll_arcs:
            price_solution[toll] = self.T[toll].x
        adj_solution = self.instance.get_adj() + price_solution
        return adj_solution, price_solution

    # def bfs_simplex(self):
    #     self.set_obj()
    #     self.set_constraints()
    #     # self.m.Params.IterationLimit = 1
    #     self.m.Params.timelimit = 1.0
    #     self.m.optimize()
    #     print(self.m.status)
    #     print(self.m.objval)

    def print(self):
        # for p in self.instance.toll_arcs:
        # self.instance.npp.edges[p]["weight"] += self.T[p].x
        # print(p, self.instance.npp.edges[p]["weight"])

        # for k in self.instance.commodities:
        #     print(k)
        #     found = False
        #     for p in self.instance.toll_arcs[:self.instance.n_toll_paths]:
        #         if self.x[p, k].x > 0.9:
        #             print(p, k.transfer_cost[p] + self.instance.npp.edges[p]["weight"] + self.T[p].x)
        #             found = True
        #     for p in self.instance.toll_arcs[-self.instance.n_toll_paths:]:
        #         if self.x[p, k].x > 0.9:
        #             print(p, k.transfer_cost[(p[1], p[0])] + self.instance.npp.edges[(p[1], p[0])]["weight"] + self.T[p].x)
        #             found = True
        #     if not found:
        #         print('path without tolls', k.cost_free)

        for k in self.instance.commodities:
            print(k)
            for p in self.instance.toll_arcs:
                if self.x[p, k].x > 0.9:
                    print('x:', p)
                    print(self.t[p, k].x)
            for p in self.instance.free_arcs:
                if self.y[p, k].x > 0.9:
                    print('y:', p)





