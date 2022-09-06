import numpy as np

from Instance.instance import Instance, Commodity
from gurobipy import Model, GRB, quicksum, Env


class GlobalSolver:

    def __init__(self, instance: Instance):
        self.instance = instance
        self.m = Model('CVRP')
        self.m.setParam("OutputFlag", 0)
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MAXIMIZE

        self.p = self.m.addVars([(p, k) for p in self.instance.p for k in self.instance.commodities])
        self.x = self.m.addVars([(p, k) for p in self.instance.p for k in self.instance.commodities], vtype=GRB.BINARY)
        self.t = self.m.addVars([p for p in self.instance.p])

    def set_obj(self):
        k: Commodity
        self.m.setObjective(quicksum(k.n_users * self.p[(p, k)]
                                     for p in self.instance.p for k in self.instance.commodities))

    def set_constraints(self):

        for k in self.instance.commodities:
            for q in self.instance.p:
                self.m.addConstr(
                    quicksum([(k.c_p[p] - k.c_od) * self.x[p, k] for p in self.instance.p]) - self.t[q] <= k.c_p[
                        q] - k.c_od
                )

        for k in self.instance.commodities:
            self.m.addConstr(
                quicksum([(k.c_p[p] - k.c_od) * self.x[p, k] for p in self.instance.p]) <= 0
            )

            self.m.addConstr(
                quicksum(self.x[p, k] for p in self.instance.p) <= 1
            )

        for k in self.instance.commodities:
            for p in self.instance.p:
                self.m.addConstr(
                    self.p[p, k] <= k.M_p[p] * self.x[p, k]
                )
                self.m.addConstr(
                    self.t[p] - self.p[p, k] <= self.instance.N_p[p] * (1 - self.x[p, k])
                )
                self.m.addConstr(
                    self.p[p, k] <= self.t[p]
                )

    def solve(self):
        self.set_obj()
        self.set_constraints()
        self.m.optimize()
        print(self.m.status)

        for p in self.instance.p:
            print(p, self.t[p].x)

        for k in self.instance.commodities:
            for p in self.instance.p:
                if self.x[p, k].x > 0.9:
                    print(k, p)
