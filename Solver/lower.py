import numpy as np

from Instance.instance import Instance, Commodity
from gurobipy import Model, GRB, quicksum, Env


class LowerSolver:

    def __init__(self, instance: Instance):
        self.instance = instance
        self.m = Model('CVRP')
        self.m.setParam("OutputFlag", 0)
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE

        self.x = self.m.addVars([(p, k) for p in self.instance.p for k in self.instance.commodities], vtype=GRB.BINARY)
        self.x_od = self.m.addVars([k for k in self.instance.commodities], vtype=GRB.BINARY)

    def set_obj(self):
        self.m.setObjective(
            quicksum(
                k.n_users * (quicksum((k.c_p[p] + self.instance.npp.edges[p]['weight']) * self.x[p, k] for p in
                                      self.instance.p) + k.c_od * self.x_od[k]) for k in self.instance.commodities)
        )

    def set_constraints(self):

        for k in self.instance.commodities:
            self.m.addConstr(
                quicksum(self.x[p, k] for p in self.instance.p) + self.x_od[k] == 1
            )

    def solve(self):
        self.set_obj()
        self.set_constraints()
        self.m.optimize()
        print(self.m.status)

        for k in self.instance.commodities:
            found = False
            for p in self.instance.p:
                if self.x[p, k].x > 0.9:
                    print(k, p, k.c_p[p] + self.instance.npp.edges[p]["weight"])
                    found = True
            if not found:
                print(k, 'c_od', k.c_od)
