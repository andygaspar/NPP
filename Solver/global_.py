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

        self.eps = 1e-2

        self.p = self.m.addVars([(p, k) for p in self.instance.toll_paths for k in self.instance.commodities])
        self.x = self.m.addVars([(p, k) for p in self.instance.toll_paths for k in self.instance.commodities], vtype=GRB.BINARY)
        self.t = self.m.addVars([p for p in self.instance.toll_paths])

    def set_obj(self):
        k: Commodity
        self.m.setObjective(quicksum(k.n_users * self.p[(p, k)]
                                     for p in self.instance.toll_paths for k in self.instance.commodities))

    def set_constraints(self):

        for k in self.instance.commodities:
            for q in self.instance.toll_paths:
                self.m.addConstr(
                    quicksum([(k.transfer_cost[p] - k.cost_free) * self.x[p, k] + self.p[p, k] for p in self.instance.toll_paths]) - self.t[q]
                    <= k.transfer_cost[q] - k.cost_free
                )

        for k in self.instance.commodities:
            self.m.addConstr(
                quicksum([(k.transfer_cost[p] - k.cost_free) * self.x[p, k] for p in self.instance.toll_paths]) <= 0
            )

            self.m.addConstr(
                quicksum(self.x[p, k] for p in self.instance.toll_paths) <= 1
            )

        for k in self.instance.commodities:
            for p in self.instance.toll_paths:
                self.m.addConstr(
                    self.p[p, k] <= k.M_p[p] * self.x[p, k]
                )
                self.m.addConstr(
                    self.t[p] - self.p[p, k] <= self.instance.N_p[p] * (1 - self.x[p, k])
                )
                self.m.addConstr(
                    self.p[p, k] <= self.t[p]
                )
        print(self.instance.commodities[8])

    def solve(self):
        self.set_obj()
        self.set_constraints()
        self.m.optimize()
        print(self.m.status)

        for p in self.instance.toll_paths:
            self.instance.npp.edges[p]["weight"] = self.t[p].x
            print(p, self.instance.npp.edges[p]["weight"])

        for k in self.instance.commodities:
            found = False
            for p in self.instance.toll_paths:
                if self.x[p, k].x > 0.9:
                    print(k, p, k.transfer_cost[p] + self.instance.npp.edges[p]["weight"])
                    found = True
            if not found:
                print(k, 'c_od', k.cost_free)


