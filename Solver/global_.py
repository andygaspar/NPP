import time

import numpy as np

from Old.instance_old import Instance, Commodity
from gurobipy import Model, GRB, quicksum


def stop(model, where):

    if where == GRB.Callback.MIP:
        num_current_solutions = model.cbGet(GRB.Callback.MIP_SOLCNT)
        run_time = model.cbGet(GRB.Callback.RUNTIME)

        if run_time > model._time_limit: #or num_current_solutions >= model._min_sol_num:
            print("stop at", run_time)
            model.terminate()

class GlobalSolver:

    def __init__(self, instance: Instance, time_limit=None, min_sol_num=None, verbose=False):
        self.instance = instance
        self.m = Model('CVRP')
        self.m._time_limit = time_limit
        self.m._min_sol_num = min_sol_num
        if not verbose:
            self.m.setParam("OutputFlag", 0)
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MAXIMIZE

        self.eps = 1e-4

        self.p = self.m.addVars([(p, k) for p in self.instance.paths for k in self.instance.commodities])
        self.x = self.m.addVars([(p, k) for p in self.instance.paths for k in self.instance.commodities],
                                vtype=GRB.BINARY)
        self.t = self.m.addVars([p for p in self.instance.paths])
        self.current_solution = None
        self.current_val = None

        self.solution = None
        self.best_val = None

    def set_obj(self):
        k: Commodity
        self.m.setObjective(quicksum(k.n_users * self.p[(p, k)]
                                     for p in self.instance.paths for k in self.instance.commodities))

    def set_constraints(self):

        for k in self.instance.commodities:
            for q in self.instance.paths:
                self.m.addConstr(
                    quicksum([(k.c_p[p.name] - k.c_od) * self.x[p, k] + self.p[p, k]
                              for p in self.instance.paths]) - self.t[q]
                    <= k.c_p[q.name] - k.c_od
                )

        for k in self.instance.commodities:
            self.m.addConstr(
                quicksum([(k.c_p[p.name] - k.c_od) * self.x[p, k] + self.p[p, k]
                          for p in self.instance.paths]) <= 0
            )

            self.m.addConstr(
                quicksum(self.x[p, k] for p in self.instance.paths) <= 1
            )

        for k in self.instance.commodities:
            for p in self.instance.paths:
                self.m.addConstr(
                    self.p[p, k] <= (k.M_p[p.name] - self.eps) * self.x[p, k]
                )
                self.m.addConstr(
                    self.t[p] - self.p[p, k] <= p.N_p * (1 - self.x[p, k])
                )
                self.m.addConstr(
                    self.p[p, k] <= self.t[p]
                )

    def solve(self):
        self.set_obj()
        self.set_constraints()
        if self.m._time_limit is not None: #and self.m._min_sol_num is not None:
            tic = time.time()
            self.m.optimize(stop)
            toc = time.time()
            self.current_solution = np.zeros(len(self.instance.paths))
            for i, p in enumerate(self.instance.paths):
                self.current_solution[i] = self.t[p].x
            self.best_val = self.m.objVal
        else:
            tic = time.time()
            self.m.optimize()
            toc = time.time()
            print(self.m.status)
            # self.solution = np.zeros(len(self.instance.paths))
            self.solution = {}
            for p in self.instance.paths:
                self.solution[p] = self.t[p].x
            self.obj = self.m.objVal

        return toc - tic, self.m.objVal, self.m.ObjBound, [self.t[p].x for p in self.instance.paths]



    def print_model(self):
        # for p in self.instance.paths:
            # self.instance.npp.edges[p]["weight"] = self.t[p].x
            # print(p, self.instance.npp.edges[p]["weight"])

        best_val = 0
        for i, k in enumerate(self.instance.commodities):
            found = False
            for j, p in enumerate(self.instance.paths):
                if i == 2 and j == 1:
                    print(k.c_p[p], self.t[p].x, k.c_p[p] + self.t[p].x)
                if self.x[p, k].x > 0.9:
                    gain = self.t[p].x * k.n_users * self.x[p, k].x
                    best_val += gain

                    print('comm', i, '  p', j, ' n users', k.n_users, "  transf", k.c_p[p],
                          "  p", self.t[p].x, "  cost ", self.t[p].x + k.c_p[p],
                          "  free", k.c_od, "   gain", gain)
                    found = True
            # if not found:
            #     print(k, 'c_od', k.c_od)

        print('actual best_val', best_val)

    def get_prices(self):
        return np.array([self.t[p].x for p in self.instance.paths])



# 19.909572739759547 < 19.909572739759554