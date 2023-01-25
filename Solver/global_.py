import time

import numpy as np

from Instance.instance import Instance, Commodity
from gurobipy import Model, GRB, quicksum, Env


def stop(model, where):

    if where == GRB.Callback.MIP:
        num_current_solutions = model.cbGet(GRB.Callback.MIP_SOLCNT)
        run_time = model.cbGet(GRB.Callback.RUNTIME)

        if run_time >= model._time_limit or num_current_solutions >= model._min_sol_num:
            print("stop at", run_time)
            model.terminate()


def store_partial_solution(model, where):

    if where == GRB.Callback.MIP:
        run_time = model.cbGet(GRB.Callback.RUNTIME)
        idx = model._time_stamp_idx

        if idx < model._n_time_stamps:
            if run_time >= model._time_stamps[idx] and not model._time_stamps_reached[idx]:
                print("reached", model._time_stamps[idx])
                model._time_stamps_reached[idx] = True
                model._time_stamp_idx += 1
                model._partial_objs[idx] = model.cbGet(GRB.Callback.MIP_OBJBST)
                print(model._time_stamps_reached)
                print(model._partial_objs)



class GlobalSolver:

    def __init__(self, instance: Instance, time_limit=None, min_sol_num=None, verbose=False,
                 time_stamps=None):
        self.instance = instance
        self.m = Model('CVRP')

        self.time_stamps, self.n_time_stamps = time_stamps, len(time_stamps)
        self.m._time_limit = time_limit
        self.m._min_sol_num = min_sol_num
        self.partial_obj_vals = None


        if not verbose:
            self.m.setParam("OutputFlag", 0)
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MAXIMIZE

        if self.time_stamps is not None:
            self.m._time_stamps = time_stamps
            self.m._time_stamps_reached = np.zeros(self.n_time_stamps, dtype=np.bool)
            self.m._time_stamp_idx = 0
            self.m._n_time_stamps = self.n_time_stamps
            self.m._partial_objs = np.zeros(self.n_time_stamps)
            self.m._partial_solutions = []

        self.eps = 1e-4

        self.p = self.m.addVars([(p, k) for p in self.instance.toll_paths for k in self.instance.commodities])
        self.x = self.m.addVars([(p, k) for p in self.instance.toll_paths for k in self.instance.commodities],
                                vtype=GRB.BINARY)
        self.t = self.m.addVars([p for p in self.instance.toll_paths])
        self.current_solution = None
        self.current_val = None

        self.solution = None
        self.best_val = None

        self.constraint_setting_time = None
        self.solving_time = None

    def set_obj(self):
        k: Commodity
        self.m.setObjective(quicksum(k.n_users * self.p[(p, k)]
                                     for p in self.instance.toll_paths for k in self.instance.commodities))

    def set_constraints(self):

        for k in self.instance.commodities:
            for q in self.instance.toll_paths:
                self.m.addConstr(
                    quicksum([(k.transfer_cost[p] - k.cost_free) * self.x[p, k] + self.p[p, k]
                              for p in self.instance.toll_paths]) - self.t[q]
                    <= k.transfer_cost[q] - k.cost_free
                )

        for k in self.instance.commodities:
            self.m.addConstr(
                quicksum([(k.transfer_cost[p] - k.cost_free) * self.x[p, k] + self.p[p, k]
                          for p in self.instance.toll_paths]) <= 0
            )

            self.m.addConstr(
                quicksum(self.x[p, k] for p in self.instance.toll_paths) <= 1
            )

        for k in self.instance.commodities:
            for p in self.instance.toll_paths:
                self.m.addConstr(
                    self.p[p, k] <= (k.M_p[p] - self.eps) * self.x[p, k]
                )
                self.m.addConstr(
                    self.t[p] - self.p[p, k] <= self.instance.N_p[p] * (1 - self.x[p, k])
                )
                self.m.addConstr(
                    self.p[p, k] <= self.t[p]
                )

    def solve(self):
        t = time.time()
        self.set_obj()
        self.set_constraints()
        self.constraint_setting_time = time.time() - t

        t = time.time()

        if self.time_stamps is not None:
            self.m.optimize(store_partial_solution)
            # print(self.m.status)
            self.solution = np.zeros(len(self.instance.toll_paths))
            for p in self.instance.toll_paths:
                self.solution = self.t[p].x
            self.obj = self.m.objVal
            self.partial_obj_vals = self.m._partial_objs


        elif self.m._time_limit is not None and self.m._min_sol_num is not None:
            self.m.optimize(stop)
            self.current_solution = np.zeros(len(self.instance.toll_paths))
            for i, p in enumerate(self.instance.toll_paths):
                self.current_solution[i] = self.t[p].x
            self.best_val = self.m.objVal

        else:
            self.m.optimize()
            print(self.m.status)
            self.solution = np.zeros(len(self.instance.toll_paths))
            for p in self.instance.toll_paths:
                self.solution = self.t[p].x
            self.obj = self.m.objVal

        self.solving_time = time.time() - t

        return self.m.objVal, self.m.objBound, [self.t[p].x for p in self.instance.toll_paths]


    def print_model(self):
        # for p in self.instance.toll_paths:
            # self.instance.npp.edges[p]["weight"] = self.t[p].x
            # print(p, self.instance.npp.edges[p]["weight"])

        best_val = 0
        for i, k in enumerate(self.instance.commodities):
            found = False
            for j, p in enumerate(self.instance.toll_paths):
                if i == 2 and j == 1:
                    print(k.transfer_cost[p], self.t[p].x, k.transfer_cost[p] + self.t[p].x)
                if self.x[p, k].x > 0.9:
                    gain = self.t[p].x * k.n_users * self.x[p, k].x
                    best_val += gain

                    print('comm', i, '  p', j, ' n users', k.n_users, "  transf", k.transfer_cost[p],
                          "  p", self.t[p].x, "  cost ", self.t[p].x + k.transfer_cost[p],
                          "  free", k.cost_free, "   gain", gain)
                    found = True
            # if not found:
            #     print(k, 'c_od', k.cost_free)

        print('actual best_val', best_val)



# 19.909572739759547 < 19.909572739759554