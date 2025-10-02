import copy
import time

import numpy as np
from gurobipy import Model, GRB

from Arc.ArcInstance.arc_instance import ArcInstance
from functools import partial


class Incumbent:
    def __init__(self):
        self.times = []
        self.sol_list = []


def add_current_sol(model: Model, where, incumbent_obj):
    if where == GRB.Callback.MIPSOL:
        incumbent_obj.sol_list.append(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        incumbent_obj.times.append(time.time() - model._start_time)


class ArcSolverNp:

    def __init__(self, instance: ArcInstance):
        self.solution = None
        self.obj = None
        self.time = None
        self.time_constr = time.time()

        self.best_bound = None
        self.gap = 0
        self.status = 0
        self.instance = copy.deepcopy(instance)
        self.adj = self.instance.get_adj().copy()
        self.adj_solution = None
        self.prices = np.zeros_like(self.instance.get_adj())
        self.m = Model('CVRP')
        self.m.modelSense = GRB.MAXIMIZE
        self.status_dict = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED',
                            6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT'}
        self.tol = 1e-9
        self.m.setParam("OptimalityTol", self.tol)
        self.m.setParam("FeasibilityTol", self.tol)

        self.x = self.m.addMVar((self.instance.n_commodities, self.instance.n_tolls), vtype=GRB.BINARY)
        self.y = self.m.addMVar((self.instance.n_commodities, self.instance.n_free), vtype=GRB.BINARY)
        self.la = self.m.addMVar((self.instance.n_commodities, self.instance.n_nodes), lb=-1e5)
        self.T = self.m.addMVar(self.instance.n_tolls)
        self.t = self.m.addMVar((self.instance.n_commodities, self.instance.n_tolls))

        self.incumbent = Incumbent()
        self.b = np.zeros((self.instance.n_commodities, self.instance.n_nodes))
        for i, k in enumerate(self.instance.commodities):
            self.b[i, k.origin] = -1
            self.b[i, k.destination] = 1

        self.A_1_bool = np.zeros((self.instance.n_nodes, self.instance.n_tolls))
        self.A_2_bool = np.zeros((self.instance.n_nodes, self.instance.n_free))

        i, j = 0, 0
        for e in self.instance.edges:
            if e in self.instance.arc_tolls:
                self.A_1_bool[e[0], i] = 1
                self.A_1_bool[e[1], i] = -1
                i += 1
            if e in self.instance.arc_free:
                self.A_2_bool[e[0], j] = 1
                self.A_2_bool[e[1], j] = -1
                j += 1

        self.c_at = np.array([self.instance.adj[e[0], e[1]] for e in self.instance.edges if e in self.instance.arc_tolls])
        self.c_af = np.array([self.instance.adj[e[0], e[1]] for e in self.instance.edges if e in self.instance.arc_free])

        self.n_k = np.array([[k.n_users for e in self.instance.arc_tolls] for k in self.instance.commodities])

        self.x_constr = None
        self.y_constr = None
        self.model_set = False

        self.m_max_price = None

    def set_constraints(self):

        N = np.array([toll.N_p for toll in self.instance.tolls])

        for k, comm in enumerate(self.instance.commodities):
            self.m.addConstr(-self.A_1_bool @ self.x[k] - self.A_2_bool @ self.y[k] == self.b[k])
            self.m.addConstr(self.A_1_bool.T @ self.la[k] <= self.c_at + self.T)
            self.m.addConstr(self.A_2_bool.T @ self.la[k] <= self.c_af)

            self.m.addConstr((self.c_at * self.x[k]).sum() + self.t[k].sum() + (self.c_af * self.y[k]).sum()
                             == self.la[k, comm.origin] - self.la[k, comm.destination])

            M = np.array([comm.M_p[e] for e in self.instance.edges if e in self.instance.arc_tolls])
            self.m.addConstr(self.t[k] <= M * self.x[k])
            self.m.addConstr(self.T - self.t[k] <= N * (1 - self.x[k]))
            self.m.addConstr(self.t[k] <= self.T)

    def set_obj(self):
        self.m.setObjective((self.n_k * self.t).sum(), sense=GRB.MAXIMIZE)

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
        self.obj = self.m.objval
        self.best_bound = self.m.getAttr('ObjBound')
        self.gap = self.m.MIPGap

        self.assign_solution()

    def get_mats(self):
        price_solution = np.zeros((len(self.instance.g.nodes), len(self.instance.g.nodes)))
        for i, e in enumerate(self.instance.arc_tolls):
            price_solution[e] = self.T[i].x
        adj_solution = self.instance.get_adj() + price_solution
        return adj_solution, price_solution

    def assign_solution(self):
        self.adj_solution, self.prices = self.get_mats()
        self.solution = {e: self.T[i].x for i, e in enumerate(self.instance.arc_tolls)}
        self.instance.assign_paths(self.adj_solution, self.prices)

    def solve_x(self, T):
        m = Model('CVRP')
        m.setParam("OutputFlag", 0)
        x = m.addMVar((self.instance.n_commodities, self.instance.n_tolls))
        y = m.addMVar((self.instance.n_commodities, self.instance.n_free))
        for k, comm in enumerate(self.instance.commodities):
            m.addConstr(-self.A_1_bool @ x[k] - self.A_2_bool @ y[k] == self.b[k])

        m.setObjective((x @ (T + self.c_at)).sum() + (y @ self.c_af).sum(), sense=GRB.MINIMIZE)
        m.optimize()
        # print(np.dot(x.x * self.n_k, T).sum())
        # print(((x.x * self.n_k) @ T).sum())

        return x.x, y.x

    def solve_max_price(self, x, y):
        if not self.model_set:
            self.m_max_price = Model('MaxPrice')
            self.set_constraints()
            self.set_obj()
            self.model_set = True
        if self.x_constr is not None:
            self.m.remove(self.x_constr)
            self.m.remove(self.y_constr)
        self.m.setParam("OutputFlag", 0)

        self.x_constr = self.m.addConstr(self.x == x)
        self.y_constr = self.m.addConstr(self.y == y)
        self.m.optimize()
        return self.T.x, self.m.objval

    def solve_max_price_2(self, x, y):

        N = np.array([toll.N_p for toll in self.instance.tolls])
        m_max_price = Model(name='MaxPrice')
        m_max_price.setParam("OutputFlag", 0)

        la = m_max_price.addMVar((self.instance.n_commodities, self.instance.n_nodes), lb=-1e5)
        T = m_max_price.addMVar(self.instance.n_tolls)
        t = m_max_price.addMVar((self.instance.n_commodities, self.instance.n_tolls))

        for k, comm in enumerate(self.instance.commodities):
            m_max_price.addConstr(self.A_1_bool.T @ la[k] <= self.c_at + T)
            m_max_price.addConstr(self.A_2_bool.T @ la[k] <= self.c_af)

            m_max_price.addConstr((self.c_at * x[k]).sum() + t[k].sum() + (self.c_af * y[k]).sum()
                             == la[k, comm.origin] - la[k, comm.destination])

            M = np.array([comm.M_p[e] for e in self.instance.edges if e in self.instance.arc_tolls])
            m_max_price.addConstr(t[k] <= M * x[k])
            m_max_price.addConstr(T - t[k] <= N * (1 - x[k]))
            m_max_price.addConstr(t[k] <= T)

        m_max_price.setObjective((self.n_k * t).sum(), sense=GRB.MAXIMIZE)
        t = time.time()
        m_max_price.optimize()
        # print('opt time', time.time() - t)
        return T.x, m_max_price.objval

    def solve_single_commodity(self, k):
        m = Model('CVRP')
        m.setParam("OutputFlag", 0)
        N = np.array([toll.N_p for toll in self.instance.tolls])

        x = m.addMVar((self.instance.n_commodities, self.instance.n_tolls), vtype=GRB.BINARY)
        y = m.addMVar((self.instance.n_commodities, self.instance.n_free), vtype=GRB.BINARY)
        la = m.addMVar((self.instance.n_commodities, self.instance.n_nodes), lb=-1e5)
        T = m.addMVar(self.instance.n_tolls)
        t = m.addMVar((self.instance.n_commodities, self.instance.n_tolls))

        comm = self.instance.commodities[k]
        m.addConstr(-self.A_1_bool @ x[k] - self.A_2_bool @ y[k] == self.b[k])
        m.addConstr(self.A_1_bool.T @ la[k] <= self.c_at + T)
        m.addConstr(self.A_2_bool.T @ la[k] <= self.c_af)

        m.addConstr((self.c_at * x[k]).sum() + t[k].sum() + (self.c_af * y[k]).sum()
                         == la[k, comm.origin] - la[k, comm.destination])

        M = np.array([comm.M_p[e] for e in self.instance.edges if e in self.instance.arc_tolls])
        m.addConstr(t[k] <= M * x[k])
        m.addConstr(T - t[k] <= N * (1 - x[k]))
        m.addConstr(t[k] <= T)
        for kk in range(self.instance.n_commodities):
            if kk != k:
                m.addConstr(t[kk] == 0)
        m.setObjective((self.n_k * t).sum(), sense=GRB.MAXIMIZE)
        m.optimize()
        return m.objval, t.x[k]

    def solve_sub_problem(self, T_init):
        m = Model('CVRP')
        # m.setParam("OutputFlag", 0)

        x = m.addMVar((self.instance.n_commodities, self.instance.n_tolls), vtype=GRB.BINARY)
        y = m.addMVar((self.instance.n_commodities, self.instance.n_free), vtype=GRB.BINARY)
        la = m.addMVar((self.instance.n_commodities, self.instance.n_nodes), lb=-1e5)
        T = m.addMVar(self.instance.n_tolls)
        t = m.addMVar((self.instance.n_commodities, self.instance.n_tolls))

        N = np.array([toll.N_p for toll in self.instance.tolls])

        for i in range(T_init.shape[0]):
            if T_init[i] == 10000:
                # for k in range(self.instance.n_commodities):
                #     m.addConstr(t[k] == 0)
                m.addConstr(T[i] == N[i])
            else:
                m.addConstr(T[i] >= T_init[i])
        # m.addConstr(x == 0)

        for k, comm in enumerate(self.instance.commodities):
            m.addConstr(-self.A_1_bool @ x[k] - self.A_2_bool @ y[k] == self.b[k])
            m.addConstr(self.A_1_bool.T @ la[k] <= self.c_at + T)
            m.addConstr(self.A_2_bool.T @ la[k] <= self.c_af)

            m.addConstr((self.c_at * x[k]).sum() + t[k].sum() + (self.c_af * y[k]).sum()
                             == la[k, comm.origin] - la[k, comm.destination])

            M = np.array([comm.M_p[e] for e in self.instance.edges if e in self.instance.arc_tolls])
            m.addConstr(t[k] <= M * x[k])
            m.addConstr(T - t[k] <= N * (1 - x[k]))
            m.addConstr(t[k] <= T)


        m.setObjective((self.n_k * t).sum(), sense=GRB.MAXIMIZE)
        m.optimize()
        print(m.objval)
        pass

