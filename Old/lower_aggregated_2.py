import numpy as np

from Instance.instance import Instance, Commodity
from gurobipy import Model, GRB, quicksum, Env


class LowerSolverAggregated2:

    def __init__(self, instance: Instance, n_particles):
        self.n_particles = n_particles
        self.instance = instance
        self.n_paths = len(self.instance.toll_paths)
        self.first_run = True

        self.m = Model('CVRP')
        self.m.setParam("OutputFlag", 0)
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE

        self.x = self.m.addMVar((self.n_particles, self.instance.n_toll_paths, self.instance.n_commodities),
                                lb=0, ub=1, vtype=GRB.CONTINUOUS)
        self.x_od = self.m.addMVar((self.n_particles, self.instance.n_commodities), lb=0, ub=1, vtype=GRB.CONTINUOUS)

        self.path_dict = dict(zip(self.instance.toll_paths, range(self.n_paths)))

    def set_obj(self, path_costs):
        path_costs = path_costs.reshape(self.n_particles, self.n_paths)
        self.m.setObjective(
            quicksum(
                quicksum(comm.n_users *
                         (quicksum((comm.transfer_cost[p] +
                                    path_costs[n, self.path_dict[p]]) *
                                   self.x[n, i, k] for i, p in enumerate(self.instance.toll_paths))
                          + comm.cost_free * self.x_od[n, k]) for k, comm in
                         enumerate(self.instance.commodities)) for n in range(self.n_particles)
            )
        )

    def set_constraint_1(self):
        for n in range(self.n_particles):
            for k in range(self.instance.n_commodities):
                self.m.addConstr(
                    self.x[n, :, k].sum() + self.x_od[n, k] == 1
                )

    def set_up(self):
        self.set_constraint_1()

    def solve(self, path_costs):
        if not self.first_run:
            self.m.reset()
            # self.m.remove(self.m.getObjective())
        self.first_run = False
        self.set_obj(path_costs)
        self.m.optimize()
        p_costs = path_costs.reshape(self.n_particles, self.n_paths)
        upper_obj = np.array([sum([comm.n_users * p_costs[n, self.path_dict[p]] * self.x[n, i, k].x
                             for i, p in enumerate(self.instance.toll_paths) for k, comm
                                   in enumerate(self.instance.commodities)])
                              for n in range(self.n_particles)])
        return upper_obj

    def print_stuff(self):

        for k in self.instance.commodities:
            found = False
            for p in self.instance.toll_paths:
                if self.x[p, k].x > 0.9:
                    print(k, p, k.transfer_cost[p] + self.instance.g.edges[p]["weight"])
                    found = True
            if not found:
                print(k, 'c_od', k.cost_free)
