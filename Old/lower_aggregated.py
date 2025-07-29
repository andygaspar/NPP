import numpy as np

from Instance.instance import Instance, Commodity
from gurobipy import Model, GRB, quicksum, Env


class LowerSolverAggregated:

    def __init__(self, instance: Instance, n_particles):
        self.n_particles = n_particles
        self.instance = instance
        self.n_paths = len(self.instance.toll_paths)
        self.first_run = True

        self.m = Model('CVRP')
        self.m.setParam("OutputFlag", 0)
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE

        self.x = self.m.addVars([(n, p, k) for n in range(self.n_particles) for p in self.instance.toll_paths
                                 for k in self.instance.commodities], lb=0, ub=1, vtype=GRB.CONTINUOUS)
        self.x_od = self.m.addVars([(n, k) for n in range(self.n_particles) for k in self.instance.commodities],
                                   lb=0, ub=1, vtype=GRB.CONTINUOUS)

        self.path_dict = dict(zip(self.instance.toll_paths, range(self.n_paths)))
        self.particle_minimum_cost = self.m.addVars([n for n in range(self.n_particles)])

    def set_obj(self):
        self.m.setObjective(
            quicksum(
                self.particle_minimum_cost[n] for n in range(self.n_particles)
            )
        )

    def set_constraint_1(self):
        for n in range(self.n_particles):
            for k in self.instance.commodities:
                self.m.addConstr(
                    quicksum(self.x[n, p, k] for p in self.instance.toll_paths) + self.x_od[n, k] == 1
                )

    def set_particle_constraint(self, path_costs):
        path_costs = path_costs.reshape(self.n_particles, self.n_paths)
        for n in range(self.n_particles):
            self.m.addConstr(
                self.particle_minimum_cost[n] == quicksum(k.n_users *
                                                          (quicksum((k.transfer_cost[p] +
                                                                     path_costs[n, self.path_dict[p]]) *
                                                                    self.x[n, p, k] for p in self.instance.toll_paths)
                                                           + k.cost_free * self.x_od[n, k]) for k in
                                                          self.instance.commodities),
                name="new_path_"+str(n)
            )

    def set_up(self):
        self.set_obj()
        self.set_constraint_1()

    def solve(self, path_costs):
        if not self.first_run:
            self.m.reset()
            for n in range(self.n_particles):
                self.m.remove(self.m.getConstrByName("new_path_"+str(n)))
        self.first_run = False
        self.set_particle_constraint(path_costs)
        self.m.optimize()
        p_costs = path_costs.reshape(self.n_particles, self.n_paths)
        upper_obj = np.array([sum([k.n_users * p_costs[n, self.path_dict[p]] * self.x[n, p, k].x
                             for p in self.instance.toll_paths for k in self.instance.commodities])
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
