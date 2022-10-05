import numpy as np

from Instance.instance import Instance, Commodity
from gurobipy import Model, GRB, quicksum, Env


class LowerSolverAggregated:

    def __init__(self, instance: Instance, n_particles):
        self.n_particles = n_particles
        self.instance = instance
        self.n_paths = len(self.instance.p)
        self.first_run = True

        self.m = Model('CVRP')
        self.m.setParam("OutputFlag", 0)
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE

        self.x = self.m.addVars([(n, p, k) for n in range(self.n_particles) for p in self.instance.p
                                 for k in self.instance.commodities], vtype=GRB.BINARY)
        self.x_od = self.m.addVars([(n, k) for n in range(self.n_particles) for k in self.instance.commodities],
                                   vtype=GRB.BINARY)

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
                    quicksum(self.x[n, p, k] + self.x_od[n, k] for p in self.instance.p) == 1
                )

    def set_particle_constraint(self, path_costs):
        path_dict = dict(zip(self.instance.p, range(self.n_paths)))
        path_costs = path_costs.reshape(self.n_particles, self.n_paths)
        for n in range(self.n_particles):
            self.m.addConstr(
                self.particle_minimum_cost[n] == quicksum(k.n_users *
                                                          (quicksum((k.c_p[p] + path_costs[n, path_dict[p]]) *
                                                                    self.x[n, p, k] for p in
                                                                    self.instance.p) + k.c_od * self.x_od[n, k]) for k in
                                                          self.instance.commodities),
                name=str(n)
            )

    def set_up(self):
        self.set_obj()
        self.set_constraint_1()

    def solve(self, path_costs):
        if not self.first_run:
            for n in range(self.n_particles):
                self.m.remove(self.m.getConstrByName(str(n)))
            self.first_run = False
        self.set_particle_constraint(path_costs)
        self.m.optimize()
        return np.array([self.particle_minimum_cost[n].x for n in range(self.n_particles)])

    def print_stuff(self):

        for k in self.instance.commodities:
            found = False
            for p in self.instance.p:
                if self.x[p, k].x > 0.9:
                    print(k, p, k.c_p[p] + self.instance.npp.edges[p]["weight"])
                    found = True
            if not found:
                print(k, 'c_od', k.c_od)
