import numpy as np

from Instance.instance import Instance, Commodity
from gurobipy import Model, GRB, quicksum, Env


class LowerSolverAggregated:

    def __init__(self, instance: Instance, n_particles):
        self.n_particles = n_particles
        self.instance = instance
        self.m = Model('CVRP')
        self.m.setParam("OutputFlag", 0)
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE
        self.n_paths = len(self.instance.p)
        self.n_commodities = len(self.instance.commodities)
        self.x = self.m.addMVar((self.n_particles, self.n_paths, self.n_commodities), vtype=GRB.BINARY)
        self.x_od = self.m.addMVar((self.n_particles, self.n_commodities), vtype=GRB.BINARY)
        self.particle_minimum_cost = self.m.addMVar(self.n_particles, vtype=GRB.CONTINUOUS)
        self.first_run = True

    def set_obj(self):
        self.m.setObjective(
            quicksum(
                self.particle_minimum_cost[n] for n in range(self.n_particles)
            )
        )

    def set_constraint_1(self):

        for k in self.instance.commodities:
            self.m.addConstr(
                quicksum(self.x[p, k] for p in self.instance.p) + self.x_od[k] == 1
            )

    def set_constraint_particle(self, path_costs):

        for n in range(self.n_particles):
            self.m.addConstr(self.particle_minimum_cost[n] == quicksum(
                    k.n_users * (quicksum((k.c_p[path] + path_costs[n, path]) * self.x[path, k] for path in
                                          self.instance.p) + k.c_od * self.x_od[k]) for k in self.instance.commodities),
                             name='particle'
        )

    def set_up(self):
        self.set_obj()
        self.set_constraint_1()

    def solve(self, path_costs):
        if not self.first_run:
            self.m.remove(self.m.getConstrByName("particle"))
            self.first_run = False
        self.set_constraint_particle(path_costs)
        self.m.optimize()

        for k in self.instance.commodities:
            found = False
            for p in self.instance.p:
                if self.x[p, k].x > 0.9:
                    print(k, p, k.c_p[p] + self.instance.npp.edges[p]["weight"])
                    found = True
            if not found:
                print(k, 'c_od', k.c_od)
