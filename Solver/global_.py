import numpy as np

from Instance.instance import Instance
from gurobipy import Model, GRB, quicksum, Env

class GlobalSolver:

    def __init__(self, instance: Instance):
        self.instance = instance

        self.m = Model('CVRP')
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE
        self.x = None
        self.locations = range(instance.n_locations)
        self.tolls = range(instance.n_tolls)

        self.c_od = {(i, j): instance.npp[i][j] for i in instance.locations for j in instance.locations if i != j}
        self.c = {(k, p): instance.npp[k][p] for k in instance.locations for p in instance.tolls}
        self.n_eta = 0
        self.p = self.m.addVars([(k, p) for k in self.locations for p in self.tolls],
                                vtype=GRB.BINARY)

    def set_constraints(self):
        pass
