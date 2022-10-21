import numpy as np

from Instance.instance import Instance
from PSO.swarm import Swarm
from Solver.lower_aggregated import LowerSolverAggregated
from Solver.lower_aggregated_2 import LowerSolverAggregated2


class PsoSolver:

    def __init__(self, npp: Instance,  path_costs, n_particles, n_iterations):
        self.best_val = None
        self.best = None
        self.npp = npp
        self.n_iterations = n_iterations
        self.path_costs = path_costs
        self.lower_solver = LowerSolverAggregated2(npp, n_particles)
        self.lower_solver.set_up()
        self.swarm = Swarm(npp.commodities_tax_free, npp.n_users, npp.transfer_costs, npp.upper_bounds,
                           npp.n_commodities, npp.n_tolls, n_particles, n_iterations)

    def run(self):
        run_best = 0
        personal_run_results = 0
        self.swarm.run()

        self.best, self.best_val = self.swarm.get_best()
        print("final ", self.best)

    def print_swarm(self):
        self.swarm.print_swarm()