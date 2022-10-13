import numpy as np

from Instance.instance import Instance
from PSO.swarm import Swarm
from Solver.lower_aggregated import LowerSolverAggregated


class PsoSolver:

    def __init__(self, npp:Instance, init_array, path_costs, n_particles, n_tolls, n_iterations):
        self.n_iterations = n_iterations
        self.path_costs = path_costs
        self.lower_solver = LowerSolverAggregated(npp, n_particles)
        self.lower_solver.set_up()
        self.swarm = Swarm(init_array, path_costs, npp.N_p.values(), n_particles, n_tolls, n_iterations)

    def run(self):
        for iteration in range(self.n_iterations):
            personal_run_results = self.lower_solver.solve(self.path_costs)
            run_best = max(personal_run_results)
            if run_best > self.swarm.get_best():
                print(self.swarm.get_best())
                self.swarm.update_best(np.argmax(personal_run_results), run_best)
                print(self.swarm.get_best())

            self.swarm.update_swarm(iteration, personal_run_results)

    def print_swarm(self):
        self.swarm.print_swarm()