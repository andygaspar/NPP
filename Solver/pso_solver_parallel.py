import numpy as np

from Instance.instance import Instance
from PSO.swarm import Swarm
from Solver.lower_aggregated import LowerSolverAggregated
from Solver.lower_aggregated_2 import LowerSolverAggregated2


class PsoSolver:

    def __init__(self, npp: Instance, init_array, path_costs, n_particles, n_tolls, n_iterations):
        self.best = None
        self.npp = npp
        self.n_iterations = n_iterations
        self.path_costs = path_costs
        self.lower_solver = LowerSolverAggregated2(npp, n_particles)
        self.lower_solver.set_up()
        self.swarm = Swarm(init_array, self.path_costs, npp.N_p.values(), n_particles, npp.n_paths, n_iterations)

    def run(self):
        run_best = 0
        personal_run_results = 0
        for iteration in range(self.n_iterations):
            personal_run_results = self.lower_solver.solve(self.path_costs)
            run_best = max(personal_run_results)
            if run_best > self.swarm.get_best():
                # print(self.swarm.get_best())
                self.swarm.update_best(np.argmax(personal_run_results), run_best)
                self.best = run_best
                # print(self.swarm.get_best())
            # print(self.path_costs)
            print(self.best, '   variance',
                  np.var(self.path_costs.reshape((self.lower_solver.n_particles, self.lower_solver.n_paths)), axis=0))

            self.path_costs = self.swarm.update_swarm(iteration, personal_run_results)

        # print(self.npp.N_p.values())
        # n = np.argmax(personal_run_results)
        self.path_costs = self.path_costs.reshape((self.lower_solver.n_particles, self.lower_solver.n_paths))
        # print(self.path_costs)
        print("final ", self.best)
        # for k in self.npp.commodities:
        #     print(k.cost_free, self.lower_solver.x_od[n, k].x,
        #           [(k.transfer_cost[p] + self.path_costs[n, self.lower_solver.path_dict[p]]) *
        #            self.lower_solver.x[n, p, k].x for p in self.npp.toll_paths])

    def print_swarm(self):
        self.swarm.print_swarm()