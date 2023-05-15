import numpy as np
from scipy.stats import qmc

from Instance.instance import Instance
from PSO.swarm import Swarm
from PSO_.swarm_new import SwarmNew
from Solver.global_ import GlobalSolver
from Solver.lower_aggregated import LowerSolverAggregated
from Solver.lower_aggregated_2 import LowerSolverAggregated2


class PsoSolverNew:

    def __init__(self, npp: Instance, n_particles, n_iterations, no_update_lim, time_limit=None):
        self.best_val = None
        self.best = None
        self.npp = npp
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.lower_solver = LowerSolverAggregated2(npp, n_particles)
        self.lower_solver.set_up()
        self.swarm = SwarmNew(npp.commodities_tax_free, npp.n_users, npp.transfer_costs, npp.upper_bounds,
                           npp.n_commodities, npp.n_toll_paths, n_particles, n_iterations, no_update_lim=no_update_lim)

        self.time_limit = time_limit

    def run(self):

        init_sol = np.random.uniform(0, 1, size=(self.npp.n_toll_paths, self.n_particles))
        vel_init = np.random.uniform(-1, 1, size=(self.npp.n_toll_paths, self.n_particles))/10
        lb = np.zeros(self.npp.n_toll_paths)
        ub = np.ones_like(lb)
        self.swarm.run(init_sol, vel_init, ub, lb)

        self.best, self.best_val = self.swarm.get_best()
        # print(self.npp.upper_bounds)
        print(self.best * self.npp.upper_bounds)
        # print("final ", self.best)

    def random_init(self):
        return np.random.uniform(size=(self.n_particles, self.npp.n_toll_paths)) * self.npp.upper_bounds

    def compute_latin_hypercube_init(self, dimensions):
        init_positions = self.random_init()

        tolls_idx = np.argsort(self.npp.upper_bounds)[::-1]
        u_bounds = self.npp.upper_bounds[tolls_idx]

        l_bounds = np.zeros(dimensions)
        sampler = qmc.LatinHypercube(d=dimensions)
        latin_positions = qmc.scale(sampler.random(n=self.n_particles), l_bounds, u_bounds[:dimensions])

        init_positions[:, tolls_idx[:dimensions]] = latin_positions

        return init_positions