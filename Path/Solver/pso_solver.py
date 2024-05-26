import time

import numpy as np
from scipy.stats import qmc

from Instance.instance import Instance
from Path.CPP.PSO.swarm import Swarm


class PsoSolver:

    def __init__(self, npp: Instance, n_particles, n_iterations, no_update_lim, time_limit=None):
        self.time = None
        self.best_val = None
        self.best = None
        self.npp = npp
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.final_iterations = None
        self.swarm = Swarm(npp.commodities_tax_free, npp.n_users, npp.transfer_costs, npp.upper_bounds,
                           npp.n_commodities, npp.n_paths, n_particles, n_iterations,
                           no_update_lim=no_update_lim)

        self.time_limit = time_limit

    def run(self, init_pos=None, speed_range=(-5, 5), stats=False, verbose=False, seed=None):
        self.time = time.time()
        if init_pos is None:
            init_pos = np.random.uniform(0, 1, size=(self.npp.n_paths, self.n_particles))
        vel_init = np.random.uniform(speed_range[0], speed_range[1], size=(self.npp.n_paths, self.n_particles))
        seed = -1 if seed is None else seed
        self.swarm.run(init_pos, vel_init, stats, verbose, seed)
        self.time = time.time() - self.time

        self.best, self.best_val = self.swarm.get_best()
        self.final_iterations = self.swarm.get_iterations()

    def get_particles(self):
        return self.swarm.get_particles()

    def get_best_n_particles(self, n_particles):
        particles = self.swarm.get_particles()
        values = self.swarm.get_values()
        idx = np.argsort(values)
        return particles[idx][-n_particles:]

    def get_stats(self):
        return self.swarm.get_stats()

    def random_init(self):
        return np.random.uniform(size=(self.n_particles, self.npp.n_paths)) * self.npp.upper_bounds

    def compute_latin_hypercube_init(self, dimensions):
        init_positions = self.random_init()

        tolls_idx = np.argsort(self.npp.upper_bounds)[::-1]
        u_bounds = self.npp.upper_bounds[tolls_idx]

        l_bounds = np.zeros(dimensions)
        sampler = qmc.LatinHypercube(d=dimensions)
        latin_positions = qmc.scale(sampler.random(n=self.n_particles), l_bounds, u_bounds[:dimensions])

        init_positions[:, tolls_idx[:dimensions]] = latin_positions

        return init_positions
