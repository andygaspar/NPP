import copy
import ctypes
import multiprocessing
import os

import numpy as np
from numpy.ctypeslib import ndpointer


class SwarmNew:

    def __init__(self, commodities_tax_free: np.array, n_users: np.array, transfer_costs: np.array,
                 upper_bounds: np.array, n_commodities, n_toll_paths,
                 n_particles, n_iterations, no_update_lim):
        num_threads = multiprocessing.cpu_count()
        self.stats = None
        self.n_tolls = n_toll_paths
        self.lib = ctypes.CDLL('PSO_/bridge.so')

        self.lib.Swarm_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_short, ctypes.c_short, ctypes.c_short, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_short]
        self.lib.Swarm_.restype = ctypes.c_void_p

        self.lib.run_.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_bool,
                                  ctypes.c_bool]

        self.lib.get_best_val_.argtypes = [ctypes.c_void_p]
        self.lib.get_best_val_.restype = ctypes.c_double

        self.lib.get_best_.argtypes = [ctypes.c_void_p]
        self.lib.get_best_.restype = ndpointer(dtype=ctypes.c_double, shape=(n_toll_paths,))

        self.lib.get_stats_len_.argtypes = [ctypes.c_void_p]
        self.lib.get_stats_len_.restype = ctypes.c_int

        self.lib.get_p_means_.argtypes = [ctypes.c_void_p]
        self.lib.get_v_means_.argtypes = [ctypes.c_void_p]
        self.lib.get_p_stds_.argtypes = [ctypes.c_void_p]
        self.lib.get_v_stds_.argtypes = [ctypes.c_void_p]

        n_users = np.array(n_users, dtype=np.intc)

        self.swarm = self.lib.Swarm_(commodities_tax_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     n_users.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     transfer_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     upper_bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     ctypes.c_short(n_commodities), ctypes.c_short(n_toll_paths),
                                     ctypes.c_short(n_particles), ctypes.c_int(n_iterations),
                                     ctypes.c_int(no_update_lim),
                                     ctypes.c_short(num_threads))

    def get_best(self):
        return self.lib.get_best_(ctypes.c_void_p(self.swarm)), self.lib.get_best_val_(ctypes.c_void_p(self.swarm))

    def run(self, init_positions, init_velocity, lb, ub, stats=False, verbose=False):
        self.stats = stats
        self.lib.run_(ctypes.c_void_p(self.swarm),
                      init_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      init_velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      lb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      ub.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      ctypes.c_bool(stats),
                      ctypes.c_bool(verbose))
        return True

    def get_stats(self):
        if self.stats is None or not self.stats:
            print("warning: stat=False in input pso.run, no stats available ")
            return None
        n_stats = self.lib.get_stats_len_(ctypes.c_void_p(self.swarm))

        self.lib.get_p_means_.restype = ndpointer(dtype=ctypes.c_double, shape=(n_stats, self.n_tolls))
        self.lib.get_v_means_.restype = ndpointer(dtype=ctypes.c_double, shape=(n_stats, self.n_tolls))
        self.lib.get_p_stds_.restype = ndpointer(dtype=ctypes.c_double, shape=(n_stats, self.n_tolls))
        self.lib.get_v_stds_.restype = ndpointer(dtype=ctypes.c_double, shape=(n_stats, self.n_tolls))

        p_means = self.lib.get_p_means_(ctypes.c_void_p(self.swarm))
        v_means = self.lib.get_v_means_(ctypes.c_void_p(self.swarm))
        p_stds = self.lib.get_p_stds_(ctypes.c_void_p(self.swarm))
        v_stds = self.lib.get_v_stds_(ctypes.c_void_p(self.swarm))

        return p_means, p_stds, v_means, v_stds
