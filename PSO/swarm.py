import ctypes
import multiprocessing
import os

import numpy as np
from numpy.ctypeslib import ndpointer


class Swarm:

    def __init__(self, commodities_tax_free: np.array, n_users: np.array, transfer_costs: np.array,
                 upper_bounds: np.array,
                 n_commodities, n_toll_paths, n_particles, n_iterations):
        num_threads = multiprocessing.cpu_count()
        self.lib = ctypes.CDLL('PSO/bridge.so')
        self.lib.Swarm_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_short, ctypes.c_short, ctypes.c_short, ctypes.c_int, ctypes.c_short, ctypes.c_short, ctypes.c_short, ctypes.c_short, ctypes.c_int]
        self.lib.Swarm_.restype = ctypes.c_void_p

        self.lib.run_.argtypes = [ctypes.c_void_p, ctypes.c_short]

        # self.lib.get_best_val_.argtypes = [ctypes.c_void_p]
        # self.lib.get_best_val_.restype = ctypes.c_double

        # self.lib.get_best_.argtypes = [ctypes.c_void_p]
        # self.lib.get_best_.restype = ndpointer(dtype=ctypes.c_double, shape=(n_toll_paths,))

        self.lib.set_init_sols_.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                            ctypes.c_int]

        n_users = np.array(n_users, dtype=np.intc)

        lower_bounds = np.zeros_like(upper_bounds)
        N_PARTS = n_particles//5
        n_cut = 3
        N_DIV = 4
        n_u_l = 500

        self.swarm = self.lib.Swarm_(commodities_tax_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     n_users.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     transfer_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     upper_bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     lower_bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     ctypes.c_short(n_commodities), ctypes.c_short(n_toll_paths),
                                     ctypes.c_short(n_particles), ctypes.c_int(n_iterations),
                                     ctypes.c_short(num_threads), ctypes.c_short(N_PARTS), ctypes.c_short(n_cut), ctypes.c_short(N_DIV), ctypes.c_short(n_u_l))

    def set_init_sols(self, solutions):
        n_solutions = 1
        # solutions = np.ascontiguousarray(solutions, dtype=np.float)
        self.lib.set_init_sols_(ctypes.c_void_p(self.swarm), solutions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                ctypes.c_int(n_solutions))
        print('c')

    def get_best(self):
        return self.lib.get_best_(ctypes.c_void_p(self.swarm)), self.lib.get_best_val_(ctypes.c_void_p(self.swarm))

    def run(self, init_sols=None):
        if init_sols is not None:
            self.set_init_sols(init_sols)
        self.lib.run_(ctypes.c_void_p(self.swarm))
        return True
