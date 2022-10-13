import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer


class Swarm:

    def __init__(self, init_norm_array: np.array, cost_array: np.array, scale_factor_array: np.array, n, n_, n_iterations):
        self.n, self.n_, self.n_iterations = n, n_, n_iterations
        self.scale_factor_array = np.array(scale_factor_array)
        self.lib = ctypes.CDLL('PSO/bridge.so')
        self.lib.Swarm_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int]
        # self.lib.update_.argtypes = [ctypes.c_int]
        # self.lib.update_.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        self.lib.update_swarm_.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.lib.test_io.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        self.lib.print_s.argtypes = []
        self.lib.Swarm_.restype = ctypes.c_void_p
        self.lib.get_best_val_.argtypes = [ctypes.c_void_p]
        self.lib.get_best_val_.restype = ctypes.c_double
        self.lib.update_best_.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]

        self.swarm = self.lib.Swarm_(init_norm_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     cost_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     self.scale_factor_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     ctypes.c_int(self.n), ctypes.c_int(self.n_),
                                     self.n_iterations)

    def test_io(self, n):
        self.lib.test_io.restype = ndpointer(dtype=ctypes.c_double, shape=(n,))
        input_vect = np.random.uniform(size=n)
        output_vect = self.lib.test_io(input_vect.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)
        print(output_vect)

    def get_best(self):
        return self.lib.get_best_val_(ctypes.c_void_p(self.swarm))

    def update_best(self, best_particle_idx, new_best_val):
        return self.lib.update_best_(ctypes.c_void_p(self.swarm), best_particle_idx, new_best_val)

    def update_swarm(self, iteration, run_values: np.array):
        self.lib.update_swarm_(ctypes.c_void_p(self.swarm), iteration,
                               run_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    def print_swarm(self):
        self.lib.print_s(self.swarm)
