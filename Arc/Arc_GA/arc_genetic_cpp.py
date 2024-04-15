import copy
import ctypes
import multiprocessing
import os

import numpy as np
from numpy.ctypeslib import ndpointer


class ArcGeneticCpp:
    def __init__(self, upper_bounds, lower_bounds, adj, tolls_idxs, n_users: np.array, origins, destinations,
                 n_commodities, n_tolls,
                 pop_size, offs_size, mutation_rate, recombination_size,
                 verbose, n_threads=None, seed=None):
        num_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
        # print('num_threads', num_threads)
        self.seed = -1 if seed is None else seed
        self.stats = None
        self.n_tolls = n_tolls
        self.offs_size = offs_size
        self.lib = ctypes.CDLL('Arc/Arc_GA/arc_ga_bridge.so')

        self.lib.ArcGenetic_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                         ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                         ctypes.c_short, ctypes.c_short,
                                         ctypes.c_short, ctypes.c_short, ctypes.c_double,
                                         ctypes.c_short,
                                         ctypes.c_short, ctypes.c_bool, ctypes.c_short]
        self.lib.ArcGenetic_.restype = ctypes.c_void_p

        self.lib.destroy.argtypes = [ctypes.c_void_p]

        self.lib.get_gen_best_val_.argtypes = [ctypes.c_void_p]
        self.lib.get_gen_best_val_.restype = ctypes.c_double

        self.lib.get_population_.argtypes = [ctypes.c_void_p]
        self.lib.get_population_.restype = ndpointer(dtype=ctypes.c_double, shape=(pop_size, self.n_tolls))

        self.lib.get_vals_.argtypes = [ctypes.c_void_p]
        self.lib.get_vals_.restype = ndpointer(dtype=ctypes.c_double, shape=(pop_size,))

        self.lib.run_arc_genetic_.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]

        origins = np.array(origins, dtype=np.int32)
        destinations = np.array(destinations, dtype=np.int32)
        n_usr = np.array(n_users, dtype=np.int32)
        tolls_idxs = np.array(tolls_idxs, dtype=np.int32)

        self.arc_genetic = self.lib.ArcGenetic_(upper_bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                lower_bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                adj.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                ctypes.c_int(adj.shape[0]),
                                                tolls_idxs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                n_usr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                origins.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                destinations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                ctypes.c_short(n_commodities), ctypes.c_short(n_tolls),
                                                ctypes.c_short(pop_size), ctypes.c_short(offs_size),
                                                ctypes.c_double(mutation_rate), ctypes.c_short(recombination_size),
                                                ctypes.c_short(num_threads), ctypes.c_bool(verbose), ctypes.c_short(self.seed))

    def __del__(self):
        self.lib.destroy(ctypes.c_void_p(self.arc_genetic))

    def run(self, population, iterations):
        self.lib.run_arc_genetic_(ctypes.c_void_p(self.arc_genetic),
                                  population.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(iterations))

        return self.lib.get_gen_best_val_(ctypes.c_void_p(self.arc_genetic))

    def get_results(self):
        pop = self.lib.get_population_(ctypes.c_void_p(self.arc_genetic))
        vals = self.lib.get_vals_(ctypes.c_void_p(self.arc_genetic))
        return pop, vals
