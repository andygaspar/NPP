import copy
import ctypes
import multiprocessing
import os

import numpy as np
from numpy.ctypeslib import ndpointer


class GeneticOperators:

    def __init__(self, upper_bounds, combs, commodities_tax_free: np.array, n_users: np.array, transfer_costs: np.array,
                 n_commodities, pop_size, offs_size, n_paths, mutation_rate, recombination_size):
        num_threads = multiprocessing.cpu_count()
        self.stats = None
        self.n_paths = n_paths
        self.lib = ctypes.CDLL('PSO/bridge.so')

        self.lib.Genetic_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_short),
                                      ctypes.POINTER(ctypes.c_double),
                                      ctypes.c_short, ctypes.c_short, ctypes.c_short,
                                      ctypes.c_short, ctypes.c_double, ctypes.c_short, ctypes.c_short]

        self.lib.Genetic_.restype = ctypes.c_void_p

        self.lib.generate_.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

        self.lib.generation_.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

        self.genetic = self.lib.Genetic_(upper_bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         combs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                         commodities_tax_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         n_users.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
                                         transfer_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         ctypes.c_short(n_commodities),
                                         ctypes.c_short(pop_size), ctypes.c_short(offs_size), ctypes.c_short(n_paths),
                                         ctypes.c_double(mutation_rate), ctypes.c_short(recombination_size), ctypes.c_short(num_threads))
        print(n_users)

    def generate(self, a_parent, b_parent, child, upper_bounds):
        return self.lib.generate_(ctypes.c_void_p(self.genetic),
                                  a_parent.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  b_parent.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  child.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  upper_bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    def generation(self, population, vals):
        self.lib.generation_(ctypes.c_void_p(self.genetic),
                                    population.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                    vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
