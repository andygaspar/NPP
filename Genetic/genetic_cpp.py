import copy
import ctypes
import multiprocessing
import os

import numpy as np
from numpy.ctypeslib import ndpointer


class GeneticOperators:

    def __init__(self, upper_bounds, commodities_tax_free: np.array, n_users: np.array, transfer_costs: np.array,
                 n_commodities, pop_size, offs_size, n_paths, mutation_rate, recombination_size, n_threads=None):
        num_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
        # print('num_threads', num_threads)
        self.stats = None
        self.n_paths = n_paths
        self.offs_size = offs_size
        self.lib = ctypes.CDLL('PSO/bridge.so')

        self.lib.Genetic_.argtypes = [ctypes.POINTER(ctypes.c_double),
                                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_short),
                                      ctypes.POINTER(ctypes.c_double),
                                      ctypes.c_short, ctypes.c_short, ctypes.c_short,
                                      ctypes.c_short, ctypes.c_double, ctypes.c_short, ctypes.c_short]

        self.lib.Genetic_.restype = ctypes.c_void_p

        self.lib.generation_.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]

        self.lib.eval_parallel_.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]

        n_usr = np.array(n_users, dtype=np.short)

        self.genetic = self.lib.Genetic_(upper_bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         commodities_tax_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         n_usr.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
                                         transfer_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         ctypes.c_short(n_commodities),
                                         ctypes.c_short(pop_size), ctypes.c_short(offs_size), ctypes.c_short(n_paths),
                                         ctypes.c_double(mutation_rate), ctypes.c_short(recombination_size),
                                         ctypes.c_short(num_threads))

    def generation(self, population):
        self.lib.generation_.restype = ndpointer(dtype=ctypes.c_double, shape=(self.offs_size, self.n_paths))
        res = self.lib.generation_(ctypes.c_void_p(self.genetic),
                                   population.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return res

    def eval_fitness(self, population):
        self.lib.eval_parallel_.restype = ndpointer(dtype=ctypes.c_double, shape=population.shape[0])
        return self.lib.eval_parallel_(ctypes.c_void_p(self.genetic),
                                       population.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))


class GeneticCpp:
    def __init__(self, upper_bounds, commodities_tax_free: np.array, n_users: np.array, transfer_costs: np.array,
                 n_commodities, n_paths,
                 pop_size, offs_size, mutation_rate, recombination_size,
                 pso_size, pso_selection_size, pso_every, pso_iterations, pso_final_iterations, pso_no_update_limit,
                 verbose, n_threads=None, seed=None):
        num_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
        # print('num_threads', num_threads)
        self.seed = -1 if seed is None else seed
        self.stats = None
        self.n_paths = n_paths
        self.offs_size = offs_size
        self.lib = ctypes.CDLL('PSO/bridge.so')

        self.lib.Genetic2_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_short, ctypes.c_short,
                                       ctypes.c_short, ctypes.c_short, ctypes.c_double, ctypes.c_short,
                                       ctypes.c_short, ctypes.c_short, ctypes.c_short, ctypes.c_short, ctypes.c_short,
                                       ctypes.c_short,
                                       ctypes.c_short, ctypes.c_bool, ctypes.c_short]
        self.lib.Genetic2_.restype = ctypes.c_void_p

        self.lib.destroy.argtypes = [ctypes.c_void_p]

        self.lib.get_gen_best_val_.argtypes = [ctypes.c_void_p]
        self.lib.get_gen_best_val_.restype = ctypes.c_double

        self.lib.get_population_.argtypes = [ctypes.c_void_p]
        self.lib.get_population_.restype = ndpointer(dtype=ctypes.c_double, shape=(pop_size, self.n_paths))

        self.lib.get_vals_.argtypes = [ctypes.c_void_p]
        self.lib.get_vals_.restype = ndpointer(dtype=ctypes.c_double, shape=(self.n_paths,))

        self.lib.run_genetic_.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]

        n_usr = np.array(n_users, dtype=np.int32)

        self.genetic = self.lib.Genetic2_(upper_bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                          commodities_tax_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                          n_usr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                          transfer_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                          ctypes.c_short(n_commodities), ctypes.c_short(n_paths),
                                          ctypes.c_short(pop_size), ctypes.c_short(offs_size),
                                          ctypes.c_double(mutation_rate), ctypes.c_short(recombination_size),
                                          ctypes.c_short(pso_size), ctypes.c_short(pso_selection_size),
                                          ctypes.c_short(pso_every), ctypes.c_short(pso_iterations),
                                          ctypes.c_short(pso_final_iterations), ctypes.c_short(pso_no_update_limit),
                                          ctypes.c_short(num_threads), ctypes.c_bool(verbose), ctypes.c_short(seed))

    def __del__(self):
        self.lib.destroy(ctypes.c_void_p(self.genetic))

    def run(self, population, iterations):
        self.lib.run_genetic_(ctypes.c_void_p(self.genetic),
                              population.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(iterations))

        return self.lib.get_gen_best_val_(ctypes.c_void_p(self.genetic))

    def get_results(self):
        a = self.lib.get_population_(ctypes.c_void_p(self.genetic))
        b = self.lib.get_vals_(ctypes.c_void_p(self.genetic))
        return a, b
