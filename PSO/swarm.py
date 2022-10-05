import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer


class Swarm:

    def __init__(self, cost_array: np.array, n, n_):

        self.n, self.n_ = n, n_
        self.lib = ctypes.CDLL('PSO/bridge.so')
        self.lib.Swarm_.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
        self.lib.update_.argtypes = [ctypes.c_int]
        self.lib.test_io.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        self.lib.print_s.argtypes = []
        self.lib.Swarm_.restype = ctypes.c_void_p

        self.swarm = self.lib.Swarm_(cost_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                               ctypes.c_int(self.n), ctypes.c_int(self.n_))

    def test_io(self, n):
        self.lib.test_io.restype = ndpointer(dtype=ctypes.c_double, shape=(n,))
        input_vect = np.random.uniform(size=n)
        output_vect = self.lib.test_io(input_vect.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)
        print(output_vect)

    def update(self, k):
        self.lib.update_(self.swarm, k)

    def print_swarm(self):
        self.lib.print_s(self.swarm)

os.system("PSO/install.sh")
n_particles = 5
n_nodes = 2
cost_array = np.random.uniform(0, 1, 10)
s = Swarm(cost_array, 10, 2)
s.update(100)
s.print_swarm()

s.test_io(10)

