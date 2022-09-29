import ctypes
from numpy.ctypeslib import ndpointer


class Swarm:

    def __init__(self, n, n_):

        self.n, self.n_ = n, n_
        self.lib = ctypes.CDLL('PSO/bridge.so')
        self.lib.Swarm_.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.update_.argtypes = [ctypes.c_int]
        self.lib.print_s.argtypes = []
        self.lib.Swarm_.restype = ctypes.c_void_p

        self.swarm = self.lib.Swarm_(ctypes.c_int(self.n), ctypes.c_int(self.n_))

    def update(self, k):
        self.lib.update_(self.swarm, k)

    def print_swarm(self):
        self.lib.print_s(self.swarm)


s = Swarm(10, 2)
s.update(100)
s.print_swarm()