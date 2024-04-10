import networkx as nx
import numpy as np


class ArcInstance:

    def __init__(self, n_locations, n_commodities):
        self.n_locations = n_locations
        self.n_commodities = n_commodities
        self.users = []
        self.n_tolls = None
        self.npp = None
        self.N_p = None
        self.toll_arcs_undirected = None
        self.toll_arcs = None
        self.free_arcs = None
        self.commodities = None

    def get_adj(self):
        return nx.to_numpy_array(self.npp)

    @staticmethod
    def min_dist(dist, spt):
        min_val = 1000
        min_index = 0
        for i in range(dist.shape[0]):
            if not spt[i] and dist[i] <= min_val:
                min_val = dist[i]
                min_index = i

        return min_index

    def dijkstra(self, adj, src):
        MAX_DIST = 1000
        dist = np.ones(adj.shape[0]) * MAX_DIST
        spt = np.zeros_like(dist, dtype=bool)
        profit = np.zeros_like(dist)

        dist[src] = 0

        for i in range(adj.shape[0] - 1):
            idx = self.min_dist(dist, spt)
            spt[idx] = True
            for j in range(adj.shape[0]):
                if not spt[j] and adj[idx, j] > 0 and dist[idx] != MAX_DIST and dist[idx] + adj[idx, j] < dist[j]:
                    dist[j] = dist[idx] + adj[idx, j]

        return dist, spt
