import copy
from typing import List

import networkx as nx
import numpy as np
from Arc.ArcInstance.arc_commodity import ArcCommodity, ArcToll, Arc


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
        self.commodities: List[ArcCommodity] = []
        self.tolls: List[ArcToll] = []
        self.free: List[Arc] = []
        self.adj = None

    def get_adj(self):
        return nx.to_numpy_array(self.npp)

    def get_mats(self):
        adj = copy.deepcopy(self.get_adj())
        prices = np.zeros_like(adj)
        for t in self.tolls:
            prices[t.idx[0], t.idx[1]] = adj[t.idx[0], t.idx[1]]

        return adj, prices

    def get_bool_mats(self):
        path_prices = np.zeros_like(self.adj, dtype=bool)
        for edge in self.toll_arcs:
            path_prices[edge[0], edge[1]] = True
            path_prices[edge[1], edge[0]] = True
        return self.adj.astype(bool), path_prices

    @staticmethod
    def min_dist_with_profit(dist, profit, visited, tol):
        min_val = 100000
        min_index = 0
        max_profit = 0
        for i in range(dist.shape[0]):
            if not visited[i] and dist[i] <= min_val + tol:
                if dist[i] < min_val - tol:
                    min_val = dist[i]
                    min_index = i
                    max_profit = profit[i]
                elif profit[i] > max_profit:
                    min_val = dist[i]
                    min_index = i
                    max_profit = profit[i]

        return min_index

    def dijkstra(self, adj, prices, commodity: ArcCommodity, tol=1e-9):
        MAX_DIST = 1000000
        dist = np.ones(adj.shape[0]) * MAX_DIST
        visited = np.zeros_like(dist, dtype=bool)
        profit = np.zeros_like(dist)

        dist[commodity.origin] = 0

        for i in range(adj.shape[0] - 1):
            idx = self.min_dist_with_profit(dist, profit, visited, tol)
            visited[idx] = True
            for j in range(adj.shape[0]):
                if not visited[j] and adj[idx, j] > 0 and dist[idx] != MAX_DIST and dist[idx] + adj[idx, j] <= dist[j] + tol:
                    if dist[idx] + adj[idx, j] < dist[j] - tol:
                        dist[j] = dist[idx] + adj[idx, j]
                        profit[j] = profit[idx] + prices[idx, j] * commodity.n_users
                    elif profit[idx] + prices[idx, j] * commodity.n_users > profit[j]:
                        dist[j] = dist[idx] + adj[idx, j]
                        profit[j] = profit[idx] + prices[idx, j] * commodity.n_users

        return dist, visited, profit

    def compute_obj(self, adj, prices, tol=1e-9):
        obj = 0
        for commodity in self.commodities:
            _, _, profit = self.dijkstra(adj, prices, commodity, tol=tol)
            obj += profit[commodity.destination]
        return obj

    def compute_obj_from_price(self, prices, tol=1e-9):
        adj = self.adj + prices
        return self.compute_obj(adj, prices, tol=tol)

    @staticmethod
    def min_dist(dist, visited, tol):
        min_val = 100000
        min_index = 0
        for i in range(dist.shape[0]):
            if not visited[i] and dist[i] <= min_val + tol:
                if dist[i] < min_val - tol:
                    min_val = dist[i]
                    min_index = i

        return min_index

    def regular_dijkstra(self, adj, node, tol=1e-9):
        MAX_DIST = 1000000
        dist = np.ones(adj.shape[0]) * MAX_DIST
        visited = np.zeros_like(dist, dtype=bool)
        dist[node] = 0

        for i in range(adj.shape[0] - 1):
            idx = self.min_dist(dist, visited, tol)
            visited[idx] = True
            for j in range(adj.shape[0]):
                if not visited[j] and adj[idx, j] > 0 and dist[idx] != MAX_DIST and dist[idx] + adj[idx, j] <= dist[j] + tol:
                    if dist[idx] + adj[idx, j] < dist[j] - tol:
                        dist[j] = dist[idx] + adj[idx, j]

        return dist
