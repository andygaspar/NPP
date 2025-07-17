import random
import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d


def get_cost(g, toll):
    return nx.dijkstra_path_length(g, toll[0], toll[1], 'cost')


class ArcNewCommodity:
    def __init__(self, origin, destination, n_users):
        self.origin = origin
        self.destination = destination
        self.n_users = n_users
        self.name = str(self.origin) + ' -> ' + str(self.destination)
        self.M_p = {}

    def compute_bounds(self, g_zero: nx.Graph, g_inf: nx.Graph, tolls):
        o, d = self.origin, self.destination
        for toll in tolls:
            c_a = g_inf.edges[toll]['weight']
            t, h = toll
            b1 = g_inf.edges[toll]['gamma_ht'] - c_a
            b2 = get_cost(g_inf, (o, h)) - get_cost(g_zero, (o, t)) - c_a
            b3 = get_cost(g_inf, (o, d)) - get_cost(g_zero, (o, t)) - c_a - get_cost(g_zero, (h, d))
            b4 = get_cost(g_inf, (t, d)) - get_cost(g_zero, (h, d)) - c_a
            self.M_p[toll] = max(0, min(b1, b2, b3, b4))

    def __repr__(self):
        return self.name


class Arc:
    def __init__(self, idx: tuple, c_a: float):
        self.idx = idx
        self.c_a = c_a

    def __repr__(self):
        return str(self.idx)


class ArcToll(Arc):

    def __init__(self, idx: tuple, commodities: List[ArcNewCommodity], c_a: float):
        super().__init__(idx, c_a)
        self.N_p = max([c.M_p[self.idx] for c in commodities])


class ArcNewInstance:

    def __init__(self, n_commodities, toll_proportion, n_nodes, g: nx.Graph):
        self.n_commodities = n_commodities
        self.toll_proportion = toll_proportion
        self.n_nodes = n_nodes
        self.n_tolls = None
        self.commodities: List[ArcNewCommodity] = []
        self.arc_tolls: List[tuple] = []
        self.tolls: List[ArcToll] = []
        self.free: List[Arc] = []
        self.npp = self.set_instance(g)
        self.set_commodities_bound()
        self.set_tolls()
        self.n_tolls *= 2

    def get_adj(self):
        return nx.to_numpy_array(self.npp)

    def set_commodities_bound(self):
        g_inf = self.npp.copy()
        for toll in self.arc_tolls:
            g_inf.edges[toll]['cost'] = 10 ** 5

        for toll in self.arc_tolls:
            g_inf.edges[toll]['gamma_ht'] = get_cost(g_inf, toll)

        for commodity in self.commodities:
            commodity.compute_bounds(self.npp, g_inf, self.arc_tolls)

    def set_tolls(self):
        for edge in self.npp.edges:
            if self.npp.edges[edge]['toll']:
                self.tolls.append(ArcToll(edge, self.commodities, self.npp.edges[edge]['weight']))
            else:
                self.free.append(Arc(edge, self.npp.edges[edge]['weight']))

    def set_instance(self, g):

        edges = list(g.edges)
        remained_edges = list(g.edges)
        for _ in range(int(len(edges) * 0.2)):
            e = remained_edges.pop(random.choice(range(len(remained_edges))))
            g.edges[e]['weight'] = 35
            g.edges[e]['toll'] = False

        for e in remained_edges:
            g.edges[e]['weight'] = np.random.uniform(5, 35)
            g.edges[e]['toll'] = False

        for e in edges:
            g.edges[e]['counter'] = 0
            g.edges[e]['color'] = 'blue'
            g.edges[e]['price'] = 0

        shortest_paths = []

        for com in range(self.n_commodities):
            o, d = random.choice(list(g.nodes)), None
            found_destination = False
            while not found_destination:
                d = random.choice(list(g.nodes))
                if o != d:
                    found_destination = True
            self.commodities.append(ArcNewCommodity(o, d, random.choice(range(1, 5))))
            node_path = nx.dijkstra_path(g, o, d)
            edge_path = []
            for i in range(len(node_path) - 1):
                edge_path.append((node_path[i], node_path[i + 1]))
            shortest_paths.append(edge_path)

        for path in shortest_paths:
            for edge in path:
                g.edges[edge]['counter'] += 1

        idxs = np.argsort([g.edges[edge]['counter'] for edge in g.edges])[::-1]
        edges = list(g.edges)
        remained_edges = list(g.edges)

        self.n_tolls = int(len(g.edges) * self.toll_proportion)

        for i in range(int(self.n_tolls * 2 / 3)):
            g.edges[edges[idxs[i]]]['toll'] = True
            self.arc_tolls.append(edges[idxs[i]])
            self.arc_tolls.append((edges[idxs[i]][1], edges[idxs[i]][0]))
            g.edges[edges[idxs[i]]]['weight'] /= 2
            g.edges[edges[idxs[i]]]['color'] = 'red'
            remained_edges.remove(edges[idxs[i]])

        for _ in range(self.n_tolls - int(self.n_tolls * 2 / 3)):  # one third random (computed this way to avoid rounding issues)
            e = remained_edges.pop(random.choice(range(len(remained_edges))))
            self.arc_tolls.append(e)
            self.arc_tolls.append((e[1], e[0]))
            g.edges[e]['toll'] = True
            g.edges[e]['weight'] /= 2
            g.edges[e]['color'] = 'red'

        for e in edges:
            g.edges[e]['cost'] = g.edges[e]['weight']
        g = nx.to_directed(g)

        return g

    def compute_obj(self, adj, prices, tol=1e-9):
        obj = 0
        for commodity in self.commodities:
            _, _, profit = self.dijkstra(adj, prices, commodity, tol=tol)
            obj += profit[commodity.destination]
        return obj

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

    def dijkstra(self, adj, prices, commodity: ArcNewCommodity, tol=1e-9):
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

    def draw_instance(self, pos, show_cost=False):
        plt.rcParams['figure.figsize'] = (12, 8)
        edge_color = [self.npp.edges[e]['color'] for e in self.npp.edges]
        nx.draw(self.npp, pos=pos, with_labels=True, edge_color=edge_color)
        if show_cost:
            labels = {e: "{:.2f}".format(self.npp.edges[e]['weight']) for e in self.npp.edges}
            nx.draw_networkx_edge_labels(self.npp, pos=pos, edge_labels=labels)
        plt.show()


class GridInstance(ArcNewInstance):

    def __init__(self, n_commodities, toll_proportion, n_nodes):
        n = int(np.sqrt(n_nodes))
        n_nodes = n ** 2
        g = nx.grid_2d_graph(n, n)
        mapping = {}
        for i, node in enumerate(g.nodes):
            g.nodes[node]['pos'] = node
            mapping[node] = i
        g = nx.relabel_nodes(g, mapping, copy=False)
        super().__init__(n_commodities, toll_proportion, n_nodes, g)

    def draw(self, show_cost=False):
        pos = {node: self.npp.nodes[node]['pos'] for node in self.npp.nodes()}
        self.draw_instance(pos, show_cost=show_cost)


class DelaunayInstance(ArcNewInstance):

    def __init__(self, n_commodities, toll_proportion, n_nodes):
        self.points = np.random.uniform(0, 1, size=(n_nodes, 2))
        tri = Delaunay(self.points)
        neighbours = tri.vertex_neighbor_vertices
        edges = []
        for i in range(n_nodes):
            for k in neighbours[1][neighbours[0][i]: neighbours[0][i + 1]]:
                if (i, k) not in edges and (k, i) not in edges:
                    edges.append((i, k))

        g = nx.Graph()
        g.add_edges_from(edges)
        super().__init__(n_commodities, toll_proportion, n_nodes, g)

    def draw(self, show_cost=False):
        self.draw_instance(self.points, show_cost=show_cost)


class VoronoiNewInstance(ArcNewInstance):
    def __init__(self, n_commodities, toll_proportion, n_nodes):
        points = np.random.uniform(0, 1, size=(n_nodes, 2))
        vor = Voronoi(points)

        valid_vertex = lambda v: 0 <= v[0] <= 1 and 0 <= v[1] <= 1

        valid_vertices = [i for i in range(vor.vertices.shape[0]) if valid_vertex(vor.vertices[i])]
        new_index = dict(zip(valid_vertices, range(len(valid_vertices))))

        self.vertices = [tuple(vor.vertices[i]) for i in valid_vertices]
        edges = [(new_index[e[0]], new_index[e[1]]) for e in vor.ridge_vertices if e[0] in new_index and e[1] in new_index]

        g = nx.Graph()
        g.add_edges_from(edges)
        super().__init__(n_commodities, toll_proportion, n_nodes, g)

    def draw(self, show_cost=False):
        self.draw_instance(self.vertices, show_cost=show_cost)


