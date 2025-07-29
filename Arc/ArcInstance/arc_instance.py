import os
import random
import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
from gurobipy import Model, GRB

def get_cost(g, toll):
    return nx.dijkstra_path_length(g, toll[0], toll[1], 'cost')


class ArcCommodity:
    def __init__(self, origin, destination, n_users):
        self.origin = origin
        self.destination = destination
        self.n_users = n_users
        self.name = str(self.origin) + ' -> ' + str(self.destination)
        self.M_p = {}

        self.solution_edges = None
        self.solution_path = None

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

    def __init__(self, idx: tuple, commodities: List[ArcCommodity], c_a: float):
        super().__init__(idx, c_a)
        self.N_p = max([c.M_p[self.idx] for c in commodities])


class ArcInstance:

    def __init__(self, n_commodities, toll_proportion, n_nodes, g: nx.Graph, preset_graph=False):
        self.n_commodities = n_commodities
        self.toll_proportion = toll_proportion
        self.n_nodes = n_nodes
        self.n_tolls = 0
        self.commodities: List[ArcCommodity] = []
        self.arc_tolls: List[tuple] = []
        self.tolls: List[ArcToll] = []
        self.free: List[Arc] = []
        if not preset_graph:
            self.g = self.set_instance(g)
        else:
            self.g = g
            self.load_graph(self.g)
        self.adj = nx.to_numpy_array(self.g, range(self.n_nodes))
        self.edges = list(self.g.edges())
        self.n_edges = len(self.edges)
        self.set_commodities_bound()
        self.set_tolls()
        self.n_free = self.n_edges - self.n_tolls
        self.arc_free: List[tuple] = [e for e in self.edges if e not in self.arc_tolls]
        self.arc_tolls = [e for e in self.edges if e in self.arc_tolls]

    def load_graph(self, g: nx.Graph):
        for k in g._commodities:
            self.commodities.append(ArcCommodity(*k))

        for edge in g.edges():
            if g.edges[edge]['toll']:
                self.arc_tolls.append(edge)
        self.n_tolls = len(self.arc_tolls)

    def get_adj(self):
        return nx.to_numpy_array(self.g, nodelist=range(self.n_nodes))

    def set_commodities_bound(self):
        g_inf = self.g.copy()
        for toll in self.arc_tolls:
            g_inf.edges[toll]['cost'] = 10 ** 5

        for toll in self.arc_tolls:
            g_inf.edges[toll]['gamma_ht'] = get_cost(g_inf, toll)

        for commodity in self.commodities:
            commodity.compute_bounds(self.g, g_inf, self.arc_tolls)

    def set_tolls(self):
        for edge in self.g.edges:
            if self.g.edges[edge]['toll']:
                self.tolls.append(ArcToll(edge, self.commodities, self.g.edges[edge]['weight']))
            else:
                self.free.append(Arc(edge, self.g.edges[edge]['weight']))

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
            self.commodities.append(ArcCommodity(o, d, random.choice(range(1, 5))))
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
        remained_edges = [edges[idxs[i]] for i in idxs]

        self.n_tolls = int(len(g.edges) * self.toll_proportion)

        added_tolls = 0
        while added_tolls < int(self.n_tolls * 2 / 3):
            feasible = self.is_feasible(remained_edges[1:])
            if feasible:
                g.edges[remained_edges[0]]['toll'] = True
                self.arc_tolls.append(remained_edges[0])
                self.arc_tolls.append((remained_edges[0][1], remained_edges[0][0]))
                g.edges[remained_edges[0]]['weight'] /= 2
                g.edges[remained_edges[0]]['color'] = 'red'
                added_tolls += 1
            remained_edges.remove(remained_edges[0])

        while added_tolls < self.n_tolls:
            e_idx = random.choice(range(len(remained_edges)))
            feasible = self.is_feasible([e for i, e in enumerate(remained_edges) if i != e_idx])
            if feasible:
                e = remained_edges.pop(e_idx)
                self.arc_tolls.append(e)
                self.arc_tolls.append((e[1], e[0]))
                g.edges[e]['toll'] = True
                g.edges[e]['weight'] /= 2
                g.edges[e]['color'] = 'red'
                added_tolls += 1

        for e in edges:
            g.edges[e]['cost'] = g.edges[e]['weight']
        g = nx.to_directed(g)
        self.n_tolls *= 2

        return g

    def compute_obj(self, adj_sol, prices, tol=1e-9):
        obj = 0
        for commodity in self.commodities:
            _, _, profit, path = self.dijkstra(adj_sol, prices, commodity, tol=tol)
            obj += profit[commodity.destination]
        return obj

    def assign_paths(self, adj_sol, prices, tol=1e-9):
        for commodity in self.commodities:
            _, _, profit, path = self.dijkstra(adj_sol, prices, commodity, tol=tol)
            commodity.solution_path = path
            commodity.solution_edges = [(path[0], path[1])]
            for i in range(1, len(path) - 1):
                commodity.solution_edges.append((path[i], path[i + 1]))

    def is_feasible(self, edges):
        edges_all = edges + [(e[1], e[0]) for e in edges]
        A = np.zeros((self.n_nodes, len(edges_all)))
        for i, e in enumerate(edges_all):
            A[e[0], i] = -1
            A[e[1], i] = 1
        b = np.zeros((self.n_commodities, self.n_nodes))
        for i, k in enumerate(self.commodities):
            b[i, k.origin] = -1
            b[i, k.destination] = 1

        m = Model()
        m.setParam('OutputFlag', 0)
        x = m.addMVar((self.n_commodities, len(edges_all)))

        for k in range(self.n_commodities):
            m.addConstr(A @ x[k] == b[k], name='feasible')
        m.optimize()
        return m.status == GRB.Status.OPTIMAL


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
        prev = np.zeros(adj.shape[0], dtype=int)
        for i in range(adj.shape[0] - 1):
            idx = self.min_dist_with_profit(dist, profit, visited, tol)
            visited[idx] = True
            for j in range(adj.shape[0]):
                if not visited[j] and adj[idx, j] > 0 and dist[idx] != MAX_DIST and dist[idx] + adj[idx, j] <= dist[j] + tol:
                    if dist[idx] + adj[idx, j] < dist[j] - tol:
                        dist[j] = dist[idx] + adj[idx, j]
                        profit[j] = profit[idx] + prices[idx, j] * commodity.n_users
                        prev[j] = idx
                    elif profit[idx] + prices[idx, j] * commodity.n_users > profit[j]:
                        dist[j] = dist[idx] + adj[idx, j]
                        profit[j] = profit[idx] + prices[idx, j] * commodity.n_users
                        prev[j] = idx
        path = [commodity.destination]
        while path[-1] != commodity.origin:
            path.append(prev[path[-1]])
        path = path[::-1]
        return dist, visited, profit, path

    def draw(self, show_cost=False):
        plt.rcParams['figure.figsize'] = (12, 8)
        pos = {node: self.g.nodes[node]['pos'] for node in self.g.nodes()}
        edge_color = [self.g.edges[e]['color'] for e in self.g.edges]
        nx.draw(self.g, pos=pos, with_labels=True, edge_color=edge_color)
        if show_cost:
            labels = {e: "{:.2f}".format(self.g.edges[e]['weight']) for e in self.g.edges}
            nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=labels)
        plt.show()

    def save_instance(self, filename):
        commodities = [(k.origin, k.destination, k.n_users) for k in self.commodities]
        self.g._commodities = commodities
        self.g._toll_proportion = self.toll_proportion
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.g, f)

    def save_problem(self, pb_name):
        folder = 'Arc/Arc_GA/Problems/' + pb_name
        os.mkdir(folder)
        np.savetxt(folder + '/ub.csv', np.array([p.N_p for p in self.tolls]), fmt='%.18f')
        np.savetxt(folder + '/lb.csv', np.zeros(self.n_tolls), fmt='%.18f')
        np.savetxt(folder + '/adj.csv', self.adj, fmt='%.18f')
        np.savetxt(folder + '/adj_size.csv', np.array([self.adj.shape[0]]), fmt='%d')
        np.savetxt(folder + '/toll_idxs.csv', np.array([p.idx for p in self.tolls]).T.flatten(), fmt='%d')
        np.savetxt(folder + '/n_users.csv', np.array([commodity.n_users for commodity in self.commodities]), fmt='%d')
        np.savetxt(folder + '/origins.csv', np.array([commodity.origin for commodity in self.commodities]), fmt='%d')
        np.savetxt(folder + '/destinations.csv', np.array([commodity.destination for commodity in self.commodities]), fmt='%d')
        np.savetxt(folder + '/origins.csv', np.array([commodity.origin for commodity in self.commodities]), fmt='%d')
        np.savetxt(folder + '/n_com.csv', np.array([self.n_commodities]), fmt='%d')
        np.savetxt(folder + '/n_tolls.csv', np.array([self.n_tolls]), fmt='%d')


class GridInstance(ArcInstance):

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


class DelaunayInstance(ArcInstance):

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
        for node in g.nodes:
            g.nodes[node]['pos'] = self.points[node]
        super().__init__(n_commodities, toll_proportion, n_nodes, g)


class VoronoiNewInstance(ArcInstance):
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
        for node in g.nodes:
            g.nodes[node]['pos'] = self.vertices[node]
        super().__init__(n_commodities, toll_proportion, n_nodes, g)


def instance_from_graph(g: nx.Graph):
    return ArcInstance(len(g._commodites), g._toll_proportion, len(g.nodes), g, preset_graph=True)


