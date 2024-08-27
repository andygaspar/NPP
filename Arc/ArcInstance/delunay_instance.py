import random
from typing import List
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

import scipy

from Arc.ArcInstance.arc_commodity import ArcCommodity, ArcToll, Arc
from Arc.ArcInstance.arc_instance import ArcInstance


class DelaunayInstance(ArcInstance):
    def __init__(self, n_locations, n_arcs, dim_grid, toll_proportion, n_commodities, costs=(5, 35), nr_users=(1, 5), seed=None):
        # costs = (2, 20)
        super().__init__(n_locations, n_commodities)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.name = 'Delaunay'
        self.toll_proportion = toll_proportion  # {5%, 10%, 15%, 20%}
        self.commodities: List[ArcCommodity] = []
        self.costs = costs
        self.nr_users = nr_users
        self.n_arcs = n_arcs
        self.toll_arcs = []
        self.free_arcs = []
        self.toll_arcs_undirected = []

        n_points = n_locations
        bound_a, bound_b = 0, 5
        points = np.random.uniform(bound_a, bound_b, (n_points, 2))
        tri = scipy.spatial.Delaunay(points)

        graph = nx.Graph()
        paths = []
        # pos = {node: [node] for i, node in enumerate(graph.nodes)}

        graph.add_nodes_from([(i, {'pos': tri.points[i]}) for i in range(n_points)])
        for sim in tri.simplices:
            paths.append(np.append(sim, sim[0]))
        for path in paths:
            nx.add_path(graph, path)


        # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
        # plt.plot(points[:, 0], points[:, 1], 'o')
        # plt.show()
        #
        nx.draw(graph, pos={i: tri.points[i] for i in range(n_points)}, with_labels=graph.nodes)
        # plt.show()

        #
        # nx.draw(graph, with_labels=graph.nodes)
        # plt.show()

        # graph = nx.convert_node_labels_to_integers(graph)

        self.n_arcs = len(graph.edges)

        for edge in graph.edges:
            graph.edges[edge]['weight'] = np.linalg.norm(graph.nodes[edge[0]]['pos'] - graph.nodes[edge[1]]['pos'])
            graph.edges[edge]['color'] = 'k'
        self.adj_start = nx.to_numpy_array(graph)

        # number of tolls from percentage to integer
        self.n_arcs = round(self.n_arcs / 100 * self.toll_proportion)

        origin_destination = list(combinations(graph.nodes, r=2))

        for i in range(self.n_commodities):
            o_d = origin_destination.pop(np.random.choice(range(len(origin_destination))))
            n_users = np.random.choice(range(*self.nr_users))
            self.commodities.append(ArcCommodity(*o_d, n_users))

        # making it "bidirectional"
        self.npp: nx.Graph
        self.npp = graph.to_directed()
        # selecting toll arcs

        # first 2/3 of the total
        two_third_total_tolls = round((self.n_arcs / 3) * 2)

        frequency_arcs = {a: 0 for a in self.npp.edges}
        for k in self.commodities:
            path = nx.dijkstra_path(graph, k.origin, k.destination, weight='weight')
            for i in range(len(path) - 1):
                frequency_arcs[(path[i], path[i + 1])] += 1

        # avoid iterating two times on the same arc
        new_freq_arcs = {}
        for a in frequency_arcs:
            if (a[1], a[0]) in new_freq_arcs:
                new_freq_arcs[(a[1], a[0])] += frequency_arcs[a]
            else:
                new_freq_arcs[a] = frequency_arcs[a]

        frequency_arcs = dict(sorted(new_freq_arcs.items(), key=lambda x: x[1], reverse=True))
        frequency_arcs_to_list = list(frequency_arcs.keys())
        n = 0

        for edge in frequency_arcs_to_list:  # [:two_third_total_tolls]:

            arcs = list(self.npp.edges)
            # remove the current edge and toll arcs if any exist
            for i in range(len(self.toll_arcs)):
                arcs.remove(self.toll_arcs[i])
            arcs.remove(edge)
            arcs.remove((edge[1], edge[0]))

            G = nx.DiGraph()
            G.add_edges_from(arcs)
            a = []
            for i, k in enumerate(self.commodities):
                try:
                    path_exists = nx.has_path(G, k.origin, k.destination)
                except nx.NodeNotFound:
                    path_exists = False

                a.append(path_exists)

            if n < two_third_total_tolls and all(a):
                self.npp.edges[edge]['color'] = 'r'
                self.toll_arcs.append(edge)
                self.toll_arcs_undirected.append(edge)
                # obv is a toll arc also the other verse
                self.npp.edges[(edge[1], edge[0])]['color'] = 'r'
                self.toll_arcs.append((edge[1], edge[0]))
                n += 1
            else:
                if n == self.n_arcs:
                    break

                if all(a):
                    # we delete it with a percentage (rn is 0.5) to scatter them among the total of the arcs
                    if np.random.uniform() > 0.5:
                        self.npp.edges[edge]['color'] = 'r'
                        self.toll_arcs.append(edge)
                        self.toll_arcs_undirected.append(edge)
                        # obv is a toll arc also the other verse
                        self.npp.edges[(edge[1], edge[0])]['color'] = 'r'
                        self.toll_arcs.append((edge[1], edge[0]))
                        n += 1
                    else:
                        if edge and (edge[1], edge[0]) not in self.free_arcs:
                            self.free_arcs.append(edge)
                            self.free_arcs.append((edge[1], edge[0]))
                else:
                    if edge and (edge[1], edge[0]) not in self.free_arcs:
                        self.free_arcs.append(edge)
                        self.free_arcs.append((edge[1], edge[0]))

        arcs_all = list(self.npp.edges)
        if self.toll_arcs:
            for i in range(len(self.toll_arcs)):
                arcs_all.remove(self.toll_arcs[i])
        if self.free_arcs:
            for i in range(len(self.free_arcs)):
                arcs_all.remove(self.free_arcs[i])
        for edge in arcs_all:
            self.free_arcs.append(edge)

        # halving the random cost of the tariff arcs to make them more attractive
        for a in self.toll_arcs:
            self.npp.edges[a]['weight'] = self.npp.edges[a]['weight'] / 2

        # creating b vector, bigM and bigN
        for comm in self.commodities:
            comm.set_b(self.npp)
            comm.set_quantities_for_M(self.npp, self.toll_arcs, self.free_arcs)
            comm.set_M(self.npp, self.toll_arcs)
        self.N_p = {p: max([k.M_p[p] for k in self.commodities]) for p in self.toll_arcs}

        self.n_users = np.array([comm.n_users for comm in self.commodities])
        #
        # print('Instance:')
        # print('n locations = ', self.n_locations, '   n arcs = ', self.n_arcs*2, '  toll proportion = ',
        #       self.toll_proportion, '%', '  n tolls = ', self.n_arcs, '  n commodities = ', self.n_commodities)
        #

        self.tolls = [ArcToll(a, self.commodities, self.npp.edges[a]['weight']) for a in self.toll_arcs]
        self.free = [Arc(a, self.npp.edges[a]['weight']) for a in self.free_arcs]
        self.n_tolls = len(self.tolls)

        self.adj = nx.to_numpy_array(self.npp)

    def show(self):
        nx.draw(self.npp, pos=[self.npp.nodes[i]['pos'] for i in self.npp.nodes], edge_color=[self.npp[u][v]['color'] for u, v in self.npp.edges],
                with_labels=True, font_size=7)
        plt.show()

    @staticmethod
    def iterations_on_arc(i, arcs):
        exiting = []  # (i, .. )  i-
        entering = []  # ( .., i)  i+
        for a in arcs:
            if a[0] == i:  # (i, .. )  i-
                exiting.append(a)
            if a[1] == i:  # ( .., i)  i+
                entering.append(a)
        return exiting, entering

