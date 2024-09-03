import random
from typing import List
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from Arc.ArcInstance.arc_commodity import ArcCommodity, ArcToll, Arc
from Arc.ArcInstance.arc_instance import ArcInstance


class GridInstance(ArcInstance):
    def __init__(self, n_locations, n_arcs, dim_grid, toll_proportion, n_commodities, costs=(5, 35), nr_users=(1, 5), seed=None):
        # costs = (2, 20)
        super().__init__(n_locations, n_commodities)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.name = 'Grid'
        self.toll_proportion = toll_proportion  # {5%, 10%, 15%, 20%}
        self.commodities: List[ArcCommodity] = []
        self.costs = costs
        self.nr_users = nr_users
        self.toll_arcs = []
        self.free_arcs = []
        self.toll_arcs_undirected = []
        self.dim_grid = dim_grid

        graph = nx.grid_graph(dim_grid, periodic=False)
        graph = nx.convert_node_labels_to_integers(graph)
        # nx.draw(graph, with_labels=True)
        # plt.show()

        # self.n_locations = dim_grid[0]*dim_grid[1]
        self.n_arcs = len(graph.edges)

        self.n_nodes = len(graph.nodes)
        self.n_edges = len(graph.edges)

        # ottanta_perc_archi = round(self.n_arcs/100 * 80)
        # i = 0

        for edge in graph.edges:
            # if i < ottanta_perc_archi:
            graph.edges[edge]['weight'] = np.random.uniform(*self.costs)
            graph.edges[edge]['color'] = 'k'
            # i += 1
            # else:
            #     graph.edges[edge]['weight'] = 35
            #     graph.edges[edge]['color'] = 'k'
            #     i += 1

        # number of tolls from percentage to integer
        self.n_tolls = round(self.n_arcs / 100 * self.toll_proportion)

        origin_destination = list(combinations(graph.nodes, r=2))

        for i in range(self.n_commodities):
            o_d = origin_destination.pop(np.random.choice(range(len(origin_destination))))
            n_users = np.random.choice(range(*self.nr_users))
            self.commodities.append(ArcCommodity(*o_d, n_users))

        # making it "bidirectional"
        self.npp = graph.to_directed()

        # selecting toll arcs

        # first 2/3 of the total
        two_third_total_tolls = round((self.n_tolls / 3) * 2)

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
                if n == self.n_tolls:
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

        self.tolls = [ArcToll(a, self.commodities, self.npp.edges[a]['weight']) for a in self.toll_arcs]
        self.free = [Arc(a, self.npp.edges[a]['weight']) for a in self.free_arcs]
        self.n_tolls = len(self.tolls)

        self.adj = nx.to_numpy_array(self.npp)
        self.edges_idx = dict((zip(self.free_arcs + self.toll_arcs, range(len(self.free_arcs) + len(self.toll_arcs)))))

        # print('Instance:')
        # print('n locations = ', self.n_locations, '   n arcs = ', self.n_arcs*2, '  toll proportion = ',
        #       self.toll_proportion, '%', '  n tolls', self.n_tolls, '  n commodities = ', self.n_commodities)

    def show(self, edge_color=True, with_labels=True, edge_weight=False, width=5, file=None):
        plt.rcParams["figure.figsize"] = (15, 10)
        # plt.rcParams["font.size"] = 20
        # pos = nx.nx_agraph.graphviz_layout(self.npp)

        pos = {}
        node = 0
        for i in range(self.dim_grid[1]):
            for j in range(self.dim_grid[0]):
                pos[node] = np.array([i * 0.1,  j * 0.1])
                node += 1

        edge_color = [self.npp[u][v]['color'] for u, v in self.npp.edges] if edge_color else None
        nx.draw(self.npp, pos=pos, edge_color=edge_color,
                with_labels=with_labels, font_size=22, width=width)
        if edge_weight:
            labels = nx.get_edge_attributes(self.npp, 'weight')
            for l in labels.keys():
                labels[l] = round(labels[l], 2)
            nx.draw_networkx_edge_labels(self.npp, pos, edge_labels=labels, font_size=15)
        if file is not None:
            plt.tight_layout()
            plt.savefig(file, transparent=True)
        else:
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
