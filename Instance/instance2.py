import os
import sys
from typing import List

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

import torch
import torch_geometric
from torch_geometric.data import HeteroData

from Instance.commodity import Commodity
import pandas as pd


class Commodity2:

    def __init__(self, name, nr_users, cr_free, cr_transfer, toll_names):
        self.name = name
        self.n_users = np.random.choice(range(*nr_users))
        self.cost_free = np.random.uniform(*cr_free)
        self.transfer = {p: np.random.uniform(*cr_transfer) for p in toll_names}
        self.M_p = {p: max([0, self.cost_free - self.transfer[p]]) for p in toll_names}

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name


class Toll:

    def __init__(self, name, commodities: List[Commodity2]):
        self.name = name
        self.cost = None
        self.N_p = max([k.M_p[self.name] for k in commodities])

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name


class Instance2:
    def __init__(self, n_tolls, n_commodities, cr_free=(10, 20), cr_transfer=(5, 10), nr_users=(1, 5),
                 seeds=False):

        if seeds:
            np.random.seed(0)
        self.n_tolls = n_tolls
        self.n_toll_paths = self.n_tolls
        self.n_commodities = n_commodities
        self.commodities: List[Commodity2]
        self.tolls: List[Toll]
        self.users = []
        self.cr_free = cr_free
        self.cr_transfer = cr_transfer
        self.nr_users = nr_users

        com_names = ['U ' + str(i) for i in range(self.n_commodities)]
        toll_names = ['T ' + str(i) for i in range(self.n_tolls)]

        self.commodities = [Commodity2(name, nr_users, cr_free, cr_transfer, toll_names) for name in com_names]
        self.com_dict = dict(zip(com_names, self.commodities))
        self.tolls = [Toll(name, self.commodities) for name in toll_names]

        self.graph = nx.Graph()

        self.graph.add_nodes_from([(com.name,
                                    {'type': 'commodity', 'type_int': 0, 'n_users': com.n_users,
                                     'free_path': com.cost_free, 'N_p': 0, 'n_commodities': self.n_commodities,
                                     'color': 'g'})
                                   for com in self.commodities])

        self.graph.add_nodes_from([(toll.name,
                                    {'type': 'toll', 'type_int': 1, 'n_users': 0, 'free_path': 0, 'N_p': toll.N_p,
                                     'n_commodities': self.n_commodities, 'color': 'r'})
                                   for toll in self.tolls])

        transfers = [(i.name, j.name) for i in self.commodities for j in self.tolls]
        for arc in transfers:
            self.graph.add_edge(*arc, color='b', transfer=self.com_dict[arc[0]].transfer[arc[1]])

        self.commodities_tax_free = np.array([comm.cost_free for comm in self.commodities])
        self.transfer_costs = np.array([comm.transfer[toll.name]
                                        for comm in self.commodities for toll in self.tolls])
        self.upper_bounds = np.array([p.N_p for p in self.tolls])
        self.lower_bounds = np.array([0.0 for _ in self.tolls])
        self.n_users = np.array([comm.n_users for comm in self.commodities])

    def show(self):
        nx.draw(self.graph, node_color=[self.graph.nodes[n]['color'] for n in self.graph.nodes],
                edge_color=[self.graph[u][v]['color'] for u, v in self.graph.edges],
                with_labels=True, font_size=7)
        plt.show()

    def make_torch_graph(self, solution):
        for com in self.commodities:
            self.graph.nodes[com.name]['x'] = [com.n_users, com.cost_free, 0, self.n_commodities]
            self.graph.nodes[com.name]['y'] = 0
        for i, toll in enumerate(self.tolls):
            self.graph.nodes[toll.name]['x'] = [0, 0, toll.N_p, self.n_commodities]
            self.graph.nodes[toll.name]['y'] = solution[i]
        data_homo = torch_geometric.utils.convert.from_networkx(self.graph,
                                                                group_node_attrs=['type_int', 'n_users',
                                                                                  'free_path',
                                                                                  'N_p', 'n_commodities'],
                                                                group_edge_attrs=['transfer'])
        return data_homo

    def make_torch_hetero_graph(self, solution):
        for com in self.commodities:
            self.graph.nodes[com.name]['x'] = [com.n_users, com.cost_free, 0]
            self.graph.nodes[com.name]['y'] = 0
        for i, toll in enumerate(self.tolls):
            self.graph.nodes[toll.name]['x'] = [0, 0, toll.N_p]
            self.graph.nodes[toll.name]['y'] = solution[i]

        data_homo = self.make_torch_graph(solution)
        data_hetero = HeteroData()
        data_hetero['commodities'].x = data_homo.x[:8, 1:3]
        data_hetero['tolls'].x = data_homo.x[self.n_commodities:, -1]
        data_hetero['tolls'].y = data_homo.y[self.n_commodities:]

        comm_tolls_idxs = torch.where(data_homo.edge_index[0] < self.n_commodities)[0]
        from_comm = data_homo.edge_index[0][comm_tolls_idxs]
        to_tolls = data_homo.edge_index[1][comm_tolls_idxs] - self.n_commodities
        data_hetero['commodities', 'transfer', 'tolls'].edge_index = torch.stack([from_comm, to_tolls])
        data_hetero['commodities', 'transfer', 'tolls'].edge_attr = data_homo.edge_attr[comm_tolls_idxs]
        return data_hetero

    def save_problem(self, folder_name=None):
        comm: Commodity
        transfer_costs = self.transfer_costs.reshape((self.n_commodities, self.n_toll_paths))
        if folder_name is None:
            folder_name = "TestCases/" + "comm_" + str(self.n_commodities) + "_tolls_" + str(self.n_tolls)
        try:
            os.mkdir(folder_name)
            pd.DataFrame(self.commodities_tax_free).to_csv(folder_name + '/commodities_tax_free.csv',
                                                           index=False, index_label=False, header=False)
            pd.DataFrame(transfer_costs).to_csv(folder_name + '/transfer_costs.csv',
                                                index=False, index_label=False, header=False)
            pd.DataFrame(self.upper_bounds).to_csv(folder_name + '/upper_bounds.csv',
                                                   index=False, index_label=False, header=False)
            pd.DataFrame(self.lower_bounds).to_csv(folder_name + '/lower_bounds.csv',
                                                   index=False, index_label=False, header=False)
            pd.DataFrame(self.n_users).to_csv(folder_name + '/n_users.csv',
                                              index=False, index_label=False, header=False)

        except:
            print(sys.exc_info())
