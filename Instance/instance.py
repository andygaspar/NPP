from typing import List

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from Instance.commodity import Commodity


class Instance:

    def __init__(self, n_locations, n_tolls, n_commodities, cr_locations=(10, 20), cr_transfer=(5, 10), nr_users=(1, 5),
                 seeds=False):

        if seeds:
            np.random.seed(0)
        self.n_locations = n_locations
        self.n_tolls = n_tolls
        self.n_commodities = n_commodities
        self.commodities: List[Commodity] = []
        self.users = []
        self.cr_locations = cr_locations
        self.cr_transfer = cr_transfer
        self.nr_users = nr_users

        self.locations = ['u ' + str(i) for i in range(self.n_locations)]
        self.tolls = ['T ' + str(i) for i in range(self.n_tolls)]
        self.toll_paths = list(combinations(self.tolls, r=2))
        self.n_paths = len(self.toll_paths)

        self.npp = nx.Graph()
        self.npp.add_nodes_from([(u, {'color': 'g'}) for u in self.locations])
        self.npp.add_nodes_from([(t, {'color': 'r'}) for t in self.tolls])

        for arc in [(i, j) for i in self.locations for j in self.locations if i < j]:
            self.npp.add_edge(*arc, color='g', weight=np.random.uniform(*self.cr_locations))

        for arc in [p for p in self.toll_paths]:
            self.npp.add_edge(*arc, color='r', weight=np.random.uniform(*self.cr_transfer))

        for arc in [(i, j) for i in self.locations for j in self.tolls]:
            self.npp.add_edge(*arc, color='b', weight=np.random.uniform(*self.cr_transfer))



        origin_destination = list(combinations(self.locations, r=2))

        for i in range(self.n_commodities):
            o_d = origin_destination.pop(np.random.choice(range(len(origin_destination))))
            n_users = np.random.choice(range(*self.nr_users))
            self.commodities.append(Commodity(*o_d, n_users, self.npp, self.toll_paths))

        self.N_p = {p: max([k.M_p[p] for k in self.commodities]) for p in self.toll_paths}



    def show(self):

        nx.draw(self.npp, node_color=[self.npp.nodes[n]['color'] for n in self.npp.nodes],
                edge_color=[self.npp[u][v]['color'] for u, v in self.npp.edges],
                with_labels=True, font_size=7)
        plt.show()
