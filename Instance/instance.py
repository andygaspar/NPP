import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Instance:

    def __init__(self, n_locations, n_tolls, n_users, cr_locations=(10, 20), cr_transfer=(5, 10),
                 seeds=False):

        if seeds:
            np.random.seed(0)
        self.n_locations = n_locations
        self.n_tolls = n_tolls
        self.n_users = n_users
        self.users = []
        self.cr_locations = cr_locations
        self.cr_transfer = cr_transfer

        self.locations = ['u ' + str(i) for i in range(self.n_locations)]
        for i in range(self.n_users):
            origin = np.random.choice(self.locations)
            destination = np.random.choice([location for location in self.locations if location != origin])
            if origin > destination:
                origin, destination = destination, origin
            self.users.append((origin, destination))
        self.tolls = ['T ' + str(i) for i in range(self.n_tolls)]
        self.npp = nx.Graph()

        self.npp.add_nodes_from([(u, {'color': 'g'}) for u in self.locations])
        self.npp.add_nodes_from([(t, {'color': 'r'}) for t in self.tolls])

        for arc in [(i, j) for i in self.locations for j in self.locations if i != j]:
            self.npp.add_edge(*arc, color='g')

        for arc in [(i, j) for i in self.tolls for j in self.tolls if i != j]:
            self.npp.add_edge(*arc, color='r', weight=np.random.uniform(
                self.cr_locations[0], self.cr_locations[1]))

        for arc in [(i, j) for i in self.locations for j in self.tolls if i != j]:
            self.npp.add_edge(*arc, color='b', weight=np.random.uniform(
                self.cr_transfer[0], self.cr_transfer[1]))

    def show(self):

        nx.draw(self.npp, node_color=[self.npp.nodes[n]['color'] for n in self.npp.nodes],
                edge_color=[self.npp[u][v]['color'] for u, v in self.npp.edges],
                with_labels=True, font_size=7)
        plt.show()



