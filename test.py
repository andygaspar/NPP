import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Instance:

    def __init__(self, n_commodities, n_tolls, cost_range=(10, 20)):
        self.n_commodities = n_commodities
        self.n_tolls = n_tolls
        self.cost_range = cost_range

        self.commodities = ['u ' + str(i) for i in range(self.n_commodities)]
        self.tolls = ['T ' + str(i) for i in range(self.n_tolls)]
        self.npp = nx.Graph()

        self.npp.add_nodes_from([(u, {'color': 'g'}) for u in self.commodities])
        self.npp.add_nodes_from([(t, {'color': 'r'}) for t in self.tolls])

        for arc in [(i, j) for i in self.commodities for j in self.commodities if i != j]:
            self.npp.add_edge(*arc, color='g')

        for arc in [(i, j) for i in self.tolls for j in self.tolls if i != j]:
            self.npp.add_edge(*arc, color='r', weight=np.random.uniform(self.cost_range[0], self.cost_range[1]))

        for arc in [(i, j) for i in self.commodities for j in self.tolls if i != j]:
            self.npp.add_edge(*arc, color='b')

    def show(self):

        nx.draw(self.npp, node_color=[self.npp.nodes[n]['color'] for n in self.npp.nodes],
                edge_color=[self.npp[u][v]['color'] for u, v in self.npp.edges],
                with_labels=True, font_size=7)
        plt.show()



