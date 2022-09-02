import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

n_commodities = 5

commodities = np.random.choice(5, 5)

npp = nx.Graph()

npp.add_nodes_from([(i * 2, {'color': 'green'}) for i in range(n_commodities)])
print(npp.nodes)
for arc in [(i, j) for i in npp.nodes for j in npp.nodes if i != j]:
    npp.add_edge(*arc, color='g')


nx.draw(npp, node_color=[npp.nodes[n]['color'] for n in npp.nodes], edge_color=[npp[u][v]['color'] for u, v in npp.edges])
plt.show()



