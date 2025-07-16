import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

n = 12
commodities = 20
toll_proportion = 0.2

g = nx.grid_2d_graph(n, n)
pos = {(x, y): (y, -x) for x, y in g.nodes()}

edges = list(g.edges)
for i in range(int(len(edges) * 0.2)):
    e = edges.pop(random.choice(range(len(edges))))
    g.edges[e]['weight'] = 35
    g.edges[e]['counter'] = 0
    g.edges[e]['tall'] = False
    g.edges[e]['color'] = 'blue'

for e in edges:
    g.edges[e]['weight'] = np.random.uniform(5, 35)
    g.edges[e]['counter'] = 0
    g.edges[e]['tall'] = False
    g.edges[e]['color'] = 'blue'

shortest_paths = []

for com in range(commodities):
    o, d = random.choice(list(g.nodes)), None
    found_destination = False
    while not found_destination:
        d = random.choice(list(g.nodes))
        if o != d:
            found_destination = True
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

n_tolls = int(len(g.edges) * toll_proportion)

for i in range(int(n_tolls * 2 / 3)):
    g.edges[edges[idxs[i]]]['tall'] = True
    g.edges[edges[idxs[i]]]['weight'] /= 2
    g.edges[edges[idxs[i]]]['color'] = 'red'
    remained_edges.remove(edges[idxs[i]])


for _ in range(n_tolls - int(n_tolls * 2 / 3)):  # one third random (computed this way to avoid rounding issues)
    e = remained_edges.pop(random.choice(range(len(remained_edges))))
    g.edges[e]['tall'] = True
    g.edges[e]['weight'] /= 2
    g.edges[e]['color'] = 'red'


plt.rcParams['figure.figsize'] = (12, 8)
edge_color = [g.edges[e]['color'] for e in g.edges]
nx.draw(g, pos=pos, with_labels=True, edge_color=edge_color)
plt.show()

t = sum(1 if g.edges[e]['tall'] else 0 for e in g.edges)


points = np.random.uniform(0, 1, size= (4 ** 2, 2))


from scipy.spatial import Delaunay

tri = Delaunay(points)
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
neighbours = tri.vertex_neighbor_vertices
edges = []
for i in range(4**2):
    pass