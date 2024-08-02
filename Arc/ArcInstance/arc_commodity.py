from typing import List

import networkx as nx
import numpy as np
from gurobipy import Model, GRB, quicksum


class ArcCommodity:
    def __init__(self, origin, destination, n_users):

        self.origin = origin
        self.destination = destination
        self.n_users = n_users
        self.name = str(self.origin) + ' -> ' + str(self.destination)
        self.solution_path = None
        self.solution_edges = None

        self.cost_free = None
        self.l_free = None
        self.l_all = None
        self.s_free = None
        self.s_all = None

        self.M_p = None
        self.b = None

        self.gamma_0 = None
        self.gamma_inf = None

    def set_b(self, graph):
        self.b = self.create_b(graph)

    def create_b(self, g):
        nodes_from_graph = list(g.nodes)
        o = nodes_from_graph.index(self.origin)
        d = nodes_from_graph.index(self.destination)
        b = np.zeros(len(g.nodes))
        b[o] = -1
        b[d] = 1
        return b

    def set_quantities_for_M(self, graph, toll_arcs, free_arcs):
        self.cost_free = self.create_cost_free(graph, toll_arcs, free_arcs)

    def create_cost_free(self, graph, toll_arcs, free_arcs):
        cost_free = {p: self.shortest_path(p[0], p[1], graph, free_arcs) for p in toll_arcs}
        # cost_free = {p: self.dijkstra_shortest_path(free_arcs=free_arcs, a=p[0], b=p[1]) for p in toll_arcs}
        return cost_free

    def set_M(self, graph, toll_arcs):
        self.M_p = self.create_M(graph, toll_arcs)

    def create_M(self, graph, toll_arcs):
        big_m = {p: max([0, self.cost_free[p] - graph.edges[p]['weight']]) for p in toll_arcs}
        return big_m

    def shortest_path(self, a, b, g, arcs):
        # a, b = p
        nodes = list(g.nodes)
        nodes.remove(a)
        if a != b:
            nodes.remove(b)

            shortest_path = Model('SP')
            shortest_path.setParam("OutputFlag", 0)
            shortest_path.modelSense = GRB.MINIMIZE

            x = shortest_path.addVars([arc for arc in arcs], vtype=GRB.BINARY)
            shortest_path.setObjective(quicksum(g.edges[arc]['weight'] * x[arc] for arc in arcs))

            exiting, entering = self.iterations_on_arc(a, arcs)
            shortest_path.addConstr(quicksum(x[k] for k in entering) - quicksum(x[k] for k in exiting) == -1)

            for i in nodes:
                exiting, entering = self.iterations_on_arc(i, arcs)
                shortest_path.addConstr(quicksum(x[k] for k in entering) - quicksum(x[k] for k in exiting) == 0)

            exiting, entering = self.iterations_on_arc(b, arcs)
            shortest_path.addConstr(quicksum(x[k] for k in entering) - quicksum(x[k] for k in exiting) == 1)

            shortest_path.optimize()

            status = shortest_path.status
            if status == 3:
                # print(p)
                path_cost = 100
            else:
                path_cost = shortest_path.objval
                # print(path_cost)
        else:
            path_cost = 0

        return path_cost

    def dijkstra_shortest_path(self, free_arcs, a, b):
        arc_on_free_path = []
        arcs = free_arcs

        graph = nx.DiGraph()
        graph.add_edges_from(arcs)

        path = nx.dijkstra_path(graph, a, b, weight='weight')
        for i in range(len(path) - 1):
            arc_on_free_path.append((path[i], path[i + 1]))
        return arc_on_free_path

    @staticmethod
    def iterations_on_arc(i, free_arcs):
        first = []  # (i, .. )  i-
        last = []  # ( .., i)  i+
        for a in free_arcs:
            if a[0] == i:  # (i, .. )  i-
                first.append(a)
            if a[1] == i:  # ( .., i)  i+
                last.append(a)
        return first, last

    def __repr__(self):
        return self.name


class Arc:
    def __init__(self, idx: tuple, c_a: float):
        self.idx = idx
        self.c_a = c_a


class ArcToll(Arc):

    def __init__(self, idx: tuple, commodities: List[ArcCommodity], c_a: float):
        super().__init__(idx, c_a)
        self.N_p = max([c.M_p[self.idx] for c in commodities])
        # self.L_p = min([c.M_p[self.idxs] for c in commodities])

