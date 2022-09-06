class Commodity:

    def __init__(self, origin, destination, n_users, graph, paths):
        self.origin = origin
        self.destination = destination
        self.n_users = n_users
        self.c_od = graph.edges[origin, destination]['weight']
        self.c_p = {p: self.path_cost(p, graph) for p in paths}
        self.M_p = {p: max([0, self.c_od - self.c_p[p]]) for p in paths}

    def path_cost(self, p, g):
        i, j = p
        a = g.edges[self.origin, i]['weight'] + g.edges[self.destination, j]['weight']
        b = g.edges[self.origin, j]['weight'] + g.edges[self.destination, i]['weight']
        return min([a, b])

    def __repr__(self):
        return self.origin + ' -> ' + self.destination
