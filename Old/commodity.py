class Commodity:

    def __init__(self, origin, destination, n_users, graph, toll_paths):
        self.origin = origin
        self.destination = destination
        self.n_users = n_users
        self.cost_free = graph.edges[origin, destination]['weight']
        self.transfer_cost = {p: self.path_cost(p, graph) for p in toll_paths}
        self.M_p = {p: max([0, self.cost_free - self.transfer_cost[p]]) for p in toll_paths}

    def path_cost(self, p, g):
        i, j = p
        a = g.edges[self.origin, i]['weight'] + g.edges[self.destination, j]['weight']
        b = g.edges[self.origin, j]['weight'] + g.edges[self.destination, i]['weight']
        return min([a, b])

    def __repr__(self):
        return self.origin + ' -> ' + self.destination
