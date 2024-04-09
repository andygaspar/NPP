class ArcInstance:

    def __init__(self, n_locations, n_commodities):
        self.n_locations = n_locations
        self.n_commodities = n_commodities
        self.users = []
        self.n_tolls = None
        self.npp = None
        self.N_p = None
        self.toll_arcs_undirected = None
        self.toll_arcs = None
        self.free_arcs = None
        self.commodities = None
