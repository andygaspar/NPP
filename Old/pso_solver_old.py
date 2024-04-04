import numpy as np
from scipy.stats import qmc

from Instance.instance import Instance
from CPP.PSO.swarm import Swarm
from Solver.solver import GlobalSolver
from Old.lower_aggregated_2 import LowerSolverAggregated2


class PsoSolver:

    def __init__(self, npp: Instance,  path_costs, n_particles, n_iterations, N_PARTS, n_cut, N_DIV, n_u_l,
                 normalised=True, init_sol_num=None, time_limit=None, verbose=True):
        self.best_val = None
        self.best = None
        self.npp = npp
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.path_costs = path_costs
        self.lower_solver = LowerSolverAggregated2(npp, n_particles)
        self.lower_solver.set_up()
        # self.swarm = Swarm(npp.commodities_tax_free, npp.n_users, npp.transfer_costs, npp.upper_bounds,
        #                    npp.n_commodities, npp.n_toll_paths, n_particles, n_iterations, N_PARTS=N_PARTS,
        #                    n_cut=n_cut, N_DIV=N_DIV, n_u_l=n_u_l, normalised=normalised, verbose=verbose)
        self.swarm = Swarm(npp.commodities_tax_free, npp.n_users, npp.transfer_costs, npp.upper_bounds,
                           npp.n_commodities, npp.n_toll_paths, n_particles, n_iterations, n_u_l=n_u_l)

        self.init_sol_num = init_sol_num
        self.time_limit = time_limit

    def run(self):
        run_best = 0
        personal_run_results = 0

        if self.time_limit is not None and self.init_sol_num is not None:
            global_solver = GlobalSolver(self.npp, time_limit=self.time_limit, min_sol_num=self.init_sol_num)
            global_solver.solve()
            init_sol = global_solver.current_solution
            print('init sol val', global_solver.best_val)
            print('init sol', init_sol)
            global_solver.print_model()

        init_sol = np.random.uniform(0, 1, size=(self.npp.n_toll_paths, self.n_particles))
        vel_init = np.random.uniform(-1, 1, size=(self.npp.n_toll_paths, self.n_particles))/100
        lb = np.zeros(self.npp.n_toll_paths)
        ub = np.ones_like(lb)
        self.swarm.run(init_sol, vel_init, ub, lb)

        self.best, self.best_val = self.swarm.get_best()
        # print(self.npp.upper_bounds)
        print(self.best * self.npp.upper_bounds)
        # print("final ", self.best)


    def random_init(self):
        return np.random.uniform(size=(self.n_particles, self.npp.n_toll_paths)) * self.npp.upper_bounds

    def compute_latin_hypercube_init(self, dimensions):
        init_positions = self.random_init()

        tolls_idx = np.argsort(self.npp.upper_bounds)[::-1]
        u_bounds = self.npp.upper_bounds[tolls_idx]

        l_bounds = np.zeros(dimensions)
        sampler = qmc.LatinHypercube(d=dimensions)
        latin_positions = qmc.scale(sampler.random(n=self.n_particles), l_bounds, u_bounds[:dimensions])

        init_positions[:, tolls_idx[:dimensions]] = latin_positions

        return init_positions