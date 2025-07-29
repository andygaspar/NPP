import copy
import time

from Arc.ArcInstance.arc_instance import DelaunayInstance, GridInstance
from Arc.ArcSolver.arc_solver import ArcSolver
import random
import numpy as np
from gurobipy import Model, GRB



seed = 9

random.seed(seed)
np.random.seed(seed)


N = 144
COMMODITIES = 30
TOLL_PROPORTION = 0.2

grid = GridInstance(COMMODITIES, TOLL_PROPORTION, N)
# grid.draw()

time_np = time.time()
adj_arc_node_bool = np.zeros((grid.n_edges, grid.n_nodes))

for i, e in enumerate(grid.edges):
    adj_arc_node_bool[i][e[0]] = -1
    adj_arc_node_bool[i][e[1]] = 1

adj_an_bool = np.zeros((grid.n_edges * grid.n_commodities, grid.n_nodes * grid.n_commodities))

b = np.zeros(grid.n_nodes * grid.n_commodities)
for i, k in enumerate(grid.commodities):
    b[i * grid.n_nodes + k.origin] = -1
    b[i * grid.n_nodes + k.destination] = 1

    adj_an_bool[grid.n_edges * i: grid.n_edges * (i + 1), grid.n_nodes * i: grid.n_nodes * (i + 1)] = adj_arc_node_bool

adj_an_bool = adj_an_bool.T

m = Model('')
x = m.addMVar(grid.n_edges * grid.n_commodities, vtype=GRB.BINARY)

c_at = np.array([grid.adj[e[0], e[1]] for e in grid.edges if e in grid.arc_tolls])
c_af = np.array([grid.adj[e[0], e[1]] for e in grid.edges if e in grid.arc_free])
la = m.addMVar(grid.n_nodes * grid.n_commodities, lb=-100)
T = m.addMVar(grid.n_tolls)
t = m.addMVar(grid.n_tolls * grid.n_commodities)
A_1_bool = np.zeros((grid.n_tolls * grid.n_commodities, grid.n_nodes))
A_2_bool = np.zeros((grid.n_free * grid.n_commodities, grid.n_nodes))
for k in range(grid.n_commodities):
    i, j = 0, 0
    for e in grid.edges:
        if e in grid.arc_tolls:
            A_1_bool[k * grid.n_tolls + i, e[0]] = -1
            A_1_bool[k * grid.n_tolls + i, e[1]] = 1
            i += 1
        if e in grid.arc_free:
            A_2_bool[k * grid.n_free + j, e[0]] = -1
            A_2_bool[k * grid.n_free + j, e[1]] = 1
            j += 1


x_idxs = np.array([i for i in range(grid.n_edges) if grid.edges[i] in grid.arc_tolls])
y_idxs = np.array([i for i in range(grid.n_edges) if grid.edges[i] in grid.arc_free])

N = np.array([toll.N_p for toll in grid.tolls])

m.addConstr( adj_an_bool @ x == b)
for k, comm in enumerate(grid.commodities):
    m.addConstr(A_1_bool[k * grid.n_tolls: (k + 1) * grid.n_tolls] @ la[k * grid.n_nodes: (k + 1) * grid.n_nodes] <= c_at + T)
    m.addConstr(A_2_bool[k * grid.n_free: (k + 1) * grid.n_free] @ la[k * grid.n_nodes: (k + 1) * grid.n_nodes] <= c_af)

    x_idxs_k = x_idxs + k * grid.n_edges
    y_idxs_k = y_idxs + k * grid.n_edges
    m.addConstr( (c_at * x[x_idxs_k]).sum() + t[k * grid.n_tolls: (k + 1) * grid.n_tolls].sum() + (c_af * x[y_idxs_k]).sum() ==
                    la[k * grid.n_nodes + comm.destination] - la[k * grid.n_nodes + comm.origin])
#
    M = np.array([comm.M_p[e] for e in grid.arc_tolls])
    m.addConstr(t[k * grid.n_tolls: (k + 1) * grid.n_tolls] <= M * x[x_idxs_k])
    m.addConstr(T - t[k * grid.n_tolls: (k + 1) * grid.n_tolls] <= N * (1 - x[x_idxs_k]))
    m.addConstr(t[k * grid.n_tolls: (k + 1) * grid.n_tolls] <= T)

max_profit = sum([k.n_users * sum(k.M_p.values()) for k in grid.commodities])
n_k = np.array([k.n_users for k in grid.commodities for e in grid.arc_tolls])

m.setObjective( (n_k * t).sum()/1000, sense=GRB.MAXIMIZE)
print('**********', seed)
m.optimize()

time_np = time.time() - time_np
print(m.objval)
problem = ArcSolver(grid)
problem.solve(verbose=True, time_limit=60)

print(m.objVal, problem.obj)
print(time_np, problem.time)
#
# test_problem = copy.deepcopy(grid)
# for k in problem.instance.commodities:
#     print(k, k.solution_edges)
# for i, k in enumerate(grid.commodities):
#     sol = np.nonzero(x.x[grid.n_edges * i: grid.n_edges * (i + 1)])[0]
#     solution = [grid.edges[e] for e in sol]
#     test_problem.commodities[i].solution_edges = solution
#     path = [e for e in solution if e[0] == k.origin]
#     while path[-1][1] != k.destination:
#         path += [e for e in solution if e[0] == path[-1][1]]
#     print(k, path)
#
# p_sol = [problem.T[e].x for e in grid.arc_tolls]
# res = grid.dijkstra(problem.adj_solution, problem.prices, grid.commodities[1])
#
# for i, e in enumerate(grid.arc_tolls):
#     print(e, T.x[i])
#
# # p = ArcSolver(problem.instance)
# # p.solve_max_price()


# problem.solution
# problem.t[(14, 19), problem.instance.commodities[0]]

pass
#
# # os.system("Arc/Arc_GA/install_arc.sh")
# time.sleep(2)
# random.seed(0)
# np.random.seed(0)
#
# N = 144
# # N = 25
#
# COMMODITIES = 30
# # COMMODITIES = 2
#
# TOLL_PROPORTION = 0.2
#
# grid = DelaunayInstance(COMMODITIES, TOLL_PROPORTION, N)
# # grid = GridInstance(COMMODITIES, TOLL_PROPORTION, N)
# print(len(grid.g.edges))
# # grid.save_problem('debug_test')
# # grid.draw(show_cost=True)
#
# ITERATIONS = 100
#
# # problem = ArcSolverNew(grid)
# # problem.solve(verbose=True, time_limit=60)
#
# g2 = GeneticArc(128, grid, mutation_rate=0.02)
# g2.run_cpp_heuristic(ITERATIONS, dijkstra_every=100, verbose=True, n_threads=16, seed=0)
# print(grid.compute_obj(g2.adj_solution, g2.prices))
# p = ArcSolver(g2.npp)
# p.solve_max_price(g2.solution)
#
#
# # g = GeneticArc(64, grid, mutation_rate=0.02)
# # g.run_cpp(ITERATIONS, verbose=True, n_threads=16, seed=1)
# #
# # print(grid.compute_obj(g.adj_solution, g.prices))
# # print(grid.compute_obj(problem.adj_solution, problem.prices))
# # p = ArcSolverNew(g.g)
# # p.solve_max_price(g.solution)
#
#
#
# #
#
#
# #
# obj_sol = grid.compute_obj(*problem.get_mats())
#
# print(problem.obj, g.best_val, g2.best_val, obj_sol)
#


# delaunay = DelaunayInstance(COMMODITIES, TOLL_PROPORTION, N)
# delaunay.draw()
# problem = ArcSolverNew(delaunay)
# problem.solve()
#
# voronoi = VoronoiNewInstance(COMMODITIES, TOLL_PROPORTION, N)
# voronoi.draw()
# problem = ArcSolverNew(voronoi)
# problem.solve(verbose=True)

#
# for var in problem.T:
#     print(problem.T[var].x)
