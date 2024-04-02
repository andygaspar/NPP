import random
import time

import numpy as np
import pandas as pd

from Genetic.genetic_old import GeneticOld
from Instance.instance import Instance


def improve_solution(pb: Instance, sol: np.array, obj_val, tol=0.0000000000001):
    improving = True
    while improving:
        profit = 0
        new_sol = sol.copy()
        prices = np.append(new_sol, [0])
        path_profit = {}
        path_dict_diff = {}
        for commodity in pb.commodities:
            costs = new_sol + commodity.c_p_vector
            costs = np.append(costs, commodity.c_od)
            idxs = np.argsort(costs)
            costs = np.sort(costs)
            duplicates = np.where(costs == costs[0])[0]
            if duplicates.shape[0] > 1:
                reorder = np.argsort(prices[idxs[duplicates]])[::-1]
                costs[duplicates] = costs[duplicates[reorder]]
                idxs[duplicates] = idxs[duplicates[reorder]]
            profit += prices[idxs[0]] * commodity.n_users
            if idxs[0] < pb.n_paths:
                if idxs[0] in path_profit.keys():
                    path_profit[idxs[0]] += new_sol[idxs[0]] * commodity.n_users
                    path_dict_diff[idxs[0]].append(costs[1] - costs[0])
                else:
                    path_profit[idxs[0]] = new_sol[idxs[0]] * commodity.n_users
                    path_dict_diff[idxs[0]] = [costs[1] - costs[0]]

        path_dict_sol = {path: min(path_dict_diff[path]) - tol for path in path_dict_diff if min(path_dict_diff[path]) > tol}

        path_profit_order = sorted(path_profit.items(), key=lambda x: x[1], reverse=True)
        for path in [o[0] for o in path_profit_order if o[0] in path_dict_sol.keys()]:
            new_sol[path] += path_dict_sol[path]

        new_val = pb.compute_solution_value(new_sol)
        # print(new_val)
        if new_val > obj_val:
            sol = new_sol
            obj_val = new_val
        else:
            improving = False
    return sol


def improve_solution2(pb: Instance, sol: np.array, obj_val):
    improving = True
    while improving:
        profit = 0
        new_sol = sol.copy()
        prices = np.append(new_sol, [0])
        path_profit = {}
        path_cost_to_improve = {}
        for commodity in pb.commodities:
            costs = new_sol + commodity.c_p_vector
            costs = np.append(costs, commodity.c_od)
            idxs = np.argsort(costs)
            costs = np.sort(costs)
            duplicates = np.where(costs == costs[0])[0]
            if duplicates.shape[0] > 1:
                reorder = np.argsort(prices[idxs[duplicates]])[::-1]
                costs[duplicates] = costs[duplicates[reorder]]
                idxs[duplicates] = idxs[duplicates[reorder]]
            profit += prices[idxs[0]] * commodity.n_users
            if idxs[0] < pb.n_paths:
                if idxs[0] in path_profit.keys():
                    path_profit[idxs[0]] += new_sol[idxs[0]] * commodity.n_users

                    if costs[1] < path_cost_to_improve[idxs[0]][1]:
                        path_cost_to_improve[idxs[0]] = [idxs[1], costs[1]]
                else:
                    path_profit[idxs[0]] = new_sol[idxs[0]] * commodity.n_users
                    path_cost_to_improve[idxs[0]] = [idxs[1], costs[1]]

        # print(new_sol)
        for path in path_cost_to_improve.keys():
            new_sol[path] = prices[path_cost_to_improve[path][0]]
        # print(new_sol)

        new_val = pb.compute_solution_value(new_sol)
        if new_val > obj_val:
            sol = new_sol
            obj_val = new_val
        else:
            improving = False
    return sol



np.random.seed(0)
random.seed(0)
sol = np.random.uniform(15, 25, size=56)

npp = Instance(n_paths=56, n_commodities=56, seeds=0)
# for c in npp.commodities:
#     c.c_od = int(c.c_od)
#     c.c_p_vector = c.c_p_vector.astype(int)
# sol = np.loadtxt('test_solution.csv')
t = time.time()
population_size = 64
values = np.array([c.c_od for c in npp.commodities] + [v for p in npp.commodities for v in p.c_p_vector])
init_sol = np.random.choice(values, size=(population_size, npp.n_paths))
print(npp.compute_solution_value(init_sol[0]))
new_sol = improve_solution(npp, init_sol[0], npp.compute_solution_value(init_sol[0]), 0)
new_sol = improve_solution2(npp, init_sol[0], npp.compute_solution_value(init_sol[0]))


g = GeneticOld(population_size=265, pso_population=0, pso_selection=0, npp=npp, mutation_rate=0.02,
                               offspring_rate=0.5, n_threads=1)



g.init_values()
for i in range(20):
    for i in range(4):
        g.population[g.pop_size - i - 1] = improve_solution(npp, g.population[i],
                                                                        g.values[g.pop_size - i - 1], 1e-9)
    g.generation_es()
    print(g.best_val)


#
genetic = GeneticOld(population_size, 16, 4, npp, 0.5)
genetic.init_values()
genetic.generation()

n = 20
t = time.time()
print(genetic.best_val)
for i in range(n):
    # for i in range(4):
    #     genetic.population[genetic.pop_size - i - 1] = improve_solution(npp, genetic.population[i], genetic.values[genetic.pop_size - i - 1], 1e-9)
    genetic.generation()
    print(genetic.best_val)
print(genetic.best_val, time.time() - t)
#
# t = time.time()
# genetic.generation(init_sol)
# for i in range(n*2):
#     genetic.generation()
# print(genetic.best_val, time.time() - t)