import random
import time

import numpy as np

from Solver.genetic_pso_solver import GeneticPso
from Old.Genetic.genetic_old import GeneticOld
from Instance.instance import Instance
from Solver.solver import GlobalSolver


def equal(a, b, tol):
    return b - tol <= a <= b + tol


def lower_equal(a, b, tol):
    return a <= b + tol


def strictly_lower_equal(a, b, tol):
    return a < b - tol


def improve_solution_3(pb: Instance, sol: np.array, tol):
    improving = True
    new_sol = sol.copy()
    while improving:
        total_profit = 0
        cost_difference = np.ones(pb.n_paths) * 100
        cost_difference_idx = []

        for commodity in pb.commodities:
            costs = new_sol + commodity.c_p_vector
            com_cost, com_profit = commodity.c_od, 0
            idx_lowest_cost = pb.n_commodities

            for i, c in enumerate(costs):
                if lower_equal(c, com_cost, tol):
                    if strictly_lower_equal(c, com_cost, tol):
                        com_profit = new_sol[i]
                        com_cost = c
                        idx_lowest_cost = i
                    elif new_sol[i] > com_profit:
                        com_profit = new_sol[i]
                        com_cost = c
                        idx_lowest_cost = i
            total_profit += com_profit * commodity.n_users
            if idx_lowest_cost < pb.n_commodities:
                second_cost = commodity.c_od
                for i, c in enumerate(costs):
                    if i != idx_lowest_cost and lower_equal(c, second_cost, tol):
                        if equal(c, second_cost, tol) and com_profit < new_sol[i]:
                            second_cost = c

                        elif strictly_lower_equal(c, second_cost, tol):
                            second_cost = c

                difference = second_cost - com_cost
                if difference < cost_difference[idx_lowest_cost] - tol:
                    cost_difference[idx_lowest_cost] = difference
                    cost_difference_idx.append(idx_lowest_cost)
        if cost_difference[cost_difference_idx].sum() > 0:
            new_sol[cost_difference_idx] += cost_difference[cost_difference_idx]
        else:
            improving = False
    new_val = pb.compute_solution_value_with_tol(new_sol, tol)
    return new_sol, new_val

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


def improve_solution2(pb: Instance, sol: np.array, obj_val, tol=0.0000000000001):
    improving = True

    while improving:
        commodity_profit = np.zeros(pb.n_commodities)

        commodity_path = np.zeros(pb.n_commodities, dtype=int)
        commodity_virtual_path = np.zeros(pb.n_commodities, dtype=int)
        profit = 0
        new_sol = sol.copy()
        prices = np.append(new_sol, [0])
        path_profit = np.zeros(pb.n_paths)
        path_second_choice = np.zeros(pb.n_paths, dtype=int)
        path_cost_diff = -np.ones(pb.n_paths)

        commodity_path_rank = np.zeros((pb.n_commodities, pb.n_paths + 1), dtype=int)

        for i, commodity in enumerate(pb.commodities):
            costs = new_sol + commodity.c_p_vector
            costs = np.append(costs, commodity.c_od)
            commodity_path_rank[i] = np.argsort(costs)
            costs = np.sort(costs)
            duplicates = np.where(costs == costs[0])[0]

            if duplicates.shape[0] > 1:
                reorder = np.argsort(prices[commodity_path_rank[i][duplicates]])[::-1]
                costs[duplicates] = costs[duplicates[reorder]]
                commodity_path_rank[i][duplicates] = commodity_path_rank[i][duplicates[reorder]]

            profit += prices[commodity_path_rank[i][0]] * commodity.n_users
            commodity_path[i] = commodity_path_rank[i][0]
            commodity_virtual_path[i] = commodity_path_rank[i][1]
            if commodity_path_rank[i][0] < pb.n_paths:
                path_profit[commodity_path_rank[i][0]] += new_sol[commodity_path_rank[i][0]] * commodity.n_users
                commodity_profit[i] = new_sol[commodity_path_rank[i][0]] * commodity.n_users

                if path_cost_diff[commodity_path_rank[i][0]] < 0 or path_cost_diff[commodity_path_rank[i][0]] > costs[1] - costs[0] + tol:
                    path_cost_diff[commodity_path_rank[i][0]] = costs[1] - costs[0] - tol
                    if commodity_path_rank[i][1] < pb.n_paths:
                        path_second_choice[commodity_path_rank[i][0]] = commodity_path_rank[i][1]

        path_order = np.argsort(path_profit)[::-1]

        old_val_test = sum([sol[commodity_path_rank[i][0]] * pb.commodities[i].n_users for i in range(pb.n_commodities)
                            if commodity_path_rank[i][0] < pb.n_paths])
        print(commodity_path_rank)
        for p in path_order:
            if path_cost_diff[p] > 0:
                new_sol[p] += path_cost_diff[p]
                print(new_sol[p], new_sol[path_second_choice[p]])
                if new_sol[p] < new_sol[path_second_choice[p]] + tol:
                    commodity_second_choice = np.where(commodity_path_rank[:, 1] == p)[0]

                    for i in commodity_second_choice:
                        j = commodity_path_rank[i][0]
                        print(p, j, new_sol[p] + pb.commodities[i].c_p_vector[p], new_sol[j] + pb.commodities[i].c_p_vector[j])
                        first_val = new_sol[j] + pb.commodities[i].c_p_vector[j] - tol
                        second_val = new_sol[p] + pb.commodities[i].c_p_vector[p]
                        if j == p and second_val - tol <= first_val <= second_val + tol:
                            temp = commodity_path_rank[i][1]
                            commodity_path_rank[i][1] = commodity_path_rank[i][0]
                            commodity_path_rank[i][0] = temp
                else:
                    commodity_first_choice = np.where(commodity_path_rank[:, 0] == p)[0]
                    for i in [v for v in commodity_first_choice]:
                        # print(commodity_path_rank[i][0], p)
                        j = commodity_path_rank[i][1]
                        first_val = new_sol[j] + pb.commodities[i].c_p_vector[j] - tol
                        second_val = new_sol[p] + pb.commodities[i].c_p_vector[p]
                        print('*', p, j, first_val, second_val)
                        if second_val - tol <= first_val <= second_val + tol:
                            temp = commodity_path_rank[i][1]
                            commodity_path_rank[i][1] = commodity_path_rank[i][0]
                            commodity_path_rank[i][0] = temp

        new_val_test = sum([new_sol[commodity_path_rank[i][0]] * pb.commodities[i].n_users for i in range(pb.n_commodities)
                            if commodity_path_rank[i][0] < pb.n_paths])

        new_val = pb.compute_solution_value(new_sol)

        # print('\n', profit, path_profit.sum())
        # print(new_val)
        if new_val > obj_val:
            sol = new_sol
            obj_val = new_val
        else:
            improving = False
    return sol


def improve_solution3(pb: Instance, sol: np.array, obj_val, tol=0.0000000000001):
    improving = True
    prices = np.zeros((pb.n_commodities, pb.n_paths + 1))
    prices[:, :-1] = np.repeat(sol, pb.n_commodities).reshape((pb.n_commodities, pb.n_paths)).T
    trans_costs = np.array([np.concatenate([c.c_p_vector, [c.c_od]]) for c in pb.commodities])
    n_users = np.array([c.n_users for c in pb.commodities])
    costs = prices + trans_costs
    idxs = np.argsort(costs, axis=1)
    for i in range(pb.n_commodities):
        duplicates = np.where(costs[i] == costs[i, idxs[i, 0]])[0]
        if duplicates.shape[0] > 0:
            reorder = np.argsort(prices[0][idxs[i][duplicates]])[::-1]
            idxs[i][duplicates] = idxs[i][duplicates[reorder]]

    obj_val = sum([prices[0, idxs[i, 0]] * n_users[i] for i in range(pb.n_commodities)])

    while improving:
        new_sol = sol.copy()
        path_profit = np.zeros(pb.n_paths)
        path_cost_diff = -np.zeros(pb.n_paths)

        for i in range(pb.n_commodities):
            if idxs[i][0] < pb.n_paths:
                path_profit[idxs[i, 0]] += new_sol[idxs[i, 0]] * n_users[i]

                if path_cost_diff[idxs[i, 0]] == 0 or path_cost_diff[idxs[i, 0]] > costs[i, idxs[i, 1]] - costs[i, idxs[i, 0]] + tol:
                    path_cost_diff[idxs[i, 0]] = costs[i, idxs[i, 1]] - costs[i, idxs[i, 0]]

        new_sol += path_cost_diff
        prices[:, :-1] = np.repeat(new_sol, pb.n_commodities).reshape((pb.n_commodities, pb.n_paths)).T

        costs = prices + trans_costs
        idxs = np.argsort(costs, axis=1)
        for i in range(pb.n_commodities):
            duplicates = np.where(costs[i] == costs[i, idxs[i, 0]])[0]
            if duplicates.shape[0] > 0:
                reorder = np.argsort(prices[0][idxs[i][duplicates]])[::-1]
                idxs[i][duplicates] = idxs[i][duplicates[reorder]]

        # path_order = np.argsort(path_profit)[::-1]

        # for p in path_order:
        #     if path_cost_diff[p] > 0:
        #         to_update = np.where(idxs == p)
        #         for i in range(pb.n_commodities):
        #             unsorted = True
        #             current_idx = to_update[1][i]
        #             while unsorted:
        #                 if current_idx == idxs.shape[1] - 1:
        #                     unsorted = False
        #                 elif costs[i, idxs[i, current_idx]] < costs[i, idxs[i, current_idx + 1]]:
        #                     unsorted = False
        #                 elif prices[0, idxs[i, current_idx]] > prices[0, idxs[i, current_idx + 1]]:
        #                     unsorted = False
        #                 else:
        #                     temp = idxs[i][current_idx + 1]
        #                     idxs[i][current_idx + 1] = idxs[i][current_idx]
        #                     idxs[i][current_idx] = temp
        #                     current_idx += 1

        new_val = sum([prices[0, idxs[i, 0]] * n_users[i] for i in range(pb.n_commodities)])

        # print('\n', profit, path_profit.sum())
        if new_val > obj_val:
            sol = new_sol
            obj_val = new_val
        else:
            improving = False
    return sol

#
# np.random.seed(0)
# random.seed(0)
# n_paths = 90
# n_commodities = 90
# npp = Instance(n_paths=n_paths, n_commodities=n_commodities)
#
# solver = GlobalSolver(npp, verbose=True, time_limit=160)
# solver.solve(ub=True)
#
# # for c in npp.commodities:
# #     c.c_od = int(c.c_od)
# #     c.c_p_vector = c.c_p_vector.astype(int)
# # sol = np.loadtxt('test_solution.csv')
# t = time.time()
# population_size = 256
# values = np.array([c.c_od for c in npp.commodities] + [v for p in npp.commodities for v in p.c_p_vector])
# init_sol = np.random.choice(values, size=(population_size, npp.n_paths))
# initial_val = npp.compute_solution_value(init_sol[0])
# print('*************', initial_val)
# new_sol = improve_solution3(npp, init_sol[0], initial_val)
#
# tt = time.time()
# #
