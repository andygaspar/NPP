import copy
from functools import total_ordering
from typing import List, Dict

import numpy as np

from Arc.ArcInstance.arc_commodity import ArcCommodity
from Arc.ArcInstance.arc_instance import ArcInstance




def min_cost(cost, profit, visited, tol):
    min_val = 100000
    min_index = 0
    max_profit = 0
    for i in range(cost.shape[0]):
        if not visited[i] and cost[i] <= min_val + tol:
            if cost[i] < min_val - tol:
                min_val = cost[i]
                min_index = i
                max_profit = profit[i]
            elif profit[i] > max_profit:
                min_val = cost[i]
                min_index = i
                max_profit = profit[i]

    return min_index


def dijkstra(adj, prices, commodity: ArcCommodity, tol=1e-9):
    MAX_DIST = 1000000
    cost = np.ones(adj.shape[0]) * MAX_DIST
    visited = np.zeros_like(cost, dtype=bool)
    profit = np.zeros_like(cost)

    cost[commodity.origin] = 0
    previous_vertex = np.zeros(adj.shape[0], dtype=int)

    for i in range(adj.shape[0] - 1):
        idx = min_cost(cost, profit, visited, tol)
        visited[idx] = True
        for j in range(adj.shape[0]):
            if not visited[j] and adj[idx, j] > 0 and cost[idx] != MAX_DIST and cost[idx] + adj[idx, j] <= cost[j] + tol:
                if cost[idx] + adj[idx, j] < cost[j] - tol:
                    cost[j] = cost[idx] + adj[idx, j]
                    profit[j] = profit[idx] + prices[idx, j] * commodity.n_users
                    previous_vertex[j] = idx
                elif profit[idx] + prices[idx, j] * commodity.n_users > profit[j]:
                    cost[j] = cost[idx] + adj[idx, j]
                    profit[j] = profit[idx] + prices[idx, j] * commodity.n_users
                    previous_vertex[j] = idx

    return cost[commodity.destination], visited, profit[commodity.destination], previous_vertex


def retrive_commodity_path(commodity: ArcCommodity, previous_vertex):
    current_node = commodity.destination
    paths = []
    while current_node != commodity.origin:
        paths.append((previous_vertex[current_node], current_node))
        current_node = previous_vertex[current_node]
    return paths[::-1]


def retrive_commodity_tolls(instance: ArcInstance, commodity: ArcCommodity, previous_vertex, prices, commodity_tolls, toll_idx):
    current_node = commodity.destination
    while current_node != commodity.origin:
        if (previous_vertex[current_node], current_node) in instance.toll_arcs:
            commodity_tolls[toll_idx[previous_vertex[current_node], current_node]] += (
                    prices[previous_vertex[current_node], current_node] * commodity.n_users)
        current_node = previous_vertex[current_node]
    return commodity_tolls


def run_arc_heuristic3(instance: ArcInstance, adj, prices, tol=1e-9):
    improving = True
    toll_idx = dict(zip(instance.toll_arcs, range(instance.n_tolls)))
    idx_to_tall = dict(zip(range(instance.n_tolls), instance.toll_arcs))
    obj, old_obj = 0, 0
    sol = np.array([prices[t] for t in instance.toll_arcs])

    while improving:
        obj = 0
        sol = np.array([prices[t] for t in instance.toll_arcs])
        old_obj = obj
        paths = {}
        commodities_tolls = np.zeros((instance.n_commodities, instance.n_tolls))
        commodity_cost = np.zeros(instance.n_commodities)

        for i, commodity in enumerate(instance.commodities):
            commodity_cost[i], _, profit, previous_vertex = dijkstra(adj, prices, commodity, tol=tol)
            obj += profit
            commodities_tolls[i] = retrive_commodity_tolls(instance, commodity, previous_vertex, prices, commodities_tolls[i], toll_idx)
            paths[i] = retrive_commodity_path(commodity, previous_vertex)
        toll_profit = commodities_tolls.sum(axis=0)
        toll_profit_idx = np.argsort(toll_profit)
        # print(obj)
        improving = False

        for toll in toll_profit_idx:
            if toll_profit[toll] > 0:
                idx = idx_to_tall[toll]
                old_value = adj[idx[0], idx[1]]
                old_price = prices[idx[0], idx[1]]
                adj[idx[0], idx[1]] = 0
                prices[idx[0], idx[1]] = 0
                cost_diff = 1000000
                for i, commodity in enumerate(instance.commodities):
                    if commodities_tolls[i, toll_idx[idx]] > 0:
                        new_cost, _, profit, previous_vertex = dijkstra(adj, prices, commodity, tol=tol)
                        if cost_diff > new_cost - commodity_cost[i] + tol:
                            cost_diff = new_cost - commodity_cost[i]
                if cost_diff > tol:
                    # print(cost_diff, 'diff')
                    adj[idx[0], idx[1]] = old_value + cost_diff
                    prices[idx[0], idx[1]] = old_price + cost_diff
                    for i, commodity in enumerate(instance.commodities):
                        if commodities_tolls[i, toll_idx[idx]] > 0:
                            commodity_cost[i] += cost_diff

                    new_obj = 0
                    for i, commodity in enumerate(instance.commodities):
                        c_p, _, p, previous_vertex = dijkstra(adj, prices, commodity, tol=tol)
                        commodities_tolls[i] = retrive_commodity_tolls(instance, commodity, previous_vertex, prices, commodities_tolls[i],
                                                                       toll_idx)
                        new_obj += p
                    print('new, ', new_obj)
                    improving = True
                else:
                    adj[idx[0], idx[1]] = old_value
                    prices[idx[0], idx[1]] = old_price

                for i, commodity in enumerate(instance.commodities):
                    new_cost, _, profit, previous_vertex = dijkstra(adj, prices, commodity, tol=tol)
                    path = retrive_commodity_path(commodity, previous_vertex)
                    # if paths[i] != path:
                    # print('\n', commodity)
                    # print(paths[i])
                    # print(path)
                    # print([adj[j[0], j[1]] for j in paths[i]], [j for j in paths[i] if j in instance.toll_arcs])
                    # print([adj[j[0], j[1]] for j in path], [j for j in path if j in instance.toll_arcs])
                    # print(sum([adj[j[0], j[1]] for j in paths[i]]), sum([adj[j[0], j[1]] for j in path]), idx_to_tall[toll])
                    paths[i] = path

                # print(instance.compute_obj(adj, prices, tol=tol), '*')

        if improving:
            if obj < old_obj:
                print(obj, old_obj, instance.compute_obj(adj, prices, tol=tol), '*')
                sol_str = ''
                for s in sol:
                    sol_str += str(s) + ', '
                print('a = np.array([' + sol_str[:-2] + '])')
                sol = np.array([prices[t] for t in instance.toll_arcs])
                sol_str = ''
                for s in sol:
                    sol_str += str(s) + ', '
                print('b = np.array([' + sol_str[:-2] + '])', '\n\n\n\n')

    # print(obj, '*')
    return sol, obj


@total_ordering
class Path:
    def __init__(self, path, prices, instance: ArcInstance, toll_to_path, commodity):
        self.path = tuple(path) if type(path) is not tuple else path
        self.instance = instance
        self.commodity = commodity
        self.toll_free_cost = sum([instance.adj[p] for p in path])
        self.tolls = tuple([p for p in path if p in instance.toll_arcs])
        self.current_cost, self.profit = None, None
        self.update(prices)
        self.tol = 1e-9

        for toll in self.tolls:
            if toll not in toll_to_path.keys():
                toll_to_path[toll] = [self]
            else:
                toll_to_path[toll].append(self)

    def update(self, prices):
        self.profit = sum([prices[t] for t in self.tolls])
        self.current_cost = self.toll_free_cost + self.profit

    def __repr__(self):
        return str(self.tolls) + ' ' + str(self.current_cost)

    def __eq__(self, other):
        if (other.current_cost > self.current_cost - self.tol and
                other.toll_free_cost < other.toll_free_cost + self.tol and self.profit == other.profit):
            return True
        else:
            return False

    def __lt__(self, other):
        if self.current_cost < other.current_cost - self.tol:
            return True
        elif self.current_cost < other.current_cost + self.tol and self.profit > other.profit:
            return True
        else:
            return False


class Com:

    def __init__(self, commodity: ArcCommodity, instance: ArcInstance, toll_to_path):
        self.c = commodity
        self.instance = instance
        self.c_od, self.c_od_path = self.compute_cod()
        self.paths = [self.compute_initial_path(toll_to_path), Path(self.c_od_path, np.zeros(1), instance, toll_to_path, self)]
        self.paths.sort()
        self.profit = self.paths[0].profit * self.c.n_users

    def compute_cod(self):
        adj = copy.deepcopy(self.instance.get_adj())
        prices = np.zeros_like(adj)
        for t in self.instance.tolls:
            prices[t.idx[0], t.idx[1]] = 100000
            adj[t.idx[0], t.idx[1]] = 100000

        res = dijkstra(adj, prices, self.c)

        return res[0], retrive_commodity_path(self.c, previous_vertex=res[3])

    def compute_initial_path(self, toll_to_path):
        adj = copy.deepcopy(self.instance.get_adj())
        prices = np.zeros_like(adj)
        res = dijkstra(adj, prices, self.c)
        return Path(retrive_commodity_path(self.c, previous_vertex=res[3]), prices, self.instance, toll_to_path, self)

    def update(self):
        self.paths.sort()
        self.profit = self.paths[0].profit * self.c.n_users

    def add_path(self, path, prices, toll_to_path):
        found_path = False
        path = tuple(path)
        for p in self.paths:
            if p.path == path:
                found_path = True
        if not found_path:
            self.paths.append(Path(path, prices, self.instance, toll_to_path, self))
            self.update()

    def __repr__(self):
        return repr(self.c)


class HeuristicNew:
    def __init__(self, instance: ArcInstance):
        self.instance = instance
        self.toll_to_path: Dict[tuple, List[Path]] = {}
        self.commodities = [Com(c, instance, self.toll_to_path) for c in instance.commodities]
        self.tol = 1e-9
        self.obj = 0

    def run(self, prices=None, obj=None):
        if prices is None:
            prices = np.zeros_like(self.instance.adj)
            self.obj = obj
        else:
            self.add_initial_solution(prices, obj)
        self.commodities.sort(key=lambda x: x.profit, reverse=True)
        # print([len(com.paths) for com in self.commodities])
        improving = True
        while improving:
            improving = False
            for commodity in self.commodities:
                for toll in commodity.paths[0].tolls:
                    diff_cost = commodity.paths[1].current_cost - commodity.paths[0].current_cost
                    if diff_cost > 0:
                        for path in self.toll_to_path[toll]:
                            if path.path == path.commodity.paths[0].path:
                                other_commodity = path.commodity
                                commodity_diff_cost = other_commodity.paths[1].current_cost - other_commodity.paths[0].current_cost
                                if commodity_diff_cost < diff_cost:
                                    diff_cost = commodity_diff_cost
                        if diff_cost > self.tol:
                            improving = True
                            prices[toll] += diff_cost
                            for path in self.toll_to_path[toll]:
                                path.update(prices)
                            for path in self.toll_to_path[toll]:
                                path.commodity.update()

        current_profit = 0
        for commodity in self.commodities:
            res = dijkstra(self.instance.adj + prices, prices, commodity.c)
            current_profit += res[2]
            commodity.add_path(retrive_commodity_path(commodity.c, res[3]), prices, self.toll_to_path)
        if current_profit > self.obj:
            self.obj = current_profit

    def compute_obj(self, prices):
        total_profit = 0
        for commodity in self.commodities:
            total_profit += dijkstra(self.instance.adj + prices, prices, commodity.c)[2]
        return total_profit

    def add_initial_solution(self, prices, obj):
        for paths in self.toll_to_path.values():
            for path in paths:
                path.update(prices)
        for commodity in self.commodities:
            res = dijkstra(self.instance.adj + prices, prices, commodity.c)
            commodity.add_path(retrive_commodity_path(commodity.c, res[3]), prices, self.toll_to_path)
            self.obj = obj
            commodity.update()


