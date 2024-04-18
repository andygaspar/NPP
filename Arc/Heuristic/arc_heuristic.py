import numpy as np

from Arc.ArcInstance.arc_commodity import ArcCommodity
from Arc.ArcInstance.arc_instance import ArcInstance


def min_dist(dist, profit, visited, tol):
    min_val = 1000
    min_index = 0
    max_profit = 0
    for i in range(dist.shape[0]):
        if not visited[i] and dist[i] <= min_val + tol:
            if dist[i] < min_val - tol:
                min_val = dist[i]
                min_index = i
                max_profit = profit[i]
            elif profit[i] > max_profit:
                min_val = dist[i]
                min_index = i
                max_profit = profit[i]

    return min_index


def dijkstra(adj, prices, commodity: ArcCommodity, tol=1e-9):
    MAX_DIST = 1000000
    dist = np.ones(adj.shape[0]) * MAX_DIST
    visited = np.zeros_like(dist, dtype=bool)
    profit = np.zeros_like(dist)

    dist[commodity.origin] = 0
    previous_vertex = np.zeros(adj.shape[0], dtype=int)

    for i in range(adj.shape[0] - 1):
        idx = min_dist(dist, profit, visited, tol)
        visited[idx] = True
        for j in range(adj.shape[0]):
            if not visited[j] and adj[idx, j] > 0 and dist[idx] != MAX_DIST and dist[idx] + adj[idx, j] <= dist[j] + tol:
                if dist[idx] + adj[idx, j] < dist[j] - tol:
                    dist[j] = dist[idx] + adj[idx, j]
                    profit[j] = profit[idx] + prices[idx, j] * commodity.n_users
                    previous_vertex[j] = idx
                elif profit[idx] + prices[idx, j] * commodity.n_users > profit[j]:
                    dist[j] = dist[idx] + adj[idx, j]
                    profit[j] = profit[idx] + prices[idx, j] * commodity.n_users
                    previous_vertex[j] = idx

    return dist[commodity.destination], visited, profit[commodity.destination], previous_vertex


def run_arc_heuristic(instance: ArcInstance, adj, prices, tol=1e-9):
    improving = True
    toll_idx = dict(zip(instance.toll_arcs, range(instance.n_tolls)))
    obj = 0
    while improving:
        obj = 0
        commodities_tolls = np.zeros((instance.n_commodities, instance.n_tolls))
        commodity_cost = np.zeros(instance.n_commodities)

        best_toll = None
        best_profit = 0
        for i, commodity in enumerate(instance.commodities):
            commodity_cost[i], _, profit, previous_vertex = dijkstra(adj, prices, commodity, tol=tol)
            obj += profit
            current_node = commodity.destination
            while current_node != commodity.origin:
                if (previous_vertex[current_node], current_node) in instance.toll_arcs:
                    commodities_tolls[i, toll_idx[previous_vertex[current_node], current_node]] += (
                            prices[previous_vertex[current_node], current_node] * commodity.n_users)
                    if commodities_tolls[i, toll_idx[previous_vertex[current_node], current_node]] > best_profit:
                        best_toll = (previous_vertex[current_node], current_node)
                        best_profit = commodities_tolls[i, toll_idx[previous_vertex[current_node], current_node]]
                current_node = previous_vertex[current_node]

        # print(obj)

        if best_toll is not None:
            old_value = adj[best_toll[0], best_toll[1]]
            old_price = prices[best_toll[0], best_toll[1]]
            adj[best_toll[0], best_toll[1]] = 0
            prices[best_toll[0], best_toll[1]] = 0
            cost_diff = 1000000
            improving = False
            for i, commodity in enumerate(instance.commodities):
                if commodities_tolls[i, toll_idx[best_toll]] > 0:
                    new_cost, _, profit, previous_vertex = dijkstra(adj, prices, commodity, tol=tol)
                    if cost_diff > new_cost - commodity_cost[i] + tol:
                        cost_diff = new_cost - commodity_cost[i]
            if cost_diff > tol:
                adj[best_toll[0], best_toll[1]] = old_value + cost_diff
                prices[best_toll[0], best_toll[1]] = old_price + cost_diff
                improving = True
            else:
                adj[best_toll[0], best_toll[1]] = old_value
                prices[best_toll[0], best_toll[1]] = old_price
        else:
            improving = False
    sol = np.array([prices[t] for t in instance.toll_arcs])
    return sol, obj
