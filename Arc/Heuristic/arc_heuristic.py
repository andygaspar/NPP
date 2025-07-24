import numpy as np

from Arc.ArcInstance.arc_instance import ArcInstance, ArcCommodity


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


def retrive_commodity_path(instance: ArcInstance, commodity: ArcCommodity, previous_vertex, prices, commodity_tolls, toll_idx,
                           best_profit, best_toll):
    current_node = commodity.destination
    path = [commodity.destination]
    while current_node != commodity.origin:
        if (previous_vertex[current_node], current_node) in instance.toll_arcs:
            commodity_tolls[toll_idx[previous_vertex[current_node], current_node]] += (
                    prices[previous_vertex[current_node], current_node] * commodity.n_users)
            if commodity_tolls[toll_idx[previous_vertex[current_node], current_node]] > best_profit:
                best_toll = (previous_vertex[current_node], current_node)
                best_profit = commodity_tolls[toll_idx[previous_vertex[current_node], current_node]]
        current_node = previous_vertex[current_node]
        path.append(current_node)
    return best_profit, best_toll, commodity_tolls, path


def run_arc_heuristic(instance: ArcInstance, adj, prices, tol=1e-9):
    improving = True
    toll_idx = dict(zip(instance.toll_arcs, range(instance.n_tolls)))
    obj, old_obj = 0, 0
    sol = np.array([prices[t] for t in instance.toll_arcs])

    while improving:
        obj = 0
        sol = np.array([prices[t] for t in instance.toll_arcs])
        old_obj = obj
        commodities_tolls = np.zeros((instance.n_commodities, instance.n_tolls))
        commodity_cost = np.zeros(instance.n_commodities)

        best_toll = None
        best_profit = 0

        for i, commodity in enumerate(instance.commodities):
            commodity_cost[i], _, profit, previous_vertex = dijkstra(adj, prices, commodity, tol=tol)
            obj += profit
            best_profit, best_toll, commodities_tolls[i], path = retrive_commodity_path(instance, commodity, previous_vertex, prices,
                                                                                        commodities_tolls[i],
                                                                                        toll_idx, best_profit, best_toll)

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
                # print(obj, cost_diff, '*')
                adj[best_toll[0], best_toll[1]] = old_value + cost_diff
                prices[best_toll[0], best_toll[1]] = old_price + cost_diff
                improving = True
            else:
                adj[best_toll[0], best_toll[1]] = old_value
                prices[best_toll[0], best_toll[1]] = old_price
        else:
            improving = False

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

