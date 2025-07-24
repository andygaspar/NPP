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


def run_arc_heuristic2(instance: ArcInstance, adj, prices, tol=1e-9):
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

