import numpy as np

from Instance.instance import Instance


def improve_solution(pb: Instance, sol: np.array, obj_val, tol=0.0000000000001):
    improving = True
    while improving:
        new_sol = sol.copy()
        prices = np.append(new_sol, [0])
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
            if idxs[0] < pb.n_paths:
                if idxs[0] in path_dict_diff.keys():
                    path_dict_diff[idxs[0]].append(costs[1] - costs[0])
                else:
                    path_dict_diff[idxs[0]] = [costs[1] - costs[0]]
        path_dict_sol = {path: min(path_dict_diff[path]) - tol for path in path_dict_diff if
                         min(path_dict_diff[path]) > tol}

        for path in path_dict_sol.keys():
            new_sol[path] += path_dict_sol[path]

        new_val = pb.compute_solution_value(new_sol)
        if new_val > obj_val:
            sol = new_sol
            obj_val = new_val
        else:
            improving = False

    return sol
