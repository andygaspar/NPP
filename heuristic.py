import numpy as np

from Instance.instance import Instance


def improve_solution(pb: Instance, sol: np.array, tol=0.00001):
    # path_selection = {}
    # path_dict = {}
    path_dict_diff = {}
    for commodity in pb.commodities:
        costs = sol + commodity.c_p_vector
        costs = np.append(costs, commodity.c_od)
        idxs = np.argsort(costs)
        costs = np.sort(costs)
        # path_selection[commodity.name] = (idxs, costs)
        if idxs[0] < pb.n_paths:
            if idxs[0] in path_dict_diff.keys():
                # path_dict[idxs[0]].append(costs[:2])
                path_dict_diff[idxs[0]].append(costs[1] - costs[0])
            else:
                # path_dict[idxs[0]] = [costs[:2]]
                path_dict_diff[idxs[0]] = [costs[1] - costs[0]]
    path_dict_sol = {path: min(path_dict_diff[path]) - tol for path in path_dict_diff if min(path_dict_diff[path]) > tol}
    print(path_dict_sol)
    new_sol = sol.copy()
    for path in path_dict_sol.keys():
        print(path_dict_sol[path] > tol)
        new_sol[path] += path_dict_sol[path]

    return new_sol