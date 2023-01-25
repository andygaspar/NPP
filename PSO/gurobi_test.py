import itertools
import time

from pathlib import Path
import pandas as pd
import numpy as np

from Instance.instance import Instance
from Solver.global_ import GlobalSolver

#loc_size = [7, 20, 56]
#comm_size = [8, 20, 56, 90]
#toll_size = [8, 20, 56, 90]

loc_size = [12,15]
comm_size = [15, 20]
toll_size = [8, 12]

time_stamps = np.array([0.5, 1, 2, 3, 4, 5])*60 # minutes


filename = 'gurobi_benchmark_data'
solutionname = 'gurobi_benchmark_solution.txt'
Path(filename).touch()
Path(solutionname).touch()

loc_dims, comm_dims, toll_dims, constr_time, sol_time, obj_vals, obj_bounds, partial_objs, status = \
    [], [], [], [], [], [], [], [], []

for loc_dim, comm_dim, toll_dim in itertools.product(loc_size, comm_size, toll_size):

    if comm_dim >= ((loc_dim)*(loc_dim - 1) / 2):
        continue

    i = Instance(loc_dim, toll_dim, comm_dim)
    i.save_problem(folder_name=f"../TestCases/loc_{loc_dim}_comm_{comm_dim}_toll_{toll_dim}")

    solver = GlobalSolver(i, time_stamps=time_stamps)

    final_obj, obj_bound, solution = solver.solve()
    loc_dims.append(loc_dim)
    comm_dims.append(comm_dim)
    toll_dims.append(toll_dim)
    constr_time.append(solver.constraint_setting_time)
    sol_time.append(solver.solving_time)
    obj_vals.append(final_obj)
    obj_bounds.append(obj_bound)
    partial_objs.append(solver.partial_obj_vals)
    status.append(True if solver.m.status == 2 else False)


    # with open(filename, 'a') as f:
    #     f.write(f'loc_{loc_dim}_comm_{comm_dim}_toll_{toll_dim}: elapsedTime {toc - tic}, solTime {out[0]}, objVal {out[1]}, objBound {out[2]}\n')
    #
    # with open(solutionname, 'a') as f:
    #     f.write(f'loc_{loc_dim}_comm_{comm_dim}_toll_{toll_dim}: solution {out[3]}\n')

partial_objs = np.array(partial_objs)

df = pd.DataFrame({'loc_dim': loc_dims, 'comm_dims': comm_dims, 'toll_dims': toll_dims, 'constr_time': constr_time,
                   'sol_time': sol_time, 'is_optimal': status, 'final_obj': obj_vals, 'obj_bounds': obj_bounds})

for i, time_stamp in enumerate(time_stamps):
    df['time_'+str(time_stamp)] = partial_objs[:, i]

df.to_csv('../TestCases/'+filename+'.csv', index_label=False, index=False)


