import itertools
import time

from pathlib import Path

from Instance.instance import Instance
from Solver.global_ import GlobalSolver

#comm_size = [8, 20, 56]
#toll_size = [8, 20, 56]
#loc_size = [5, 20, 60]

loc_size = [7, 20, 56]
comm_size = [8, 20, 56, 90]
toll_size = [8, 20, 56, 90]

filename = 'gurobi_benchmark_times.txt'
Path(filename).touch()

for loc_dim, comm_dim, toll_dim in itertools.product(loc_size, comm_size, toll_size):

    if comm_dim >= ((loc_dim)*(loc_dim - 1) / 2):
        continue

    i = Instance(loc_dim, toll_dim, comm_dim)
    i.save_problem(folder_name=f"TestCases/loc_{loc_dim}_comm_{comm_dim}_toll_{toll_dim}")

    solver = GlobalSolver(i, time_limit=7200)

    tic = time.time()
    out = solver.solve()
    toc = time.time()

    with open(filename, 'a') as f:
        f.write(f'loc_{loc_dim}_comm_{comm_dim}_toll_{toll_dim}: elapsedTime {toc - tic}, solTime {out[0]}, objVal {out[1]}, objBound {out[2]}\n')



