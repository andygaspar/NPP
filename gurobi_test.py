import itertools
import time

from pathlib import Path

from Instance.instance import Instance
from Solver.global_ import GlobalSolver

#loc_size = [7, 20, 56]
#comm_size = [8, 20, 56, 90]
#toll_size = [8, 20, 56, 90]

loc_size = [50]
comm_size = [50]
toll_size = [10]

filename = './TestCases/gurobi_benchmark_times.txt'
solutionname = './TestCases/gurobi_benchmark_solution.txt'
Path(filename).touch()
Path(solutionname).touch()

for loc_dim, comm_dim, toll_dim in itertools.product(loc_size, comm_size, toll_size):

    if comm_dim >= ((loc_dim)*(loc_dim - 1) / 2):
        continue

    i = Instance(loc_dim, toll_dim, comm_dim)
    i.save_problem(folder_name=f"./TestCases/loc_{loc_dim}_comm_{comm_dim}_toll_{toll_dim}")

    solver = GlobalSolver(i, time_limit=180)

    tic = time.time()
    out = solver.solve()
    toc = time.time()

    with open(filename, 'a') as f:
        f.write(f'loc_{loc_dim}_comm_{comm_dim}_toll_{toll_dim}: elapsedTime {toc - tic}, solTime {out[0]}, objVal {out[1]}, objBound {out[2]}\n')

    with open(solutionname, 'a') as f:
        f.write(f'loc_{loc_dim}_comm_{comm_dim}_toll_{toll_dim}: solution {out[3]}\n')


