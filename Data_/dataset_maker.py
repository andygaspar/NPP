import random

import numpy as np
import torch
from Instance.instance2 import Instance2
from Old.global_new import GlobalSolverNew

np.random.seed(0)


data_list, data_list_homo = [], []


for i in range(40_000):
    n_commodities = random.choice(range(5, 15))
    n_tolls = random.choice(range(5, 15))
    npp = Instance2(n_tolls=n_tolls, n_commodities=n_commodities, seeds=False)

    global_solver = GlobalSolverNew(npp)
    global_solver.solve()
    print(i, "obj val global", global_solver.m.objVal)
    # data_set = npp.make_torch_hetero_graph(solution=global_solver.get_prices())
    # data_list.append(data_set)
    data_set = npp.make_torch_graph(solution=global_solver.get_prices())
    data_list_homo.append(data_set)
#
# class MyDataset(InMemoryDataset):
#     def __init__(self, root, data_list, transform=None):
#         self.data_list = data_list
#         super().__init__(root, transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def processed_file_names(self):
#         return 'data_hetero2.pt'
#
#     def process(self):
#         torch.save(self.collate(self.data_list), self.processed_paths[0])
#
#
# md = MyDataset('Data_', data_list)
# md.process()
# kk = DataLoader(md)
#
#
# ll = torch.load('Data_/processed/data_hetero2.pt')
# lll = DataLoader(ll)
# data, slices = InMemoryDataset.collate(data_list)

torch.save(data_list_homo, 'Data_/data_homo2.pth')
# ll = torch.load('Data_/data_homo.pth')
# llld = DataLoader(ll)



# data_set = DataLoader(data_list)
# torch.save(data_set, 'test_dataset.pth')
#
# ddd = torch.load('test_dataset.pth')
#
#
# n_iterations = 10000
# n_particles = 128
# no_update_lim = 1000
# #
# path_costs = np.random.uniform(size=npp.n_paths*n_particles)
# init_norm_costs = np.random.uniform(size=npp.n_paths*n_particles)
#
#

# pso = PsoSolver(npp, n_particles, n_iterations, no_update_lim)
# k = pso.random_init()
#
# latin_hyper = pso.compute_latin_hypercube_init(dimensions=5)
# pso.run(init_pos=latin_hyper, stats=False, verbose=True)
#
# data_loaded = torch.load('test_dataset.pth')
#
# import torch_geometric.transforms as T
#
# data_set = T.ToUndirected()(data_set)
# data_set = T.AddSelfLoops()(data_set)
# data_set = T.NormalizeFeatures()(data_set)
#
# #From node property prediction import :
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


# download the dataset
# dataset = PygNodePropPredDataset(name='ogbn-arxiv',transform=T.ToSparseTensor())
#
# from torch_geometric.datasets import Planetoid
#
# # Import dataset from PyTorch Geometric
# dataset1 = Planetoid(root=".", name="CiteSeer")
#
# dataset1[1]





