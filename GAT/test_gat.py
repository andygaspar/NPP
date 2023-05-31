import numpy as np
import torch
import torch_geometric
from Instance.instance import Instance
from Instance.instance2 import Instance2
from Solver.global_new import GlobalSolverNew
from Solver.pso_solver_ import PsoSolverNew
from torch_geometric.loader import DataLoader, HGTLoader
from torch_geometric.data import HeteroData, Batch, InMemoryDataset


def to_hetero(data_homo):
    n_commodities = int(data_homo.x[0, 4].item())
    data_hetero = HeteroData()
    data_hetero['commodities'].x = data_homo.x[:8, 1:3]
    data_hetero['tolls'].x = data_homo.x[n_commodities:, -1]
    data_hetero['tolls'].y = data_homo.y[n_commodities:]

    comm_tolls_idxs = torch.where(data_homo.edge_index[0] < n_commodities)[0]
    from_comm = data_homo.edge_index[0][comm_tolls_idxs]
    to_tolls = data_homo.edge_index[1][comm_tolls_idxs] - n_commodities
    data_hetero['commodities', 'transfer', 'tolls'].edge_index = torch.stack([from_comm, to_tolls])
    data_hetero['commodities', 'transfer', 'tolls'].edge_attr = data_homo.edge_attr[comm_tolls_idxs]
    return data_hetero

ll = torch.load('Data_/data_homo.pth')
data_list = []
for l in ll:
    data_list.append(to_hetero(l))

data_loader = DataLoader(data_list)

l = 0

