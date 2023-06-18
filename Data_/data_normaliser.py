import random

import numpy as np
import torch
import torch_geometric
from Instance.instance import Instance
from Instance.instance2 import Instance2
from Solver.global_new import GlobalSolverNew
from Solver.pso_solver_ import PsoSolverNew
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData, Batch, InMemoryDataset


def normalise_dataset(data_set):
    x_comm_min = []
    x_comm_max = []
    x_toll_min = []
    x_toll_max = []
    y_min = []
    y_max = []
    edge_min = []
    edge_max = []

    for d in data_set:
        n_commodities = int((1 - d.x[:, 0]).sum())
        x_comm_min.append(d.x[:n_commodities, :].min(dim=0)[0])
        x_toll_min.append(d.x[n_commodities:, :].min(dim=0)[0])
        y_min.append(d.y[n_commodities:].min(dim=0)[0])
        edge_min.append(d.edge_attr.min(dim=0)[0])

        x_comm_max.append(d.x[:n_commodities, :].max(dim=0)[0])
        x_toll_max.append(d.x[n_commodities:, :].max(dim=0)[0])
        y_max.append(d.y[n_commodities:].max(dim=0)[0])
        edge_max.append(d.edge_attr.max(dim=0)[0])

    x_comm_min = torch.stack(x_comm_min)
    x_toll_min = torch.stack(x_toll_min)
    y_min = torch.stack(y_min)
    edge_min = torch.stack(edge_min)

    x_comm_max = torch.stack(x_comm_max)
    x_toll_max = torch.stack(x_toll_max)
    y_max = torch.stack(y_max)
    edge_max = torch.stack(edge_max)

    x_comm_min = x_comm_min.min(dim=0)[0]
    x_toll_min = x_toll_min.min(dim=0)[0]
    y_min = y_min.min(dim=0)[0]
    edge_min = edge_min.min(dim=0)[0]

    x_comm_max = x_comm_max.max(dim=0)[0]
    x_toll_max = x_toll_max.max(dim=0)[0]
    y_max = y_max.max(dim=0)[0]
    edge_max = edge_max.max(dim=0)[0]

    if y_min == y_max:
        y_min = y_max - 0.0001

    for i in range(x_comm_max.shape[0]):
        if x_comm_max[i] == x_comm_min[i]:
            x_comm_min[i] = x_comm_max[i] - 0.0001

    for i in range(x_comm_max.shape[0]):
        if x_toll_max[i] == x_toll_min[i]:
            x_toll_min[i] = x_toll_max[i] - 0.0001

    x_min = torch.zeros(x_comm_max.shape)
    x_max = torch.zeros(x_comm_max.shape)
    x_min[:3] = x_comm_min[:3]
    x_min[3:] = x_toll_min[3:]

    x_max[:3] = x_comm_max[:3]
    x_max[3:] = x_toll_max[3:]

    for d in data_set:
        n_commodities = int((1 - d.x[:, 0]).sum())
        d.x = (d.x - x_min.repeat(d.x.shape[0], 1)) / (x_max.repeat(d.x.shape[0], 1) - x_min.repeat(d.x.shape[0], 1))
        d.x[:n_commodities, 0] = 0
        d.x[n_commodities:, 0] = 1
        d.x[:n_commodities, 3] = 0
        d.x[n_commodities:, 1] = 0
        d.x[n_commodities:, 2] = 0

        d.y = (d.y - y_min) / (y_max - y_min)
        d.y[:n_commodities] = 0
        d.edge_attr = (d.edge_attr - edge_min) / (edge_max - edge_min)

