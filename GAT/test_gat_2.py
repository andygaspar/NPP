import random

import networkx as nx
import numpy as np
import torch
from torch import nn
import torch_geometric
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch_geometric.datasets import OGB_MAG
from torch_geometric.graphgym import optim
from torch_geometric.graphgym.init import init_weights
from torch_geometric.loader import DataLoader, HGTLoader
from torch_geometric.nn import GATConv, Linear, Sequential, to_hetero, GATv2Conv
import torch.nn.functional as F
import torch_geometric.transforms as T

from Data_.data_normaliser import normalise_dataset
from GAT.gat1 import GAT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(linewidth=200)

all_data = torch.load('Data_/data_homo.pth', map_location=device)

normalise_dataset(all_data)

split = len(all_data) * 3 // 4
train_set = all_data[:split]
test_set = all_data[split:]
train_set = DataLoader(train_set, batch_size=128, shuffle=True)
test_set = DataLoader(test_set, batch_size=128, shuffle=True)

model = GAT(hidden_channels=128, out_channels=1)
# init_weights(model)
# g = torch_geometric.utils.to_networkx(d, to_undirected=True)
# nx.draw(g)
# plt.show()

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
# scheduler = StepLR(optimizer = optimizer, step_size=50, gamma=0.9)
criterion = torch.nn.MSELoss()
loss = None
for epoch in range(50000):
    for data in train_set:
        optimizer.zero_grad()
        output = model(data.x.float(), data.edge_index, data.edge_attr)
        y = data.y.float() * 10
        # y_min = out.min(dim=0)[0]
        # y = (y - y_min) / (y_max - y_min)
        loss = criterion(output, y)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), 1)
        optimizer.step()
    if epoch % 1000 == 0:
        with torch.no_grad():
            for d in all_data[:5]:
                # d = train_set.dataset[random.choice(range(len(train_set.dataset)))]
                # d = train_set.dataset[0]
                # print('lr', scheduler.get_last_lr())
                out = model(d.x.float(), d.edge_index, d.edge_attr)
                print(out[10:])
                print(d.y[10:] * 10, '\n')
            for data in test_set:
                output = model(data.x.float(), data.edge_index, data.edge_attr)
                y = data.y.float() * 10
                loss = criterion(output, y)
                print('test loss', loss)
            for d in all_data[-5:]:
                # d = train_set.dataset[random.choice(range(len(train_set.dataset)))]
                # d = train_set.dataset[0]
                # print('lr', scheduler.get_last_lr())
                out = model(d.x.float(), d.edge_index, d.edge_attr)
                idxs = d.y > 0
                print(out[idxs])
                print(d.y[idxs] * 10, '\n')

    # scheduler.step()
    if epoch % 100 == 0:
        print(epoch, loss.item())

with torch.no_grad():
    for data in test_set:
        output = model(data.x.float(), data.edge_index, data.edge_attr)
        loss = criterion(output, data.y.float())
        print(loss)
