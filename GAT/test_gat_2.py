import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.datasets import OGB_MAG
from torch_geometric.graphgym import optim
from torch_geometric.loader import DataLoader, HGTLoader
from torch_geometric.data import HeteroData, Batch, InMemoryDataset
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch_geometric.transforms as T


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.hidden_dim = hidden_channels
        self.commodity_embedding = Linear(-1, hidden_channels)
        self.toll_embedding = Linear(-1, hidden_channels)
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=3)
        self.lin1 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x_ = x[:, 1:-1]
        mask = x[:, 0].unsqueeze(1).repeat_interleave(self.hidden_dim, -1)
        x_ = self.commodity_embedding(x_) * (1 - mask) + self.toll_embedding(x_) * mask
        x_ = x_.relu()
        x_ = self.conv1(x_, edge_index)
        x_ = self.lin1(x_).squeeze(1)
        x_ = x_ * x[:, 0]
        return x_


all_data = torch.load('Data_/data_homo.pth')
train_set = all_data[:4000]
test_set = all_data[4000:]
train_set = DataLoader(train_set, batch_size=64)
d = train_set.dataset[0]

model = GAT(hidden_channels=64, out_channels=2)
# g = torch_geometric.utils.to_networkx(d, to_undirected=True)
# nx.draw(g)
# plt.show()



optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(1000):
    loss = None
    for data in train_set:
        optimizer.zero_grad()
        output = model(data.x.float(), data.edge_index)
        y = data.y.float()
        loss = criterion(output, y)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), 1)
        optimizer.step()

    print(epoch, loss.item())

with torch.no_grad():
    for data in test_set:
        output = model(data.x.float(), data.edge_index)
        loss = criterion(output, data.y.float())
        print(loss)