import networkx as nx
import numpy as np
import torch
from torch import nn
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.datasets import OGB_MAG
from torch_geometric.graphgym import optim
from torch_geometric.graphgym.init import init_weights
from torch_geometric.loader import DataLoader, HGTLoader
from torch_geometric.nn import GATConv, Linear, Sequential, to_hetero
import torch.nn.functional as F
import torch_geometric.transforms as T
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.hidden_dim = hidden_channels
        self.commodity_embedding = nn.Sequential(
            Linear(-1, hidden_channels//2),
            nn.ReLU(),
            Linear(hidden_channels//2, hidden_channels))

        self.toll_embedding = nn.Sequential(
            Linear(-1, hidden_channels//2),
            nn.ReLU(),
            Linear(hidden_channels//2, hidden_channels))
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=3, edge_dim=1)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=3, edge_dim=1)
        self.conv3 = GATConv((-1, -1), out_channels, add_self_loops=False, heads=1, concat=False)

    def forward(self, x, edge_index, edge_attr=None, max_val=1):

        x_ = x[:, 1:-1]/max_val
        mask = x[:, 0].unsqueeze(1).repeat_interleave(self.hidden_dim, -1)
        x_ = self.commodity_embedding(x_[:, :2]) * (1 - mask) + self.toll_embedding(x_[:, 2].view(-1, 1)) * mask

        x_ = self.conv1(x_, edge_index, edge_attr=edge_attr/max_val)
        x_ = F.elu(self.conv2(x_, edge_index, edge_attr=edge_attr / max_val))
        x_ = self.conv3(x_, edge_index, edge_attr=edge_attr/max_val).squeeze(1)
        x_ = x_ * x[:, 0]
        return x_ * max_val


all_data = torch.load('Data_/data_homo.pth')
all_data = [d.to('cuda:0', non_blocking=True) for d in all_data]

max_val_ = 0
for d in all_data:
    mv = max([d.x.max(), d.edge_attr.max()])

    if mv > max_val_:
        max_val_ = mv


split = len(all_data)*3//4
train_set = all_data[:split]
test_set = all_data[split:]
train_set = DataLoader(train_set, batch_size=8, shuffle=True)
test_set = DataLoader(test_set, batch_size=64, shuffle=True)

d = train_set.dataset[0]

model = GAT(hidden_channels=64, out_channels=1)
init_weights(model)
model.to(device)
# g = torch_geometric.utils.to_networkx(d, to_undirected=True)
# nx.draw(g)
# plt.show()

optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(1000):
    loss = None
    for data in train_set:
        optimizer.zero_grad()
        output = model(data.x.float(), data.edge_index, data.edge_attr, max_val_)
        y = data.y.float()
        loss = criterion(output, y)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), 1)
        optimizer.step()
    if epoch% 50 == 0:
        with torch.no_grad():
            out = model(d.x.float(), d.edge_index, d.edge_attr, max_val_)
            print(out)
            print(d.y)

    print(epoch, loss.item())

with torch.no_grad():
    for data in test_set:
        output = model(data.x.float(), data.edge_index, max_val_)
        loss = criterion(output, data.y.float())
        print(loss)