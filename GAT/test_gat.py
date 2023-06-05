import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import DataLoader, HGTLoader
from torch_geometric.data import HeteroData, Batch, InMemoryDataset
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch_geometric.transforms as T


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


def to_hetero_data(data_homo):
    n_commodities = int(data_homo.x[0, 4].item())
    data_hetero = HeteroData()
    data_hetero['commodities'].x = data_homo.x[:n_commodities, 1:3]
    data_hetero['tolls'].x = data_homo.x[n_commodities:, -1]
    data_hetero['tolls'].y = data_homo.y[n_commodities:]

    comm_tolls_idxs = torch.where(data_homo.edge_index[0] < n_commodities)[0]
    from_comm = data_homo.edge_index[0][comm_tolls_idxs]
    to_tolls = data_homo.edge_index[1][comm_tolls_idxs] - n_commodities
    data_hetero['commodities', 'transfer', 'tolls'].edge_index = torch.stack([from_comm, to_tolls])
    data_hetero['commodities', 'transfer', 'tolls'].edge_attr = data_homo.edge_attr[comm_tolls_idxs]
    data_hetero = T.ToUndirected()(data_hetero)
    return data_hetero


ll = torch.load('Data_/data_homo.pth')
data_list = []
for l in ll:
    data_list.append(to_hetero_data(l))

data_loader = DataLoader(data_list)


model = GAT(hidden_channels=64, out_channels=1)

meta_data_hetero = data_loader.dataset[0].metadata()
model = to_hetero(model, meta_data_hetero, aggr='sum')


d = data_loader.dataset[0]

g = torch_geometric.utils.to_networkx(d.to_homogeneous(), to_undirected=True)
nx.draw(g)
plt.show()
print(d.edge_index_dict)
model(d.x_dict, d.edge_index_dict)

