import torch
from torch import nn
from torch_geometric.nn import Linear, GATv2Conv
import torch.nn.functional as F
from GAT.network import Network


class GAT_RL(Network):
    def __init__(self, params):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(params)
        self.params['network'] = 'GAT_RL'

        hidden_channels = params['hidden_channels']

        self.hidden_dim = hidden_channels
        self.commodity_embedding = nn.Sequential(
            Linear(-1, hidden_channels),
            nn.ReLU()
        )

        self.toll_embedding = nn.Sequential(
            Linear(-1, hidden_channels),
            nn.ReLU(),
        )

        self.edge_embedding = nn.Sequential(
            Linear(-1, hidden_channels),
            nn.ReLU(),
        )

        self.heads = 3
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=True, heads=self.heads, edge_dim=hidden_channels, dropout=params['dropout'])
        self.conv2 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=True, heads=self.heads, edge_dim=hidden_channels, dropout=params['dropout'])
        self.conv3 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=True, heads=1, edge_dim=hidden_channels, concat=False, dropout=params['dropout'])

        self.lin_c1 = torch.nn.Linear(hidden_channels, hidden_channels * self.heads)
        self.lin_c2 = torch.nn.Linear(hidden_channels * self.heads, hidden_channels * self.heads)
        self.lin_c3 = torch.nn.Linear(hidden_channels * self.heads, hidden_channels)

        self.lin_t1 = torch.nn.Linear(hidden_channels, hidden_channels * self.heads)
        self.lin_t2 = torch.nn.Linear(hidden_channels * self.heads, hidden_channels * self.heads)
        self.lin_t3 = torch.nn.Linear(hidden_channels * self.heads, hidden_channels)

        self.out_layer = nn.Sequential(
            Linear(-1, hidden_channels // 2),
            nn.ReLU(),
            Linear(hidden_channels // 2, 2))

        self.scale_factor = 10
        self.features_extension = self.scale_factor * torch.linspace(0, 1, hidden_channels, device=self.device) ** 2

        self.to(self.device)

    def forward(self, x, edge_index, edge_attr=None):
        x_ = x[:, 1:-1]
        mask = x[:, 0].unsqueeze(1).repeat_interleave(self.hidden_dim, -1)

        x_comm_1 = x_[:, 0].unsqueeze(1).repeat_interleave(self.hidden_dim, -1) * self.features_extension
        x_comm_2 = x_[:, 1].unsqueeze(1).repeat_interleave(self.hidden_dim, -1) * self.features_extension
        x_toll = x_[:, 2].unsqueeze(1).repeat_interleave(self.hidden_dim, -1) * self.features_extension
        edges = edge_attr.unsqueeze(1).repeat_interleave(self.hidden_dim, -1) * self.features_extension

        x_comm = torch.hstack([x_comm_1, x_comm_2])

        x_ = self.commodity_embedding(x_comm) * (1 - mask) + self.toll_embedding(x_toll) * mask
        del x_toll
        del x_comm
        # x_ = torch.hstack([x_, self.scale_factor*torch.rand(size=x_.shape, device=self.device)])
        edges = self.edge_embedding(edges)

        mask2 = x[:, 0].unsqueeze(1).repeat_interleave(self.hidden_dim * self.heads, -1)

        x_ = F.elu(self.conv1(x_, edge_index, edge_attr=edges)
                   + self.lin_c1(x_) * (1 - mask2) + self.lin_t1(x_) * mask2)

        x_ = F.elu(self.conv2(x_, edge_index, edge_attr=edges)
                   + self.lin_c2(x_) * (1 - mask2) + self.lin_t2(x_) * mask2)

        x_ = self.conv3(x_, edge_index, edge_attr=edges) + self.lin_c3(x_) * (1 - mask) + self.lin_t3(x_) * mask
        x_ = self.out_layer(x_).squeeze(1)
        normal = torch.distributions.normal.Normal(x_[:, 0], torch.abs(x_[:, 1]) + 0.1)
        y = normal.rsample()
        y = y * x[:, 0]
        return y