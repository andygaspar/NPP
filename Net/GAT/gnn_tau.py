import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from Instance.commodity import Commodity
from Instance.instance import Instance
from Net.network import Network


def init_w(w):
    if type(w) == nn.Linear:
        nn.init.xavier_uniform_(w.weight)


class Encoder(nn.Module):
    def __init__(self, din, embedding_dimension, hidden_dim, device=None):
        super(Encoder, self).__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(din, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embedding_dimension),
            nn.LeakyReLU(),
        ).to(self.device)

        self.fc.apply(init_w)

    def forward(self, x):
        return self.fc(x)


class FA(nn.Module):
    def __init__(self, h_dimension, hidden_dim, drop_out, device):
        self.device = device
        super(FA, self).__init__()
        self.fc1 = nn.Linear(h_dimension, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, h_dimension).to(self.device)
        self.drop_out = drop_out

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        # x = nn.functional.dropout(x, p=self.drop_out)
        q = self.fc2(x)
        return q


class EGAT(Network):
    def __init__(self, net_params, network=None):
        super().__init__()
        self.taxa_inputs, self.internal_inputs, self.edge_inputs = \
            net_params["taxa_inputs"], net_params["internal_inputs"], net_params["edge_inputs"]
        self.embedding_dim, self.hidden_dim = net_params["embedding_dim"], net_params["hidden_dim"]
        self.rounds, self.num_heads = net_params["num_messages"], net_params["num_heads"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.taxa_encoder = Encoder(self.taxa_inputs, self.embedding_dim, self.hidden_dim, self.device)
        self.internal_encoder = Encoder(self.internal_inputs, self.embedding_dim, self.hidden_dim, self.device)
        self.edge_encoder = Encoder(self.edge_inputs, self.embedding_dim, self.hidden_dim, self.device)

        self.attention_dims = [self.embedding_dim]
        for i in range(self.rounds):
            self.attention_dims.append(self.attention_dims[-1] * self.num_heads)

        self.W = nn.ModuleList(
            [nn.Linear(self.attention_dims[i], self.attention_dims[i + 1], bias=False).to(self.device)
             for i in range(self.rounds)])
        self.W_m = nn.ModuleList(
            [nn.Linear(self.attention_dims[i], self.attention_dims[i + 1], bias=False).to(self.device)
             for i in range(self.rounds)])
        self.a = [nn.Parameter(torch.Tensor(1, 1, self.attention_dims[i + 1] * 3)).to(self.device)
                  for i in range(self.rounds)]

        # self.drop_out = net_params['drop out']
        self.drop_out = None
        self.leakyReLU = nn.LeakyReLU(0.2)

        self.fa = FA(self.attention_dims[-1], self.attention_dims[-1], self.drop_out, self.device)

        if network is not None:
            self.load_weights(network)

        self.init_params()

    def init_params(self):
        for i in range(self.rounds):
            nn.init.xavier_uniform_(self.W[i].weight)
            nn.init.xavier_uniform_(self.W_m[i].weight)
            nn.init.xavier_uniform_(self.a[i])

    def forward(self, data):
        taxa, internal, messages, curr_mask, size_mask, action_mask = data

        a = self.taxa_encoder(taxa) * size_mask[:, :, 0].unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        b = self.internal_encoder(internal) * size_mask[:, :, 1].unsqueeze(-1).expand(-1, -1, self.embedding_dim)

        h = a + b
        m_z = self.edge_encoder(messages)
        batch_size = h.shape[0]

        for i in range(self.rounds):
            z = self.W[i](h).view(batch_size, -1, self.num_heads * self.attention_dims[i])
            m_z = self.W_m[i](m_z).view(batch_size, -1, self.num_heads * self.attention_dims[i])
            e_i = z.repeat_interleave(z.shape[1], 1)
            e_j = z.repeat(1, z.shape[1], 1)
            e = self.leakyReLU((self.a[i] * torch.cat([e_i, e_j, m_z], dim=-1)).sum(dim=-1))
            alpha = nn.functional.softmax(e.view(-1, z.shape[1], z.shape[1]) * curr_mask - 9e15 * (1 - curr_mask),
                                          dim=-1)
            h = torch.tanh(torch.matmul(alpha, z))

        y_h = torch.matmul(h, h.permute(0, 2, 1)) * action_mask - 9e15 * (1 - action_mask)
        mat_size = y_h.shape
        y_hat = F.softmax(y_h.view(mat_size[0], -1), dim=-1)

        return y_hat, F.log_softmax(y_h.view(mat_size[0], -1), dim=-1)

    def get_net_input(self, instance: Instance):
        comm: Commodity
        commodities_input = torch.tensor([[comm.n_users, comm.cost_free, comm.M_p[toll], comm.transfer_cost[toll]]
                                          for toll in instance.toll_paths for comm in instance.commodities]).to(
            torch.float).to(self.device)
        toll_input = torch.tensor(instance.upper_bounds).to(torch.float).to(self.device)
        return commodities_input, toll_input
