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

class Message(nn.Module):
    def __init__(self, embedding_dimension, hidden_dim, device=None):
        super(Message, self).__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(embedding_dimension * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embedding_dimension),
            nn.LeakyReLU(),
        ).to(self.device)

        self.fc.apply(init_w)

    def forward(self, x):
        return self.fc(x)


class NormalParams(nn.Module):
    def __init__(self, embedding_dimension, hidden_dim, out_dim, drop_out, device):
        self.device = device
        super(NormalParams, self).__init__()
        self.fc1 = nn.Linear(embedding_dimension, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, out_dim).to(self.device)
        self.drop_out = drop_out

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        # x = nn.functional.dropout(x, p=self.drop_out)
        q = self.fc2(x)
        return q


class EGAT(Network):
    def __init__(self, net_params, network=None):
        super().__init__()
        self.commodities_input = 4
        self.toll_inputs = 1
        self.embedding_dim, self.hidden_dim = net_params["embedding_dim"], net_params["hidden_dim"]
        self.commodities_rounds = net_params["commodities_messages"]
        self.toll_rounds = net_params["toll_messages"]
        self.num_heads = net_params["num_heads"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.commodity_message_encoder = Encoder(self.commodities_input, self.embedding_dim, self.hidden_dim,
                                                 self.device)
        self.toll_encoder = Encoder(self.toll_inputs, self.embedding_dim, self.hidden_dim, self.device)

        self.comm_toll_attention_dims = [self.embedding_dim]
        for i in range(self.commodities_rounds):
            self.comm_toll_attention_dims.append(self.comm_toll_attention_dims[-1] * self.num_heads)

        self.W_comm_toll = nn.ModuleList(
            [nn.Linear(self.comm_toll_attention_dims[i], self.comm_toll_attention_dims[i + 1], bias=False).to(self.device)
             for i in range(self.commodities_rounds)])
        self.a_comm_tolls = [nn.Parameter(torch.Tensor(1, 1, self.comm_toll_attention_dims[i + 1] * 2)).to(self.device)
                  for i in range(self.commodities_rounds)]

        self.tolls_attention_dims = [self.embedding_dim]
        for i in range(self.toll_rounds):
            self.tolls_attention_dims.append(self.tolls_attention_dims[-1] * self.num_heads)
        self.W_toll = nn.ModuleList(
            [nn.Linear(self.tolls_attention_dims[i], self.tolls_attention_dims[i + 1], bias=False).to(self.device)
             for i in range(self.toll_rounds)])

        self.a_tolls = [nn.Parameter(torch.Tensor(1, 1, self.tolls_attention_dims[i + 1] * 2)).to(self.device)
                  for i in range(self.toll_rounds)]

        # self.drop_out = net_params['drop out']
        self.drop_out = None
        self.leakyReLU = nn.LeakyReLU(0.2)

        self.message_merge = Message(self.embedding_dim, self.hidden_dim, self.device)

        self.normal_params = NormalParams(self.tolls_attention_dims[-1], self.hidden_dim, 2, self.drop_out, self.device)

        if network is not None:
            self.load_weights(network)

        self.init_params()

    def init_params(self):
        for i in range(self.commodities_rounds):
            nn.init.xavier_uniform_(self.W_comm_toll[i].weight)
            nn.init.xavier_uniform_(self.a_comm_tolls[i])

        for i in range(self.toll_rounds):
            nn.init.xavier_uniform_(self.W_toll[i].weight)
            nn.init.xavier_uniform_(self.a_tolls[i])

    def forward(self, data):
        commodity_message, toll_input, n_commodities = data
        n_tolls = toll_input.shape[1]
        commodities = self.commodity_message_encoder(commodity_message)
        tolls = self.toll_encoder(toll_input)
        batch_size = 1

        for i in range(self.commodities_rounds):
            commodities = self.W_comm_toll[i](commodities)\
                .view(batch_size, -1, self.num_heads * self.comm_toll_attention_dims[i])
            # tolls_repeated = tolls.repeat_interleave(commodities.shape[1], 1)
            tolls_repeated = tolls.repeat(1, n_commodities, 1)
            e = self.leakyReLU((self.a_comm_tolls[i] * torch.cat([commodities, tolls_repeated], dim=-1)).sum(dim=-1))
            alpha = nn.functional.softmax(e.view(-1, n_tolls, n_commodities), dim=-1).view(1, n_tolls*n_commodities, 1).repeat(1, 1, self.embedding_dim)
            m = (alpha * commodities).view(-1, n_tolls, n_commodities, self.embedding_dim)
            m = torch.sum(m, dim=-2)
            tolls = self.message_merge(torch.cat([tolls, m], dim=-1))

        for i in range(self.toll_rounds):
            z = self.W_toll[i](tolls).view(batch_size, -1, self.num_heads * self.tolls_attention_dims[i])
            tolls_i = tolls.repeat_interleave(n_tolls, 1)
            tolls_j = tolls.repeat(1, n_tolls, 1)
            e = self.leakyReLU((self.a_tolls[i] * torch.cat([tolls_i, tolls_j], dim=-1)).sum(dim=-1))
            alpha = nn.functional.softmax(e.view(-1, n_tolls, n_tolls), dim=-1)
            tolls = torch.tanh(torch.matmul(alpha, z))

        mu_sigma = self.normal_params(tolls)

        prices_distribution = torch.distributions.Normal(mu_sigma[:, :, 0], torch.abs(mu_sigma[:, :, 1]))
        prices = prices_distribution.sample((tolls.shape[0],))
        return prices

    def get_net_input(self, instance: Instance):
        comm: Commodity
        commodities_input = torch.tensor([[comm.n_users, comm.cost_free, comm.M_p[toll], comm.transfer_cost[toll]]
                                          for toll in instance.toll_paths for comm in instance.commodities]).to(
            torch.float).unsqueeze(0).to(self.device)
        toll_input = torch.tensor(instance.upper_bounds).to(torch.float).to(self.device).unsqueeze(0).unsqueeze(-1)
        return commodities_input, toll_input, instance.n_commodities

