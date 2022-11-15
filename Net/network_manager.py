import json

import torch
from torch import nn

from Net.GAT.gnn_tau import EGAT


def cross_entropy(output, data):
    adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, taus, tau_maks, y = data
    loss = nn.CrossEntropyLoss()
    return loss(output, y)


nets_dict = {

    'GAT': EGAT,
}

criterion_dict = {
    'cross': cross_entropy,
}


class NetworkManager:

    def __init__(self, folder, file=None, training=False):
        self.folder = folder
        self.file = file
        self.path = 'Net/' + folder + '/'
        if self.file is not None:
            self.path += self.file + '/'
        self.standings = []
        if training:
            file = open('.current_run.txt.swp', 'w')
            file.close()

        with open(self.path + 'params.json', 'r') as json_file:
            self.params = json.load(json_file)

        self.train_params, self.net_params = self.params["train"], self.params["net"]

    def make_network(self):
        self.print_info()
        dgn = nets_dict[self.folder](net_params=self.net_params)

        return dgn

    def get_network(self):
        if self.file is not None:
            self.print_info()
            dgn = nets_dict[self.folder](net_params=self.net_params, network=self.path + "weights.pt")
            return dgn
        else:
            return None

    def get_params(self):
        return self.params

    @staticmethod
    def compute_loss(criterion, output, data):
        # print(sum(output[output > 0.9]))
        # loss = criterion(output, y.float())
        return criterion_dict[criterion](output, data)

    def print_info(self):
        print("Training")
        for key in self.train_params:
            print(key + ':', self.train_params[key])
        print("Network")
        for key in self.net_params:
            print(key + ':', self.net_params[key])
        if 'comment' in list(self.params.keys()):
            print('comment:', self.params['comment'])

    def write_standings(self):
        file = open('.current_run.txt.swp', 'a')
        for line in self.standings:
            s = ""
            for el in line:
                s += " " + el if type(el) == str else " " + str(el)
            file.write(s + "\n")
        file.close()
        self.standings = []

