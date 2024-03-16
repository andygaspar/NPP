import abc
import copy
import json
import os

from torch import nn
import torch
import datetime


class Network(nn.Module):
    def __init__(self, params: dict):
        super(Network, self).__init__()
        self.params = params
        self.parent_dir = os.getcwd() + "/GAT/Weights"
        self.initial_folder = os.path.join(self.parent_dir, str(datetime.datetime.now())[:16])
        self.path = self.initial_folder
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.best_test_loss = 10**8
        self.best_training_loss = 10 ** 8

        self.max_not_improve_epochs = params['reset_weights_no_improve']
        self.not_improving_counter = 0

    def predict(self, x):
        with torch.no_grad():
            state: torch.Tensor
            return self.forward(x.unsqueeze(0))

    def forward_no_grad(self, x):
        with torch.no_grad():
            state: torch.Tensor
            return self.forward(x)

    def take_weights(self, model_network):
        self.load_state_dict(copy.deepcopy(model_network.state_dict()))

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))

    def update_checkpoint(self, test_loss, train_loss, cycle, epoch):
        self.best_test_loss = test_loss
        self.params["test_loss"] = self.best_test_loss
        self.params['train_loss'] = train_loss
        self.params["best_cycle"] = cycle
        self.params["best_epoch"] = epoch
        new_folder_name = self.initial_folder + ' ' + str(self.best_test_loss)[:5]
        os.rename(self.path, new_folder_name)
        self.path = new_folder_name
        with open(self.path + '/params.json', 'w') as outfile:
            json.dump(self.params, outfile, indent=2)
        torch.save(self.state_dict(), self.path + '/weights.pt')

    def not_improving_counter_update(self):
        self.not_improving_counter += 1
        if self.not_improving_counter > self.max_not_improve_epochs:
            print('Reached', self.max_not_improve_epochs, 'non improving epochs. \nResetting parameters to previous best test loss',
                  self.best_test_loss)
            self.load_weights(self.path + '/weights.pt')
            self.not_improving_counter = 0

    # def save_model(self, filename: str):
    #     model = self.vgg16(pretrained=True)
    #     torch.save(model.state_dict(), filename + '.pt')
