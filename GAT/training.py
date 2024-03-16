import json
import time

import torch
from torch_geometric.graphgym import optim
from torch_geometric.loader import DataLoader
from Data_.data_normaliser import normalise_dataset
from GAT.gat1 import GAT
from GAT.network import Network

print('************  Available cuda **************', torch.cuda.is_available())


def training(cycle, model: Network, train_set: DataLoader, test_set: DataLoader,criterion, epochs: int, optimizer,
             input_scale_factor: float = 1, clip_gradient: bool = False, verbose: int = None, scheduler=None):
    model.train()
    for epoch in range(epochs):
        loss_list = []
        for data in train_set:
            optimizer.zero_grad()
            output = model(data.x.float(), data.edge_index, data.edge_attr)
            y = data.y.float() * input_scale_factor
            # y_min = out.min(dim=0)[0]
            # y = (y - y_min) / (y_max - y_min)
            loss = criterion(output, y)

            loss.backward()
            loss_list.append(loss.item())
            if clip_gradient:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001, norm_type=float('inf'))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        train_loss = sum(loss_list) / len(loss_list)

        with torch.no_grad():
            model.eval()
            loss_list = []
            for data in test_set:
                output = model(data.x.float(), data.edge_index, data.edge_attr)
                y = data.y.float() * input_scale_factor
                loss = criterion(output, y)
                loss_list.append(loss.item())

            test_loss = sum(loss_list) / len(loss_list)
            if test_loss < model.best_test_loss:
                model.update_checkpoint(test_loss, train_loss, cycle, epoch)
            else:
                model.not_improving_counter_update()
            if verbose is not None and epoch % verbose == 0:
                print('lr', '%.6f' % scheduler.get_last_lr()[0], '   epoch:', epoch, ' train loss:',
                      train_loss, '   test loss:', test_loss)

                # print(y[:23])
                # print(output[:23])
                # print('\n')


with open('params.json') as json_file:
    params = json.load(json_file)


print('params:', params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_printoptions(linewidth=200)

all_data = torch.load('Data_/' + params['dataset'], map_location=device)


normalise_dataset(all_data)

split = len(all_data) * 3 // 4
train_set = all_data[:split]
test_set = all_data[split:]

print('dataset size', len(all_data), '  training', len(train_set), '  test', len(test_set))
train_set = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
test_set = DataLoader(test_set, batch_size=128, shuffle=True)

net = GAT(params=params)
# init_weights(net)
# g = torch_geometric.utils.to_networkx(d, to_undirected=True)
# nx.draw(g)
# plt.show()

opt = optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=params['lb_lr'], max_lr=params['ub_lr'], cycle_momentum=False)

# scheduler = StepLR(optimizer = optimizer, step_size=50, gamma=0.9)
loss_fun = torch.nn.MSELoss()

INPUT_SCALE_FACTOR = params['INPUT_SCALE_FACTOR']

for cycle in range(params['cycles']):
    print('\n\nCycle:', cycle)
    training(cycle=cycle, model=net, train_set=train_set, test_set=test_set, criterion=loss_fun, epochs=params['cycle_epochs'],
             input_scale_factor=INPUT_SCALE_FACTOR, optimizer=opt, verbose=10, scheduler=scheduler)

