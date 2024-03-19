import time

import torch
from GAT import network_loader
from Instance.instance2 import Instance2
from Old.global_new import GlobalSolverNew

print('************  Available cuda **************', torch.cuda.is_available())

net, params = network_loader.load_network('GAT/Weights/2023-10-26 11:13 0.753')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('params:', params)
torch.set_printoptions(linewidth=200)

n_locations = 10
n_commodities = 8
n_tolls = 15

for _ in range(2):

    npp = Instance2(n_tolls=n_tolls, n_commodities=n_commodities, seeds=False)
    # npp.save_problem()
    # npp.show()

    t = time.time()
    global_solver = GlobalSolverNew(npp)
    global_solver.solve()

    data_set = npp.make_torch_graph(solution=global_solver.get_prices()).to(device)

    y = net(data_set.x.float(), data_set.edge_index, data_set.edge_attr)
    pass





