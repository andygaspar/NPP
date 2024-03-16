import torch


def normalise_dataset(data_set):
    x_comm_min = []
    x_comm_max = []
    x_toll_min = []
    x_toll_max = []
    y_min = []
    y_max = []
    edge_min = []
    edge_max = []

    device = data_set[0].x.device

    for d in data_set:
        n_commodities = int((1 - d.x[:, 0]).sum())
        x_comm_min.append(d.x[:n_commodities, :].min(dim=0)[0])
        x_toll_min.append(d.x[n_commodities:, :].min(dim=0)[0])
        y_min.append(d.y[n_commodities:].min(dim=0)[0])
        edge_min.append(d.edge_attr.min(dim=0)[0])

        x_comm_max.append(d.x[:n_commodities, :].max(dim=0)[0])
        x_toll_max.append(d.x[n_commodities:, :].max(dim=0)[0])
        y_max.append(d.y[n_commodities:].max(dim=0)[0])
        edge_max.append(d.edge_attr.max(dim=0)[0])

    x_comm_min = torch.stack(x_comm_min)
    x_toll_min = torch.stack(x_toll_min)
    y_min_vect = torch.stack(y_min)
    edge_min = torch.stack(edge_min)

    x_comm_max = torch.stack(x_comm_max)
    x_toll_max = torch.stack(x_toll_max)
    y_max_vect = torch.stack(y_max)
    edge_max = torch.stack(edge_max)

    # plt.hist(y_min_vect.cpu(), bins=30)
    # plt.show()

    x_comm_min = x_comm_min.min(dim=0)[0]
    x_toll_min = x_toll_min.min(dim=0)[0]
    y_min = y_min_vect.min(dim=0)[0]
    edge_min = edge_min.min(dim=0)[0]

    x_comm_max = x_comm_max.max(dim=0)[0]
    x_toll_max = x_toll_max.max(dim=0)[0]
    y_max = y_max_vect.max(dim=0)[0]
    edge_max = edge_max.max(dim=0)[0]

    # print(y_min, y_max)

    assert (y_min != y_max)
    # y_min = y_max - 0.0001

    for i in range(x_comm_max.shape[0]):
        if x_comm_max[i] == x_comm_min[i]:
            x_comm_min[i] = x_comm_max[i] - 0.0001

    for i in range(x_comm_max.shape[0]):
        if x_toll_max[i] == x_toll_min[i]:
            x_toll_min[i] = x_toll_max[i] - 0.0001

    x_min = torch.zeros(x_comm_max.shape, device=device)
    x_max = torch.zeros(x_comm_max.shape, device=device)
    x_min[:3] = x_comm_min[:3]
    x_min[3:] = x_toll_min[3:]

    x_max[:3] = x_comm_max[:3]
    x_max[3:] = x_toll_max[3:]

    for i, d in enumerate(data_set):
        n_commodities = int((1 - d.x[:, 0]).sum())
        d.x = (d.x - x_min.repeat(d.x.shape[0], 1)) / (x_max.repeat(d.x.shape[0], 1) - x_min.repeat(d.x.shape[0], 1))
        d.x[:n_commodities, 0] = 0
        d.x[n_commodities:, 0] = 1
        d.x[:n_commodities, 3] = 0
        d.x[n_commodities:, 1] = 0
        d.x[n_commodities:, 2] = 0

        d.y = (d.y - y_min) / (y_max - y_min)
        d.y[:n_commodities] = 0
        d.edge_attr = (d.edge_attr - edge_min) / (edge_max - edge_min)
        d.y_min = y_min_vect[i]
        d.y_max = y_max_vect
