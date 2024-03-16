import json

from GAT.gat1 import GAT
from GAT.gat_rl import GAT_RL

networks = {
    'GAT': GAT,
    'GAT_RL': GAT_RL
}


def load_network(folder):
    with open(folder + '/params.json') as json_file:
        params = json.load(json_file)
    net = networks[params['network']](params)
    net.load_weights(folder + '/weights.pt')
    return net, params
