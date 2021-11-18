from models import GCN
from graph_utils import ZonesPartitioner
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

''' Dataset selection '''
dataset = Planetoid(root='data/Planetoid', name='Cora',
                    transform=NormalizeFeatures())
data = dataset[0]


def get_indexes(data):
    datanum = data.num_nodes
    intersection_zone = [False] * datanum
    for i in range(datanum):
        if data.train_mask[i] == data.test_mask[i] == True:
            intersection_zone[i] = True
    return intersection_zone


print(get_indexes(data))
