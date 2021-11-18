from graph_utils import edge_list_to_adj_list, print_adj_list
from graph_utils import coo_to_edge_list, adj_list_to_edge_list
from graph_utils import get_zones, get_all_graph_zones
from graph_utils import visualize_partitions, visualize_all_partitions, get_second_seed

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import Linear

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx

import networkx as nx

dataset = Planetoid(root='data/Planetoid', name='Cora',
                    transform=NormalizeFeatures())
data = dataset[0]

graph = coo_to_edge_list(data.edge_index)
graph = edge_list_to_adj_list(graph)
# print_adj_list(graph)
# print(graph)

#core_zone, undiscovered_zone = get_zones(graph, 2, 4)

core_zone1, undiscovered_zone1, core_zone2, undiscovered_zone2 = get_all_graph_zones(
    graph,
    1209,
    7,
    4,
    3
)


G = to_networkx(data, to_undirected=True)
visualize_all_partitions(G, core_zone1, undiscovered_zone1,
                         core_zone2, undiscovered_zone2)

print(core_zone1)
print('----------------------------------------------------------')
print(undiscovered_zone1)
print('==================================================')
print(core_zone2)
print('----------------------------------------------------------')
print(undiscovered_zone2)
intersection_zone = set(undiscovered_zone1).intersection(
    set(undiscovered_zone2)) - set(core_zone1) - set(core_zone2)

print('********************************************')
print(intersection_zone)
print(len(intersection_zone))

print(len(core_zone1))
print(len(core_zone2))
print(len(undiscovered_zone1))
print(len(undiscovered_zone2))
