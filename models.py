import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv

class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(GCN, self).__init__()
        self.dataset = dataset
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(self.dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, self.dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSage(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(GraphSage, self).__init__()
        self.dataset = dataset
        torch.manual_seed(1234567)
        self.sage1 = SAGEConv(self.dataset.num_features, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, self.dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sage2(x, edge_index)
        return x

class ChebyshevNet(torch.nn.Module):
    '''Chebyshev NN has K (from center) parameter'''
    def __init__(self, dataset, hidden_channels, k_parameter):
        super(ChebyshevNet, self).__init__()
        self.dataset = dataset
        torch.manual_seed(1234567)
        self.cheb1 = ChebConv(self.dataset.num_features, hidden_channels, K=k_parameter)
        self.cheb2 = ChebConv(hidden_channels, self.dataset.num_classes, K=k_parameter)

    def forward(self, x, edge_index):
        x = self.cheb1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.cheb2(x, edge_index)
        return x


