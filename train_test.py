from models import GCN, GraphSage, ChebyshevNet
import numpy as np
import matplotlib.pyplot as plt
from graph_utils import PartitionFactory
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import classification_report 
from graph_utils import plot_losses

''' Dataset selection '''
dataset = Planetoid(root='data/Planetoid', name='Cora',
                    transform=NormalizeFeatures())
data = dataset[0]
''' Identcal neural networks'''

model1 = GraphSage(dataset, hidden_channels=16)
model2 = GraphSage(dataset, hidden_channels=16)

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=5e-4)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)
criterion1 = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.CrossEntropyLoss()

''' Zone Partitions '''

zone_partitioner = PartitionFactory(data, 1209, 7, 4, 3)  # Meta parameters
zone_partitioner.create_partitions()  # compute the partitions
train_mask_1 = torch.tensor(zone_partitioner.get_core_zone1(
    bool_list=True))  # core zone, 1st graph
test_mask_1 = torch.tensor(zone_partitioner.get_undiscovered_zone1(
    bool_list=True))  # undiscovered zone, 1st graph
train_mask_2 = torch.tensor(zone_partitioner.get_core_zone2(
    bool_list=True))  # core zone, 2nd graph
test_mask_2 = torch.tensor(zone_partitioner.get_undiscovered_zone2(
    bool_list=True))  # undiscovered zone, 2nd graph
intersection_mask = torch.tensor(zone_partitioner.get_intersection_zone(
    bool_list=True))  # intersection zone


#zone_partitioner.print_partitions()
#print(zone_partitioner)

'''Metrics'''


def train(nn_model, data, optimizer, criterion, train_mask):
    nn_model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = nn_model(data.x, data.edge_index)  # Perform a single forward pass.
    # Compute the loss solely based on the training nodes.
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test(nn_model, data, test_mask, target_names):
    nn_model.eval()
    out = nn_model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # Check against ground-truth labels.
    test_correct = pred[test_mask] == data.y[test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / test_mask.sum()
    report = classification_report(pred[intersection_mask],
        data.y[intersection_mask],
        target_names=target_names,
        zero_division=1)
    return test_acc, report

def test_intersection(nn_model1, nn_model2, data, intersection_mask, target_names):
    "The Intersection zone: tested by both neural networks"
    nn_model1.eval()
    nn_model2.eval()

    out1 = nn_model1(data.x, data.edge_index)
    out2 = nn_model2(data.x, data.edge_index)

    # get the data from the both predictions
    pred = ((out1 + out2)/2).argmax(dim=1)

    # Check against ground-truth labels.
    test_correct = pred[intersection_mask] == data.y[intersection_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / intersection_mask.sum()

    report = classification_report(pred[intersection_mask],
        data.y[intersection_mask],
        target_names=target_names,
        zero_division=1)
    return test_acc, report




loss_list1 = []
loss_list2 = []

for epoch in range(1, 150):
    loss1 = train(model1, data, optimizer1, criterion1, train_mask_1)
    loss2 = train(model2, data, optimizer2, criterion2, train_mask_2)
    loss_list1.append(loss1.item())
    loss_list2.append(loss2.item())
    print(f'Epoch: {epoch:03d}')
    print(f'NN1 Loss:{loss1:.4f}, NN2 Loss: {loss2:.4f}')

def plot_losses(loss_list1, loss_list2, model_title, plot_style='ggplot'):
    plt.style.use(plot_style)
    plt.plot(np.arange(1, len(loss_list1)+1), loss_list1)
    plt.plot(np.arange(1, len(loss_list2)+1), loss_list2)
    plt.title(f'{model_title} model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['nn1', 'nn2'], loc='upper right')
    plt.show()

plot_losses(loss_list1, loss_list2, 'GraphSage')

target_names=['c1','c2', 'c3','c4','c5','c6','c7']

test_acc1, report1 = test(model1, data, test_mask_1, target_names)
test_acc2, report2 = test(model2, data, test_mask_2, target_names)
test_acc_intersection, report_intersection = test_intersection(
model1, model2, data, intersection_mask, target_names)

'''
print(f"NN1:{test_acc1:.5}, NN2: {test_acc2:.5}, Intersection {test_acc_intersection:.5}")
print(report1)
print(report2)
print(report_intersection)
'''

'''
def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # Compute the loss solely based on the training nodes.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
'''
