from collections import deque
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import random
import networkx as nx
import numpy as np
'''
This file contains Utility functions
for graph preprocessing

The main Goal is to create two graph 
from each of two cores and produce the intersection
'''

def mem_decorator(fun):
    cache = {}
    def wrapper(*args, **kwargs):
        key = args + tuple(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = fun(*args, **kwargs) 
        return cache[key]
    return wrapper

def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=2, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()


def visualize_partitions(graph, core_zone_nodes, undiscovered_zone_nodes,
                         options={"edgecolors": "tab:gray",
                                  "node_size": 40, "alpha": 0.9}
                         ):
    ''' visualize the graph's 
        core partition with green nodes
        undiscovered partition with yellow nodes
        rest of the graph with grey color    
    '''
    pos = nx.spring_layout(graph, seed=3113794652)  # positions for all nodes
    # visualization order depends on number of nodes in desceding order undiscovered > core zone > intersection zone
    nx.draw(graph, pos, node_color='gray', **options)  # default nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=undiscovered_zone_nodes,
                           node_color="yellow", **options)  # undiscovered nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=core_zone_nodes,
                           node_color="green", **options)  # core nodes
    plt.show()


def visualize_all_partitions(graph,
                             core_zone_nodes1,
                             undiscovered_zone_nodes1,
                             core_zone_nodes2,
                             undiscovered_zone_nodes2,
                             options={"edgecolors": "tab:gray",
                                      "node_size": 40, "alpha": 0.9}
                             ):
    ''' visualize the graph's all partitions 
        core partition with green nodes
        undiscovered partition with yellow nodes
        insersection zone with red nodes
        rest of the graph with grey color    
    '''
    intersection_zone = set(undiscovered_zone_nodes1).intersection(
        set(undiscovered_zone_nodes2)) - set(core_zone_nodes1) - set(core_zone_nodes2)
    pos = nx.spring_layout(graph, seed=3113794652)  # positions for all nodes
    nx.draw(graph, pos, node_color='gray', **options)  # default nodes
    # visualization order depends on number of nodes in desceding order undiscovered > core zone > intersection zone
    nx.draw_networkx_nodes(graph, pos, nodelist=undiscovered_zone_nodes1,
                           node_color="yellow", **options)  # undiscovered nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=undiscovered_zone_nodes2,
                           node_color="orange", **options)  # undiscovered nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=core_zone_nodes1,
                           node_color="green", **options)  # core nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=core_zone_nodes2,
                           node_color="blue", **options)  # core nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=intersection_zone,
                           node_color="red", **options)  # core nodes
    plt.show()


def coo_to_edge_list(coo_graph):
    '''Converts graph from coo to edge'''
    return coo_graph.t().tolist()


def edge_list_to_adj_list(edge_list):
    '''converts graph from list to adj list:
    for the future DFS'''
    graph = {}
    for i, j in edge_list:
        if i not in graph:
            graph[i] = []
        else:
            graph[i].append(j)
    return graph


def adj_list_to_edge_list(graph):
    '''convers adj graph to edge list'''
    ans = []
    for node in graph:
        for neighbour in graph[node]:
            ans.append([node, neighbour])
    return ans


def print_adj_list(adj):
    '''prints adjacency list'''
    for node in adj:
        print(node, end=': ')
        for neighbour in adj[node]:
            print(neighbour, end=', ')
        print()


def get_zones(graph, seed, core_depth=2, undiscovered_depth=2):
    ''' Utilite to creates three zone partitions starting from the seed
        the graph must be given as adj. list
        seed the start point for bfs
        uses BFS strategy
    '''
    core_zone = []  # res1
    undiscovered_zone = []  # res2

    queue = deque([seed])  # for BFS
    layer = {seed: 0}  # tracks visited nodes and depth

    while queue:
        node = queue.popleft()

        if layer[node] > undiscovered_depth + 1:
            break

        for neighbour in graph[node]:

            if neighbour not in layer:
                queue.append(neighbour)  # for the next iteration
                layer[neighbour] = layer[node] + 1

                # zone partition
                if layer[neighbour] < core_depth:
                    core_zone.append(neighbour)

                if core_depth <= layer[neighbour] <= undiscovered_depth + core_depth:
                    undiscovered_zone.append(neighbour)

    return core_zone, undiscovered_zone

def get_second_seed(graph, seed, step=10):
    '''Utilite function returns the second seed for the second
    graph starting from the first seed
    uses DFS strategy
    '''
    stack = deque([seed])
    visited = {seed: 0}  # can to track the layer
    while stack:
        node = stack.pop()
        for neighbour in graph[node]:
            if neighbour not in visited:
                stack.append(neighbour)
                visited[neighbour] = visited[node] + 1
                if visited[neighbour] >= step:
                    return neighbour  # the next seed
    return None  # out of range

def get_second_seed_bfs(graph, seed, step=10):
    '''Utilite function, returns the second seed for the second
    graph by traversing from the first graph's seed N number of steps
    uses BFS strategy
    '''
    queue = deque([seed])  # for BFS
    layer = {seed: 0}  # tracks visited nodes and depth

    while queue:
        node = queue.popleft()
        for neighbour in graph[node]:

            if neighbour not in layer:
                queue.append(neighbour)  # for the next iteration
                layer[neighbour] = layer[node] + 1
            # select random node from the last N step layer
            if layer[neighbour] > step:
                choices = []
                for k, v in layer.items():
                    if v > step:
                        choices.append(k)
                return random.choice(choices)
    print("you're out of graph")
    return None

def get_all_graph_zones(graph,
                        first_seed,
                        second_seed_step,
                        core_depth,
                        undiscovered_depth):
    '''wrapper function that returns 4 graph partitions:
    1) core zone for graph_1
    2) undiscovered zone for graph_1 
    3) core zone for graph_2
    4) undiscovered zone for graph_2    
    the intersection zone can be constructed by the client from
    set(undiscovered1).intersect(undiscovered2) - set(core1) - set(core2)
    '''
    core_zone1, undiscovered_zone1 = get_zones(graph,
                                               first_seed,
                                               core_depth,
                                               undiscovered_depth)
    try:
        second_seed = get_second_seed_bfs(graph, first_seed, second_seed_step)
        assert second_seed is not None
    except AssertionError as asm:
        print(f'The number of steps is out of graph {asm}')
        return None
    else:
        core_zone2, undiscovered_zone2 = get_zones(graph,
                                                   second_seed,
                                                   core_depth,
                                                   undiscovered_depth)
    return core_zone1, undiscovered_zone1, core_zone2, undiscovered_zone2

class MetaSingleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class PartitionFactory(metaclass = MetaSingleton):
    '''Partition Factory
    -- Constructor parameters:
        1) data with one graph
        2) first seed
        3) the distance to the second seed starting from the first
        4) core's depth
        5) undiscovered zone depth   
    '''
    def __init__(self, data, first_seed, second_seed_step, core_depth, undiscovered_depth):
        self.is_partitions = False  # partition flag
        self.data = data

        self.first_seed = first_seed
        self.second_seed_step = second_seed_step
        self.core_depth = core_depth
        self.undiscovered_depth = undiscovered_depth

        # default partition
        self.core_zone1 = None
        self.undiscovered_zone1 = None
        self.core_zone2 = None
        self.undiscovered_zone2 = None
        self.intersection_zone = None

    def __str__(self):
        '''print the partition length'''
        try:
            assert(self.is_partitions == True)
        except AssertionError as err:
            print('Error:{err} The partitions must be created before printing')

        cz1 = len(self.core_zone1)
        cz2 = len(self.core_zone2)
        uz1 = len(self.undiscovered_zone1)
        uz2 = len(self.undiscovered_zone2)
        iz = len(self.intersection_zone)
        res1 =f"\n--------------------------\n"\
            f"Partition Report:\n"\
            f"--------------------------\n"\
            f"first seed id: {self.first_seed}\n"\
            f"distance to the second seed: {self.second_seed_step}\n"\
            f"core depth: {self.core_depth}\n"\
            f"undiscovered zone's depth: {self.undiscovered_depth}\n"\
            f"--------------------------\n"

        res2 = "Node number in zones:\n"\
            f"--------------------------\n"\
            f"core zone_1: {cz1}\n"\
            f"undiscovered zone_1: {uz1}\n"\
            f"core zone_2: {cz2}\n"\
            f"undiscovered zone_2: {uz2}\n"\
            f"intersection zone: {iz}\n"\
            f"--------------------------\n"
        return res1 + res2
            
    def bool_list_util(self, zone):
        res = [False] * self.data.num_nodes
        for item in zone:
            res[item] = True
        return res

    def create_partitions(self):
        graph = coo_to_edge_list(self.data.edge_index)
        self.plotting_graph = graph.copy() # for the plot partition method
        graph = edge_list_to_adj_list(graph)
        core_zone1, undiscovered_zone1, core_zone2, undiscovered_zone2 = get_all_graph_zones(
            graph,
            self.first_seed,
            self.second_seed_step,
            self.core_depth,
            self.undiscovered_depth
        )
        self.core_zone1 = core_zone1
        self.undiscovered_zone1 = undiscovered_zone1
        self.core_zone2 = core_zone2
        self.undiscovered_zone2 = undiscovered_zone2
        self.intersection_zone = set(undiscovered_zone1).intersection(
            set(undiscovered_zone2)) - set(core_zone1) - set(core_zone2)

        self.is_partitions = True

    def get_undiscovered_zone1(self, bool_list=False):
        if bool_list:
            return self.bool_list_util(self.undiscovered_zone1)
        return self.undiscovered_zone1

    def get_undiscovered_zone2(self, bool_list=False):
        if bool_list:
            return self.bool_list_util(self.undiscovered_zone2)
        return self.undiscovered_zone

    def get_core_zone1(self, bool_list=False):
        if bool_list:
            return self.bool_list_util(self.core_zone1)
        return self.core_zone1

    def get_core_zone2(self, bool_list=False):
        if bool_list:
            return self.bool_list_util(self.core_zone2)
        return self.core_zone2

    def get_intersection_zone(self, bool_list=False):
        if bool_list:
            return self.bool_list_util(self.intersection_zone)
        return self.intersection_zone
    
    def plot_partitions(self, **options):
        try:
            assert(self.is_partitions == True)        
        except AssertionError as err:
            print('Error:{err} The partitions must be created before plotting')

        G = nx.Graph()
        G.add_edges_from(self.plotting_graph)
        visualize_all_partitions(
            G,
            self.core_zone1,
            self.undiscovered_zone1,
            self.core_zone2,
            self.undiscovered_zone2,
            **options
        )


def plot_losses(loss_list1, loss_list2, model_title, plot_style='ggplot'):
    plt.style.use(plot_style)
    plt.plot(np.arange(1, len(loss_list1)+1), loss_list1)
    plt.plot(np.arange(1, len(loss_list2)+1), loss_list2)
    plt.title(f'{model_title} model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['nn1', 'nn2'], loc='upper right')
    plt.show()

