import math
import time 
import random

def euclidean_distance(first_point: list, second_point: list):
    op_1 = (second_point[0] - first_point[0])**2
    op_2 = (second_point[1] - first_point[1])**2
    return math.sqrt(op_1 + op_2)
    
def BFS(bfs_graph, original_graph, delay = 0, start_node = None):
    layers = []
    nodes_added = {}    
    print("Starting BFS")
    node_origin = None
    if not start_node:
        node_origin = random.choice(list(original_graph.nodes.values()))
    else:
        node_origin = original_graph.get_node_by_name(start_node)
        
    #root = bfs_graph.add_node(node_origin.id)
    layers.append({ node_origin.id: node_origin })
    nodes_added[node_origin.id] = node_origin
    
    for edge in original_graph.edges.values():
        edge.attrs["bfs_visit"] = False
        
    layer_idx = 0
    while layer_idx < len(layers):
        next_layer = {}
        current_layer = layers[layer_idx]
        for node in current_layer.values():
            for edge in node.get_edges():
                m = edge.target_node if edge.source_node.id == node.id else edge.source_node
                if m.id not in next_layer and m.id not in nodes_added:
                    nn = bfs_graph.add_node(node.id)
                    mm = bfs_graph.add_node(m.id)
                    edge_name = "{}->{}".format(str(node.id), str(m.id))
                    bfs_graph.add_edge(edge_name, nn.id, mm.id)
                    edge.attrs["bfs_visit"] = True
                    next_layer[m.id] = m
                    nodes_added[m.id] = m
                    if delay:
                        time.sleep(delay)
        layer_idx += 1
        if len(next_layer) != 0:
            layers.append(next_layer)
    print("BFS Finish")
    

def DFS(dfs_graph, original_graph, delay = 0, start_node = None):
    added_nodes = {}

    print('Starting DFS - Rec')
    if not start_node:
        origin_node = random.choice(list(original_graph.nodes.values()))
    else:
        origin_node = original_graph.get_node_by_name(start_node)

    DFS_R(origin_node, dfs_graph, added_nodes, delay, 0)
    print('DFS finished.')
    
def DFS_R(initial_node, dfs_graph, added_nodes, delay= 0, layer = 0):
    added_nodes[initial_node.id] = initial_node
    
    for ed in initial_node.get_edges():
        m = ed.target_node if ed.source_node.id == initial_node.id else ed.source_node
        
        if m.id not in added_nodes:
            nn = dfs_graph.add_node(initial_node.id)
            mm = dfs_graph.add_node(m.id)
            
            edge_name = "{}->{}".format(str(nn.id), str(mm.id))
            dfs_graph.add_edge(edge_name, nn.id, mm.id)
            
            if delay:
                time.sleep(delay)
                
            DFS_R(m, dfs_graph, added_nodes, delay, layer + 1)