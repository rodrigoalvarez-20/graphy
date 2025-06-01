import math
import time 
import random
import heapq
from classes.disjoint_set import DisjointSet
from collections import defaultdict

def euclidean_distance(first_point: list, second_point: list):
    op_1 = (second_point[0] - first_point[0])**2
    op_2 = (second_point[1] - first_point[1])**2
    return math.sqrt(op_1 + op_2)

def build_adj_matrix(base_graph):
    adj_graph = {}
    for node in base_graph.nodes:
        adj_graph[node] = []
    for edge in base_graph.edges:
        source_node = base_graph.edges[edge].source_node
        target_node = base_graph.edges[edge].target_node
        weight = base_graph.edges[edge].weight
        adj_graph[source_node.id].append((target_node.id, weight))
    return adj_graph

def build_adj_matrix_v2(edges_list):
    adj = defaultdict(list)
    for u, v, _ in edges_list:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def dfs_v2(node, visited, adj):
    visited[node] = True
    for neighbor in adj[node]:
        if not visited[neighbor]:
            dfs_v2(neighbor, visited, adj)
            
def is_graph_connected(edges, total_nodes):
    adj = build_adj_matrix_v2(edges)
    visited = [False] * total_nodes
    dfs_v2(0, visited, adj)
    return all(visited)

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
            
def dijkstra(djk_graph, original_graph, delay = 0, start_node = None):
    if not start_node:
        start_node = original_graph.get_random_node()
    print("Dijkstra start node:", start_node)
    distances = { node: float("inf") for node in original_graph.nodes }
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    while priority_queue:
        curr_dist, curr_node = heapq.heappop(priority_queue)
        node_name = "{}_{}".format(curr_node, curr_dist)
        if curr_dist > distances[curr_node]:
            continue
        for edg in original_graph.nodes[curr_node].edges:
            dest_node = edg.source_node.id if edg.source_node.id != curr_node else edg.target_node.id
            weight = original_graph.edges[edg.id].weight
            distance = curr_dist + weight
            if distance < distances[dest_node]:
                tgt_node = "{}_{}".format(dest_node, distance)
                edg_name = "{} --> {}".format(node_name, tgt_node)     
                #print(tgt_node)       
                # Eliminar si es que existe una arista o conexion con el valor anterior del camino
                posible_last_node = "_".join(tgt_node.split("_")[:-1]) # Only node Prefix
                past_nodes = [ x for x in djk_graph.nodes.keys() if "_".join(x.split("_")[:-1]) == posible_last_node ]
                #print(past_nodes)
                for pstn in past_nodes:                
                    for edg in djk_graph.nodes[pstn].edges:
                        if edg.id != edg_name and djk_graph.edges.get(edg.id):
                            del djk_graph.edges[edg.id]
                    del djk_graph.nodes[pstn]
                distances[dest_node] = distance
                heapq.heappush(priority_queue, (distance, dest_node))
                # Una vez que se encuentra la mejor distancia, se conecta el nodo actual con el nodo encontrado            
                djk_graph.add_edge(edg_name, node_name, tgt_node)
                if delay > 0:
                    time.sleep(delay)

def kruskal_direct(kruskal_graph, original_graph):
    vertices = list(original_graph.nodes.keys())
    edges = []
    for edge in original_graph.edges:
        source_node = original_graph.edges[edge].source_node.id
        target_node = original_graph.edges[edge].target_node.id
        weight =  original_graph.edges[edge].weight
        edges.append((source_node, target_node, weight))
    edges.sort(key=lambda x: x[2])
    ds = DisjointSet(vertices)
    for u, v, weight in edges:
        if ds.union(u, v):
            edge_name = "{} -> {}".format(u, v)
            added_edge = kruskal_graph.add_edge(edge_name, u, v)
            added_edge.weight = weight
            
def kruskal_inverse(kruskali_graph, original_graph):
    vertices = list(original_graph.nodes.keys())
    edges = []
    for edge in original_graph.edges:
        source_node = int(original_graph.edges[edge].source_node.id.split("_")[-1])
        target_node = int(original_graph.edges[edge].target_node.id.split("_")[-1])
        weight =  original_graph.edges[edge].weight
        edges.append((source_node, target_node, weight))
    edges.sort(key=lambda x: x[2], reverse=True)
    mst_edges = edges[:]
    for edge in edges:
        temp_edges = mst_edges[:]
        temp_edges.remove(edge)
        if is_graph_connected(temp_edges, len(vertices)):
            mst_edges.remove(edge)
    node_prefix = "_".join(original_graph.get_random_node().split("_")[:-1])
    for edg in mst_edges:
        source_node = "{}_{}".format(node_prefix, edg[0])
        target_node = "{}_{}".format(node_prefix, edg[1])
        edge_name = "{} --> {}".format(source_node, target_node)
        added_edge = kruskali_graph.add_edge(edge_name, source_node, target_node)
        added_edge.weight = edg[-1]

def prim_alg(out_graph, original_graph, starting_node):
    adj_graph = build_adj_matrix(original_graph)
    visited = set()
    min_heap = [(0, starting_node)]
    prev = {}
    while min_heap and len(visited) < len(adj_graph):
        weight, u = heapq.heappop(min_heap)
        if u in visited:
            continue
        visited.add(u)
        if weight != 0:
            edge_name = "{} --> {}".format(prev[u], u)
            added_edge = out_graph.add_edge(edge_name, prev[u], u)
            added_edge.weight = weight
        for v, w in adj_graph[u]:
            if v not in visited:
                heapq.heappush(min_heap, (w, v))
                prev[v] = u
                