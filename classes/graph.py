from uuid import uuid4
import numpy as np
import random
from classes.edge import Edge
from classes.node import Node
from utils.common import euclidean_distance

class Graph:
    GRAPH_NAMES = {
        "grid": {
            "node": "GRID_GRAPH_NODE_",
            "edge": "GRID_GRAPH_EDGE_"
        },
        "erdos": {
            "node": "ERDOS_GRAPH_NODE_",
            "edge": "ERDOS_GRAPH_EDGE_",
        },
        "gilbert": {
            "node": "GILBERT_GRAPH_NODE_",
            "edge": "GILBERT_GRAPH_EDGE_",
        },
        "geo": {
            "node": "GEO_GRAPH_NODE_",
            "edge": "GEO_GRAPH_EDGE_",
        },
        "barabasi": {
            "node": "BARABASI_GRAPH_NODE_",
            "edge": "BARABASI_GRAPH_EDGE_",
        },
        "dorogov": {
            "node": "DOROGOV_GRAPH_NODE_",
            "edge": "DOROGOV_GRAPH_EDGE_",
        }
    }
    
    def __init__(self, graph_name: str):
        self.graph_name = graph_name if graph_name else uuid4().hex
        self.nodes = {}         # Diccionario que contendrá a los nodos
        self.edges = {}         # Diccionario que contendrá la informacion de las aristas
        self.attrs = {}         # Proximas implementaciones
        self.is_directed = False
    
    def get_node_by_name(self, node_name: str):
        return self.nodes.get(node_name)
    
    def add_node(self, node_name: str):
        existing_node = self.nodes.get(node_name)
        if not existing_node:
            existing_node = Node(node_name)
            self.nodes[node_name] = existing_node
        return existing_node
    
    # TODO Agregar capacidad de pasar Nodos directamente
    def add_edge(self, edge_name: str, src_node_name: str, tgt_node_name: str):
        existing_edge = self.edges.get(edge_name)
        #print("Ya existe la arista")
        if not existing_edge:
            # La arista no existe, se debe de crear
            src_node = self.add_node(src_node_name)
            tgt_node = self.add_node(tgt_node_name)
            existing_edge = Edge(edge_name, src_node, tgt_node, is_directed=self.is_directed)
            self.edges[edge_name] = existing_edge
            src_node.add_neighbor(tgt_node)
            tgt_node.add_neighbor(src_node)
            
            src_node.add_edge(existing_edge)
            tgt_node.add_edge(existing_edge)
        
        return existing_edge    
    
    def getRandomEdge(self):
        return random.choice(list(self.edges.values()))
    
    def get_random_node(self):
        return random.choice(list(self.nodes.keys()))
    
    def create_grid_graph(self, m, n = None, use_diagonals = False):
        # Limpieza de datos
        self.nodes = {}
        self.edges = {}
        # m y n >= 2
        # si n es indefinido o invalido , n = m
        if n is None or n < 2:
            n = m
            
        m = max(2, m)
        n = max(2, n)
        
        for i in range(m):
            for j in range(n):
                node_name = self.GRAPH_NAMES["grid"]["node"] + str(i * n + j)
                added_node = self.add_node(node_name)
                added_node.pos = np.array([float(i), float(j)])
                if j < n - 1:
                    next_node_name = self.GRAPH_NAMES["grid"]["node"] + str( i * n + j + 1)
                    edge_name = "{} -> {}".format(node_name, next_node_name)
                    #print("Connecting node {} to {} via {}".format(node_name, next_node_name, edge_name))
                    self.add_edge(edge_name, node_name, next_node_name)
                if i < m - 1:
                    next_node_name = self.GRAPH_NAMES["grid"]["node"] + str((i + 1) * n + j)
                    edge_name = "{} -> {}".format(node_name, next_node_name)
                    #print("Connecting node {} to {} via {}".format(node_name, next_node_name, edge_name))
                    self.add_edge(edge_name, node_name, next_node_name)
                if i < m - 1 and j < n - 1 and use_diagonals:
                    next_node_name = self.GRAPH_NAMES["grid"]["node"] + str((i + 1) * n + j)
                    edge_name = "{} -> {}".format(node_name, next_node_name)
                    self.add_edge(edge_name, node_name, next_node_name)
                if i > 0 and j < n - 1 and use_diagonals:
                    next_node_name = self.GRAPH_NAMES["grid"]["node"] + str(((i - 1) * n + j + 1))
                    edge_name = "{} -> {}".format(node_name, next_node_name)
                    self.add_edge(edge_name, node_name, next_node_name)
 
    def create_erdos_renyi_graph(self, n = 1, m = 0):
        self.nodes = {}
        self.edges = {}
        if n < 1 or m < (n-1):
            print("Los valores de N y M son incorrectos")
            return
        
        for i in range(n):
            self.add_node(self.GRAPH_NAMES["erdos"]["node"] + str(i))

        random_tuple_values = [ (np.random.randint(0, n - 1), np.random.randint(0, n - 1)) for _ in range(m) ]
        
        for u,v in random_tuple_values:
            if u != v:
                src_node_name = self.GRAPH_NAMES["erdos"]["node"] + str(u)
                tgt_node_name = self.GRAPH_NAMES["erdos"]["node"] + str(v)
                
                edge_name = "{} -> {}".format(src_node_name, tgt_node_name)
                self.add_edge(edge_name, src_node_name, tgt_node_name)
    
    def create_gilbert_graph(self, n, p = 0.5):
        self.nodes = {}
        self.edges = {}
        for i in range(n):
            for j in range(n):
                edge_prob = np.random.rand()
                if ( edge_prob >= p or p == 1) and j != i:
                    src_node_name = self.GRAPH_NAMES["gilbert"]["node"] + str(i)
                    tgt_node_name = self.GRAPH_NAMES["gilbert"]["node"] + str(j)
                    edge_name = "{} -> {}".format(src_node_name, tgt_node_name)
                    self.add_edge(edge_name, src_node_name, tgt_node_name)
    
    def create_geo_graph(self, n = 5, rad = 0.3):
        self.nodes = {}
        self.edges = {}
        for i in range(n):
            self.add_node(self.GRAPH_NAMES["geo"]["node"] + str(i))
        nodes_list = [ x[1] for x in list(self.nodes.items()) ] 
        for i in range(n):
            for j in range(n):
                if i != j:
                    u = nodes_list[i]
                    v = nodes_list[j]
                    point_dist = euclidean_distance(u.pos, v.pos)
                    #print(point_dist)
                    if point_dist <= rad:
                        edge_name = "{} -> {}".format(u.id, v.id)
                        self.add_edge(edge_name, u.id, v.id)
    
    def create_barabasi_graph(self, n, d_max = 5):
        #Se agrega el primer nodo
        for idx in range(n):
            self.add_node(self.GRAPH_NAMES["barabasi"]["node"] + str(idx))
        for u in range(1, n):
            randomized_nodes_order = list(range(u))
            np.random.shuffle(randomized_nodes_order)
            for v in range(u):
                inner_node = self.GRAPH_NAMES["barabasi"]["node"] + str(randomized_nodes_order[v])
                #print(inner_node)
                deg = self.get_node_by_name(inner_node).get_degree()
                #print("Node: {} - {}".format(inner_node, deg))
                prob = 1 - deg / d_max
                rd = np.random.random()
                #print(rd)
                if rd <= prob and randomized_nodes_order[v] != u:
                    src_node_name = self.GRAPH_NAMES["barabasi"]["node"] + str(u)
                    tgt_node_name = self.GRAPH_NAMES["barabasi"]["node"] + str(randomized_nodes_order[v])
                    edge_name = "{} -> {}".format(src_node_name, tgt_node_name)
                    self.add_edge(edge_name, src_node_name, tgt_node_name)
    
    def create_dorogov_graph(self, n = 10):
        self.nodes = {}
        self.edges = {}
        if n < 3:
            n = 3
        initial_points = [
            [0, 1],
            [1, 2],
            [2, 0]
        ]
        for point in initial_points:
            src_node = self.GRAPH_NAMES["dorogov"]["node"] + str(point[0])
            tgt_node = self.GRAPH_NAMES["dorogov"]["node"] + str(point[1])
            edge_name = "{} -> {}".format(src_node, tgt_node)
            self.add_edge(edge_name, src_node, tgt_node)
        
        for idx in range(3, n):
            tgt_nodes: Edge = random.choice(list(self.edges.values()))
            src_node_name = self.GRAPH_NAMES["dorogov"]["node"] + str(idx)
            first_edge_name = "{} -> {}".format(src_node_name, tgt_nodes.source_node.id)
            second_edge_name = "{} -> {}".format(src_node_name, tgt_nodes.target_node.id)        
            self.add_edge(first_edge_name, src_node_name, tgt_nodes.source_node.id)
            self.add_edge(second_edge_name, src_node_name, tgt_nodes.target_node.id)
    
    def generate_graphviz_graph(self):
        graph_str = r"{} {}".format(
            "graph" if not self.is_directed else "digraph",
            self.graph_name
        )
        
        graph_str += "{\n"
        # Modificar para utilizar los nodos en vez de las aristas
        for edge in self.edges.values():
            src_node: Node = edge.source_node
            tgt_node: Node = edge.target_node
            graph_str += "{} -{} {} [weight={}];\n".format(
                src_node.id, ">" if self.is_directed else "-", tgt_node.id, edge.weight
            )
        
        graph_str += "}\n"
        
        return graph_str
                
    def export_to_graphviz_file(self, filename: str):
        graph_repr = self.generate_graphviz_graph()
        with open(filename, "w", encoding="utf-8") as gh:
            gh.write(graph_repr)
        
        