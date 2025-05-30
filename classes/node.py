from uuid import uuid4
import numpy as np

class Node:

    def __init__(self, node_id: str = None, 
                 initial_edges: list = [], 
                 initial_neighbors: list = [], 
                 initial_position: list = None,
                 initial_attrs = {}):
        
        self.id = node_id if node_id not in ["", None] else uuid4().hex
        self.edges = set(initial_edges) if initial_edges != None else set()
        self.neighbors = set(initial_neighbors) if initial_neighbors != None else set()
        self.pos = initial_position if initial_position != None else round(np.random.random(), 3), round(np.random.random(), 3)
        self.attrs = {
            # Aqui irán todos los valores por defecto de los atributos
            **initial_attrs         # Agregar o sobreescribir los parametros pasados al nodo
        }

    def get_edges(self):
        return [x for x in self.edges if x.id.startswith(self.id) ]
    
    def get_neighbors(self):
        return self.neighbors
    
    def add_neighbors(self, neighbors_list: list):
        if self.neighbors == None:
            self.neighbors = set()
        for n in neighbors_list:
            if n not in self.neighbors:
                self.neighbors.add(n)
                
    def add_neighbor(self, node):
        nodes_names = [ x.id for x in self.neighbors ]
        existing_node = node.id in nodes_names
        if not existing_node and node.id != self.id:
            self.neighbors.add(node)
    
    def get_position(self):
        return self.pos
    
    def set_edges(self, edges_list: list):
        self.edges = set(edges_list) 
        
    def set_neigbors(self, neighbors: list):
        self.neighbors = set(neighbors)
        
    def add_edge(self, new_edge):
        if new_edge not in self.edges:
            self.edges.add(new_edge)
        return new_edge
        
    def get_degree(self):
        return len(set(self.edges))
    
    def __str__(self):
        str_data = [
            "Node: {}".format(self.id),
            "Position: [{}, {}]".format(self.pos[0], self.pos[1]),
            "Total Edges: {}".format(len(self.edges)),
            "Total Neigbors: {}".format(len(self.neighbors))
        ]
        
        return "\n".join(str_data)