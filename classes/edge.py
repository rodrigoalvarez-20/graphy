from uuid import uuid4
import numpy as np
from classes.node import Node

class Edge:
    
    def __init__(self, edge_id: str = None,
                 source_node: Node = None,
                 target_node: Node = None,
                 initial_attrs: dict = {},
                 weight: int = None,
                 is_directed: bool = False):
        
        self.id = edge_id if edge_id else uuid4().hex
        self.source_node = source_node
        self.target_node = target_node
        self.weight = weight if weight and weight >= 0 else np.random.randint(1, 20)
        self.attrs = {
            ## Atributos definidos
            **initial_attrs
        }
        self.is_directed = is_directed
        
    def get_id(self):
        return self.id
        
    def get_source_node(self):
        return self.source_node
    
    def get_target_node(self):
        return self.target_node
    
    def get_attributes(self):
        return self.attrs
    
    def set_source_node(self, new_source_node: Node):
        if new_source_node not in [self.source_node, self.target_node]:
            self.source_node = new_source_node
    
    def set_target_node(self, new_target_node: Node):
        if new_target_node not in [self.source_node, self.target_node]:
            self.target_node = new_target_node
            
    def set_attr(self, key, value):
        self.attrs[key] = value
        
    def __str__(self):
        return self.id