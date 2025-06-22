import math
import numpy as np

WIDTH, HEIGHT = 1000, 700
AREA = WIDTH * HEIGHT
REPULSION_COEFF = 5.0
K = math.sqrt(AREA / 100)  # constante ideal
THETA = 0.5  # umbral Barnes-Hut

class QuadTree:
    def __init__(self, x, y, w, h, depth=0):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.children = []
        self.nodes = []
        self.center_of_mass = np.zeros(2)
        self.total_mass = 0
        self.depth = depth

    def insert(self, node):
        if self.children:
            self._insert_child(node)
        else:
            self.nodes.append(node)
            if len(self.nodes) > 1 and self.depth < 10:
                self._subdivide()
                for n in self.nodes:
                    self._insert_child(n)
                self.nodes = []
            else:
                self._update_mass_center(node)

    def _insert_child(self, node):
        for child in self.children:
            if child.contains(node):
                child.insert(node)
                break
        self._update_mass_center(node)

    def _update_mass_center(self, node):
        self.center_of_mass = (self.center_of_mass * self.total_mass + node.pos) / (self.total_mass + 1)
        self.total_mass += 1

    def _subdivide(self):
        hw, hh = self.w / 2, self.h / 2
        for dx in [0, hw]:
            for dy in [0, hh]:
                self.children.append(
                    QuadTree(self.x + dx, self.y + dy, hw, hh, self.depth + 1)
                )

    def contains(self, node):
        x, y = node.pos
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def compute_repulsion(self, node, theta=THETA):
        force = np.zeros(2)
        if self.total_mass == 0 or (len(self.nodes) == 1 and self.nodes[0] is node):
            return force

        dx, dy = self.center_of_mass - node.pos
        dist = math.hypot(dx, dy) + 1e-6
        if (self.w / dist < theta and not self.children) or len(self.nodes) == 1:
            magnitude = REPULSION_COEFF * (K * K) / dist
            force += (np.array([dx, dy]) / dist) * magnitude
        else:
            for child in self.children:
                force += child.compute_repulsion(node, theta)
        return force