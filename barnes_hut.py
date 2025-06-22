import pygame
import math
import numpy as np
import threading
import time
from classes.quad_tree import QuadTree
from classes.graph import Graph
from classes.node import Node
from classes.edge import Edge

# Constantes y parámetros globales
WIDTH, HEIGHT = 1000, 700
NODE_RADIUS = 10
AREA = WIDTH * HEIGHT

# Parámetros dinámicos
FRICTION = 0.9
ATTRACTION_COEFF = 0.1
REPULSION_COEFF = 1.0
K = math.sqrt(AREA / 100)
VELOCITY_SCALE = 8.0

# Sincronización
lock = threading.Lock()
simulation_running = True

def simulate_thread(nodes, edges):
    global simulation_running
    while simulation_running:
        with lock:
            simulate_forces(nodes, edges)
        time.sleep(0.001)  # Evita usar 100% de CPU

def simulate_forces(nodes: list[Node], edges: list[Edge]):
    qt = QuadTree(0, 0, WIDTH - 220, HEIGHT)
    for node in nodes:
        qt.insert(node)

    for node in nodes:
        node.attrs["disp"] = np.zeros(2)

    for node in nodes:
        node.attrs["disp"] += qt.compute_repulsion(node, REPULSION_COEFF)

    for edge in edges:
        delta = edge.source_node.pos - edge.target_node.pos
        dist = np.linalg.norm(delta) + 1e-6
        force = (dist * dist) / K
        dir = delta / dist
        edge.source_node.attrs["disp"] -= dir * force * ATTRACTION_COEFF
        edge.target_node.attrs["disp"] += dir * force * ATTRACTION_COEFF

    for node in nodes:
        disp = node.attrs["disp"]
        d = np.linalg.norm(disp)
        if d > 0:
            node.attrs["velocity"] = (node.attrs["velocity"] + (disp / d) * min(d, 10)) * FRICTION
            node.pos += node.attrs["velocity"] * VELOCITY_SCALE
        node.pos = np.clip(node.pos, NODE_RADIUS, [WIDTH - 220 - NODE_RADIUS, HEIGHT - NODE_RADIUS])

def draw_graph(screen, nodes: list[Node], edges: list[Edge]):
    screen.fill((255, 255, 255))
    with lock:
        for edge in edges:
            pygame.draw.line(screen, (0, 0, 0), edge.source_node.pos.astype(int), edge.target_node.pos.astype(int), 1)
        for node in nodes:
            pygame.draw.circle(screen, (0, 102, 204), node.pos.astype(int), NODE_RADIUS)

def draw_slider(screen, x, y, w, h, min_val, max_val, value, label, font):
    pygame.draw.rect(screen, (200, 200, 200), (x, y + h // 2 - 3, w, 6))
    rel = (value - min_val) / (max_val - min_val)
    slider_x = int(x + rel * w)
    pygame.draw.circle(screen, (0, 102, 204), (slider_x, y + h // 2), 10)
    label_text = font.render(f"{label}: {value:.2f}", True, (0, 0, 0))
    screen.blit(label_text, (x, y - 25))
    return pygame.Rect(slider_x - 10, y + h // 2 - 10, 20, 20)

def draw_controls(screen, font):
    panel_w = 220
    pygame.draw.rect(screen, (240, 240, 240), (WIDTH - panel_w, 0, panel_w, HEIGHT))

    x_base, slider_w = WIDTH - panel_w + 20, 160
    y0 = 40
    step = 80

    sliders = [
        draw_slider(screen, x_base, y0 + 0*step, slider_w, 30, 0.1, 5.0, REPULSION_COEFF, "Repulsión", font),
        draw_slider(screen, x_base, y0 + 1*step, slider_w, 30, 0.01, 1.0, ATTRACTION_COEFF, "Atracción", font),
        draw_slider(screen, x_base, y0 + 2*step, slider_w, 30, 0.5, 1.0, FRICTION, "Fricción", font),
        draw_slider(screen, x_base, y0 + 3*step, slider_w, 30, 5.0, 80.0, K, "K (dist. ideal)", font),
        draw_slider(screen, x_base, y0 + 4*step, slider_w, 30, 1.0, 20.0, VELOCITY_SCALE, "Escala velocidad", font),
    ]

    # Botón reiniciar
    btn_rect = pygame.Rect(x_base + 10, y0 + 5*step + 10, slider_w - 20, 40)
    pygame.draw.rect(screen, (200, 50, 50), btn_rect)
    label = font.render("Reiniciar", True, (255, 255, 255))
    screen.blit(label, (btn_rect.x + 20, btn_rect.y + 10))

    return sliders, btn_rect

def reset_simulation(nodes):
    with lock:
        for node in nodes:
            node.pos = np.array([np.random.uniform(0, WIDTH - 220), np.random.uniform(0, HEIGHT)])
            node.attrs["velocity"] = np.zeros(2)

def main():
    global FRICTION, ATTRACTION_COEFF, REPULSION_COEFF, K, VELOCITY_SCALE, simulation_running

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Force-Directed Barnes-Hut")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    g = Graph("BARNES_GRAPH")
    
     # GRID GRAPH
    #g.create_grid_graph(10, 10)
    #g.create_grid_graph(50, 10)
    
    # ERDOS RENYI
    #g.create_erdos_renyi_graph(100, 130)
    #g.create_erdos_renyi_graph(500, 530)
    
    # Gilbert
    #g.create_gilbert_graph(100, 0.95)
    #g.create_gilbert_graph(500, 0.992)
    
    # Geo Graph
    #g.create_geo_graph(100, 0.5)
    #g.create_geo_graph(500, 0.1)
    
    # Barabasi
    #g.create_barabasi_graph(100, 10)
    #g.create_barabasi_graph(500, 5)
    
    # Dorogov
    #g.create_dorogov_graph(100)
    g.create_dorogov_graph(500)
    
    print(len(g.nodes), len(g.edges))

    nodes, edges = list(g.nodes.values()), list(g.edges.values())
    for node in nodes:
        node.attrs["velocity"] = np.zeros(2)

    # Lanzar el hilo de simulación
    sim_thread = threading.Thread(target=simulate_thread, args=(nodes, edges))
    sim_thread.start()

    dragging = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for i, rect in enumerate(slider_rects):
                    if rect.collidepoint((mx, my)):
                        dragging = (i, mx)
                if btn_rect.collidepoint((mx, my)):
                    reset_simulation(nodes)
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                i, mx0 = dragging
                mx, _ = pygame.mouse.get_pos()
                rel = min(max((mx - (WIDTH - 220 + 20)) / 160, 0), 1)
                if i == 0:
                    REPULSION_COEFF = 0.1 + rel * (5.0 - 0.1)
                elif i == 1:
                    ATTRACTION_COEFF = 0.01 + rel * (1.0 - 0.01)
                elif i == 2:
                    FRICTION = 0.5 + rel * (1.0 - 0.5)
                elif i == 3:
                    K = 5.0 + rel * (80.0 - 5.0)
                elif i == 4:
                    VELOCITY_SCALE = 1.0 + rel * (20.0 - 1.0)

        draw_graph(screen, nodes, edges)
        slider_rects, btn_rect = draw_controls(screen, font)
        pygame.display.flip()
        clock.tick(60)

    # Detener el hilo al salir
    simulation_running = False
    sim_thread.join()
    pygame.quit()

if __name__ == "__main__":
    main()
