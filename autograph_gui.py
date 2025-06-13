import pygame
import numpy as np
from classes.graph import Graph
from classes.node import Node
from classes.edge import Edge


def spring_frame(
    graph: Graph,
    pos,
    width=1920,
    height=1800,
    k=None,
    C=1,
    repulsion=0.4,
    friction=0.15,
    velocidad=None,
):
    nodes = list(graph.nodes.values())
    n = len(nodes)
    if n == 0:
        return pos, velocidad if velocidad is not None else {}
    if k is None:
        k = np.sqrt((width * height) / n)
    if velocidad is None:
        velocidad = {node.id: np.zeros(2) for node in nodes}
    disp = {node.id: np.zeros(2) for node in nodes}
    # Repulsión
    for i, v in enumerate(nodes):
        for j, u in enumerate(nodes):
            if i != j:
                delta = pos[v.id] - pos[u.id]
                dist = np.linalg.norm(delta) + 1e-6
                disp[v.id] += repulsion * (delta / dist) * (k * k / dist)
    # Atracción
    for edge in graph.edges.values():
        source = edge.source_node.id
        target = edge.target_node.id
        delta = pos[source] - pos[target]
        dist = np.linalg.norm(delta) + 1e-6
        force = (dist * dist) / k
        disp[source] -= (delta / dist) * force * C
        disp[target] += (delta / dist) * force * C
    # Limitar desplazamiento, actualizar posiciones y aplicar fricción
    for node in nodes:
        d = np.linalg.norm(disp[node.id])
        if d > 0:
            velocidad[node.id] = friction * (
                velocidad[node.id] + (disp[node.id] / d) * min(d, 10)
            )
            pos[node.id] += velocidad[node.id]
        pos[node.id] = np.clip(pos[node.id], 30, [width - 30, height - 30])
    return pos, velocidad


def draw_slider(screen, x, y, w, h, min_val, max_val, value, label):
    pygame.draw.rect(screen, (200, 200, 200), (x, y + h // 2 - 3, w, 6))
    # Posición del slider
    rel = (value - min_val) / (max_val - min_val)
    slider_x = int(x + rel * w)
    pygame.draw.circle(screen, (0, 102, 204), (slider_x, y + h // 2), 12)
    # Etiqueta y valor
    font = pygame.font.SysFont(None, 22)
    label_text = font.render(f"{label}: {value:.2f}", True, (0, 0, 0))
    screen.blit(label_text, (x, y - 22))
    return pygame.Rect(slider_x - 12, y + h // 2 - 12, 24, 24)


def draw_controles(screen, repulsion, atraccion, friction, width, height):
    panel_w = 220
    pygame.draw.rect(screen, (230, 230, 230), (width - panel_w, 0, panel_w, height))
    font = pygame.font.SysFont(None, 26)
    titulo = font.render("Controles", True, (0, 0, 0))
    screen.blit(titulo, (width - panel_w + 20, 20))
    # Sliders
    y0 = 70
    slider_w = panel_w - 40
    rep_rect = draw_slider(
        screen,
        width - panel_w + 20,
        y0,
        slider_w,
        30,
        0.01,
        2.0,
        repulsion,
        "Repulsión",
    )
    att_rect = draw_slider(
        screen,
        width - panel_w + 20,
        y0 + 70,
        slider_w,
        30,
        0.01,
        2.0,
        atraccion,
        "Atracción",
    )
    fric_rect = draw_slider(
        screen,
        width - panel_w + 20,
        y0 + 140,
        slider_w,
        30,
        0.5,
        0.99,
        friction,
        "Fricción",
    )
    # Botón reiniciar
    pygame.draw.rect(
        screen, (200, 50, 50), (width - panel_w + 40, y0 + 210, slider_w - 40, 40)
    )
    btn_font = pygame.font.SysFont(None, 24)
    btn_text = btn_font.render("Reiniciar", True, (255, 255, 255))
    screen.blit(btn_text, (width - panel_w + 60, y0 + 220))
    btn_rect = pygame.Rect(width - panel_w + 40, y0 + 210, slider_w - 40, 40)
    return rep_rect, att_rect, fric_rect, btn_rect, panel_w


def draw_graph(graph: Graph, width=1200, height=660, iterations=None):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Grafo Spring")
    clock = pygame.time.Clock()
    nodes = list(graph.nodes.values())

    def reset():
        return (
            {
                node.id: np.array(
                    [
                        np.random.uniform(0, width - 220),
                        np.random.uniform(30, height - 30),
                    ]
                )
                for node in nodes
            },
            {node.id: np.zeros(2) for node in nodes},
            0,
        )

    pos, velocidad, iter_count = reset()
    k = np.sqrt(((width - 220) * height) / max(1, len(nodes)))
    repulsion = 0.8
    atraccion = 1.0
    friction = 0.85
    dragging = None
    run = True
    stop = False
    min_move = 0.05  # Umbral de movimiento mínimo
    cuenta = 0
    cuenta_limite = 20  # Frames consecutivos sin movimiento relevante para detener
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if rep_rect.collidepoint(mouse_pos):
                    dragging = ("rep", mouse_pos[0])
                elif att_rect.collidepoint(mouse_pos):
                    dragging = ("att", mouse_pos[0])
                elif fric_rect.collidepoint(mouse_pos):
                    dragging = ("fric", mouse_pos[0])
                elif btn_rect.collidepoint(mouse_pos):
                    pos, velocidad, iter_count = reset()
                    stop = False
                    cuenta = 0
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                typ, _ = dragging
                mx = event.pos[0]
                slider_x = mx - (width - 220 + 20)
                slider_w = 160
                rel = min(max(slider_x / slider_w, 0), 1)
                if typ == "rep":
                    repulsion = 0.01 + rel * (2.0 - 0.01)
                elif typ == "att":
                    atraccion = 0.01 + rel * (2.0 - 0.01)
                elif typ == "fric":
                    friction = 0.5 + rel * (0.99 - 0.5)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    pos, velocidad, iter_count = reset()
                    stop = False
                    cuenta = 0
        # Animación
        if not stop:
            prev_pos = {nid: p.copy() for nid, p in pos.items()}
            pos, velocidad = spring_frame(
                graph,
                pos,
                width - 220,
                height,
                k,
                C=atraccion,
                repulsion=repulsion,
                friction=friction,
                velocidad=velocidad,
            )
            iter_count += 1
            # Ver si hay movimiento relevante
            max_move = max(np.linalg.norm(pos[nid] - prev_pos[nid]) for nid in pos)
            if max_move < min_move:
                cuenta += 1
            else:
                cuenta = 0
            if cuenta > cuenta_limite:
                stop = True
        screen.fill((255, 255, 255))
        rep_rect, att_rect, fric_rect, btn_rect, panel_w = draw_controles(
            screen, repulsion, atraccion, friction, width, height
        )
        # Dibujar aristas
        for edge in graph.edges.values():
            src = pos[edge.source_node.id]
            tgt = pos[edge.target_node.id]
            pygame.draw.line(screen, (0, 0, 0), src, tgt, 2)
        # Dibujar nodos
        for node in graph.nodes.values():
            p = pos[node.id]
            if node.get_degree() == 0:
                color = (180, 180, 180)
            else:
                color = (0, 102, 204)
            pygame.draw.circle(screen, color, p.astype(int), 18)
        pygame.display.flip()
        clock.tick()
    pygame.quit()


# la animacion se para si no hay cambios luego de un tiempo
# se reinicia con el boton o con la tecla R
if __name__ == "__main__":
    g = Graph("demo")
    # g.create_erdos_renyi_graph(100, 200)
    g.create_gilbert_graph(100)
    # g.create_geo_graph(25)
    # g.create_barabasi_graph(25)
    # g.create_dorogov_graph(25)
    draw_graph(g)
