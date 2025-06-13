import pygame
import numpy as np
from classes.graph import Graph
from classes.node import Node
from classes.edge import Edge
import multiprocessing
import time

# --- Parámetros globales para la grilla de optimización de repulsión ---
GRID_CELL_SIZE = 100  # Tamaño de cada celda de la grilla en píxeles


def spring_frame_worker(
    graph_data,
    initial_pos_serializable,  # Recibido como lista de listas
    initial_velocidad_serializable,  # Recibido como lista de listas
    width,
    height,
    k_init,  # Initial k
    C_init,  # Initial C (attraction)
    repulsion_init,
    friction_init,
    output_queue,
    input_queue,
):
    """
    Función worker para calcular las posiciones de los nodos en un proceso separado.
    Incorpora optimizaciones para el cálculo de repulsión.
    """
    # Reconstruir el grafo a partir de los datos serializables
    temp_graph = Graph("temp")
    node_id_map = {node_id: Node(node_id) for node_id in graph_data["nodes_ids"]}
    temp_graph.nodes = node_id_map

    # Reconstruir aristas de forma más eficiente (si fuera necesario para la lógica de Edge)
    # Por ahora, solo necesitamos las tuplas (source_id, target_id) para la atracción
    edges_tuples = graph_data["edges_tuples"]

    nodes = list(temp_graph.nodes.values())
    node_ids = graph_data["nodes_ids"]  # Lista de IDs para orden consistente

    # Convertir de nuevo a NumPy arrays
    pos = {nid: np.array(p_val) for nid, p_val in initial_pos_serializable.items()}
    velocidad = {
        nid: np.array(v_val) for nid, v_val in initial_velocidad_serializable.items()
    }

    current_k = k_init
    current_C = C_init
    current_repulsion = repulsion_init
    current_friction = friction_init

    running = True
    while running:
        # Comprobar si hay nuevos parámetros o señales de control
        while not input_queue.empty():
            msg = input_queue.get()
            if msg == "STOP":
                running = False
                break
            elif msg == "RESET":
                # Realizar un reinicio de la simulación
                pos = {
                    node_id: np.array(
                        [
                            np.random.uniform(0, width),
                            np.random.uniform(30, height - 30),
                        ]
                    )
                    for node_id in node_ids
                }
                velocidad = {node_id: np.zeros(2) for node_id in node_ids}
                current_k = np.sqrt((width * height) / max(1, len(node_ids)))
                current_repulsion = repulsion_init  # Reiniciar a valores iniciales
                current_C = C_init
                current_friction = friction_init
            elif isinstance(msg, dict) and "params" in msg:
                params = msg["params"]
                current_repulsion = params.get("repulsion", current_repulsion)
                current_C = params.get("attraction", current_C)
                current_friction = params.get("friction", current_friction)

        if not running:
            break

        n = len(node_ids)
        if n == 0:
            output_queue.put({})
            time.sleep(0.01)
            continue

        disp = {node_id: np.zeros(2) for node_id in node_ids}

        # --- OPTIMIZACIÓN DE REPULSIÓN CON GRILLA ---
        # Construir la grilla
        grid = {}
        for node_id in node_ids:
            x, y = pos[node_id]
            grid_x = int(x // GRID_CELL_SIZE)
            grid_y = int(y // GRID_CELL_SIZE)
            if (grid_x, grid_y) not in grid:
                grid[(grid_x, grid_y)] = []
            grid[(grid_x, grid_y)].append(node_id)

        # Calcular repulsión solo con nodos en celdas adyacentes
        for node_id_v in node_ids:
            x_v, y_v = pos[node_id_v]
            grid_x_v = int(x_v // GRID_CELL_SIZE)
            grid_y_v = int(y_v // GRID_CELL_SIZE)

            # Buscar en la celda actual y las 8 celdas vecinas
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_grid_x = grid_x_v + dx
                    neighbor_grid_y = grid_y_v + dy
                    if (neighbor_grid_x, neighbor_grid_y) in grid:
                        for node_id_u in grid[(neighbor_grid_x, neighbor_grid_y)]:
                            if node_id_v != node_id_u:
                                delta = pos[node_id_v] - pos[node_id_u]
                                dist = (
                                    np.linalg.norm(delta) + 1e-6
                                )  # Evitar división por cero
                                force = current_repulsion * (
                                    current_k * current_k / dist
                                )
                                disp[node_id_v] += (delta / dist) * force
        # --- FIN OPTIMIZACIÓN DE REPULSIÓN ---

        # Atracción (este bucle es O(E), donde E es el número de aristas,
        # lo cual es eficiente a menos que el grafo sea extremadamente denso)
        for source_id, target_id in edges_tuples:
            delta = pos[source_id] - pos[target_id]
            dist = np.linalg.norm(delta) + 1e-6
            force = (dist * dist) / current_k
            disp[source_id] -= (delta / dist) * force * current_C
            disp[target_id] += (delta / dist) * force * current_C

        # Limitar desplazamiento, actualizar posiciones y aplicar fricción
        max_move_this_frame = 0.0
        new_velocidad = {}
        for node_id in node_ids:
            d = np.linalg.norm(disp[node_id])
            if d > 0:
                # El "min(d, 10)" ayuda a prevenir movimientos excesivamente grandes
                # en las primeras iteraciones o con fuerzas muy altas.
                accel_component = (disp[node_id] / d) * min(d, 10)
                new_velocidad[node_id] = current_friction * (
                    velocidad[node_id] + accel_component
                )
                pos[node_id] += new_velocidad[node_id]
            else:
                new_velocidad[node_id] = (
                    velocidad[node_id] * current_friction
                )  # Apply friction even if no displacement

            # Asegurar que las posiciones estén dentro de los límites
            pos[node_id] = np.clip(
                pos[node_id], 10, [width - 10, height - 10]
            )  # Reduced padding for worker

            # Calcular el movimiento para la detección de parada
            # Usar la magnitud del cambio en la posición de este frame
            max_move_this_frame = max(
                max_move_this_frame, np.linalg.norm(new_velocidad[node_id])
            )  # Basado en velocidad para estabilidad

        velocidad = new_velocidad

        # Enviar las posiciones actualizadas y el movimiento máximo al proceso principal
        serializable_pos = {nid: p.tolist() for nid, p in pos.items()}
        serializable_velocidad = {nid: v.tolist() for nid, v in velocidad.items()}
        output_queue.put(
            {
                "pos": serializable_pos,
                "velocidad": serializable_velocidad,
                "max_move": max_move_this_frame,
            }
        )

        # Pequeña pausa para permitir que el proceso principal procese la cola
        # Esto es importante para evitar que el worker inunde la cola
        time.sleep(0.001)


def draw_slider(screen, x, y, w, h, min_val, max_val, value, label, font):
    pygame.draw.rect(screen, (200, 200, 200), (x, y + h // 2 - 3, w, 6))
    rel = (value - min_val) / (max_val - min_val)
    slider_x = int(x + rel * w)
    pygame.draw.circle(screen, (0, 102, 204), (slider_x, y + h // 2), 12)
    label_text = font.render(f"{label}: {value:.2f}", True, (0, 0, 0))
    screen.blit(label_text, (x, y - 22))
    return pygame.Rect(slider_x - 12, y + h // 2 - 12, 24, 24)


def draw_controles(
    screen, repulsion, atraccion, friction, width, height, font, btn_font
):
    panel_w = 220
    pygame.draw.rect(screen, (230, 230, 230), (width - panel_w, 0, panel_w, height))
    titulo = font.render("Controles", True, (0, 0, 0))
    screen.blit(titulo, (width - panel_w + 20, 20))
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
        font,
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
        font,
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
        font,
    )
    pygame.draw.rect(
        screen, (200, 50, 50), (width - panel_w + 40, y0 + 210, slider_w - 40, 40)
    )
    btn_text = btn_font.render("Reiniciar", True, (255, 255, 255))
    screen.blit(btn_text, (width - panel_w + 60, y0 + 220))
    btn_rect = pygame.Rect(width - panel_w + 40, y0 + 210, slider_w - 40, 40)
    return rep_rect, att_rect, fric_rect, btn_rect, panel_w


def draw_graph(
    graph: Graph, width=1200, height=720
):  # Aumenta un poco el alto para los controles
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Grafo Spring")
    clock = pygame.time.Clock()

    # Fuentes cargadas una vez
    font = pygame.font.SysFont(None, 22)
    btn_font = pygame.font.SysFont(None, 24)

    nodes = list(graph.nodes.values())
    node_ids = [node.id for node in nodes]  # Guardar IDs para referencia

    output_queue = multiprocessing.Queue()
    input_queue = multiprocessing.Queue()

    graph_data_for_worker = {
        "nodes_ids": node_ids,
        "edges_tuples": [
            (edge.source_node.id, edge.target_node.id) for edge in graph.edges.values()
        ],
    }

    # Definir el área de dibujo para el grafo (excluyendo el panel de control)
    graph_draw_width = width - 220
    graph_draw_height = height

    def initialize_positions():
        return {
            node_id: np.array(
                [
                    np.random.uniform(0, graph_draw_width),
                    np.random.uniform(30, graph_draw_height - 30),
                ]
            )
            for node_id in node_ids
        }

    current_pos_ui = initialize_positions()  # Posiciones usadas para dibujar en la UI
    initial_velocidad = {node_id: np.zeros(2) for node_id in node_ids}

    # Parámetros iniciales de la simulación
    k = np.sqrt((graph_draw_width * graph_draw_height) / max(1, len(nodes)))
    repulsion = 0.8
    atraccion = 1.0
    friction = 0.85

    worker_process = multiprocessing.Process(
        target=spring_frame_worker,
        args=(
            graph_data_for_worker,
            {nid: p.tolist() for nid, p in current_pos_ui.items()},
            {nid: v.tolist() for nid, v in initial_velocidad.items()},
            graph_draw_width,  # Pasar el ancho y alto del área de dibujo al worker
            graph_draw_height,
            k,
            atraccion,
            repulsion,
            friction,
            output_queue,
            input_queue,
        ),
    )
    worker_process.start()

    iter_count = 0
    dragging = None
    run = True
    stop_simulation_in_worker = (
        False  # Esto controla si el worker debe pausar sus cálculos
    )
    min_move = 0.05
    cuenta = 0
    cuenta_limite = 60  # Más frames consecutivos sin movimiento relevante para detener (para mayor estabilidad)

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                # Check slider clicks
                if rep_rect.collidepoint(mouse_pos):
                    dragging = ("rep", mouse_pos[0])
                elif att_rect.collidepoint(mouse_pos):
                    dragging = ("att", mouse_pos[0])
                elif fric_rect.collidepoint(mouse_pos):
                    dragging = ("fric", mouse_pos[0])
                elif btn_rect.collidepoint(mouse_pos):
                    input_queue.put("RESET")
                    current_pos_ui = initialize_positions()  # Reset UI positions
                    # No reiniciar velocidad en UI, el worker la manejará
                    stop_simulation_in_worker = False
                    cuenta = 0
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                typ, _ = dragging
                mx = event.pos[0]
                slider_x_panel = (
                    width - panel_w + 20
                )  # X inicial de los sliders en el panel
                slider_w = panel_w - 40  # Ancho de los sliders
                rel = min(max((mx - slider_x_panel) / slider_w, 0), 1)

                new_params = {}
                if typ == "rep":
                    repulsion = 0.01 + rel * (2.0 - 0.01)
                    new_params["repulsion"] = repulsion
                elif typ == "att":
                    atraccion = 0.01 + rel * (2.0 - 0.01)
                    new_params["attraction"] = atraccion
                elif typ == "fric":
                    friction = 0.5 + rel * (0.99 - 0.5)
                    new_params["friction"] = friction

                if new_params:
                    input_queue.put({"params": new_params})
                    stop_simulation_in_worker = False
                    cuenta = 0

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    input_queue.put("RESET")
                    current_pos_ui = initialize_positions()
                    stop_simulation_in_worker = False
                    cuenta = 0

        # Recibir las posiciones actualizadas del proceso worker
        # Intentamos obtener sin bloquear para no detener el bucle de renderizado
        try:
            data = output_queue.get_nowait()  # Non-blocking get
            if "pos" in data:
                current_pos_ui = {
                    nid: np.array(p_list) for nid, p_list in data["pos"].items()
                }
                # velocidad_ui = {nid: np.array(v_list) for nid, v_list in data["velocidad"].items()}
                max_move = data["max_move"]
                iter_count += 1

                if max_move < min_move:
                    cuenta += 1
                else:
                    cuenta = 0

                # Le decimos al worker que se detenga si la simulación está estable
                if cuenta > cuenta_limite and not stop_simulation_in_worker:
                    input_queue.put(
                        "STOP_CALC"
                    )  # Nuevo comando para detener solo los cálculos
                    stop_simulation_in_worker = True
                elif cuenta <= cuenta_limite and stop_simulation_in_worker:
                    input_queue.put("START_CALC")  # Nuevo comando para reanudar
                    stop_simulation_in_worker = False

        except multiprocessing.queues.Empty:
            # No hay datos nuevos, simplemente usamos las últimas posiciones conocidas
            pass

        screen.fill((255, 255, 255))
        rep_rect, att_rect, fric_rect, btn_rect, panel_w = draw_controles(
            screen, repulsion, atraccion, friction, width, height, font, btn_font
        )

        # Dibujar aristas (usando las posiciones actualizadas de la UI)
        for edge in graph.edges.values():
            src = current_pos_ui.get(edge.source_node.id)
            tgt = current_pos_ui.get(edge.target_node.id)
            if src is not None and tgt is not None:
                # Asegurar que las coordenadas estén dentro del área de dibujo del grafo
                # y ajustadas para la posición de la pantalla (ya que el worker calculó para un área menor)
                pygame.draw.line(screen, (0, 0, 0), src, tgt, 2)
        # Dibujar nodos
        for node in graph.nodes.values():
            p = current_pos_ui.get(node.id)
            if p is not None:
                if node.get_degree() == 0:
                    color = (180, 180, 180)
                else:
                    color = (0, 102, 204)
                pygame.draw.circle(screen, color, p.astype(int), 18)

        pygame.display.flip()
        clock.tick(
            60
        )  # Limitar a 60 FPS para no consumir CPU innecesariamente en el renderizado

    # Al salir del bucle principal, enviar señal de parada al proceso worker y esperar a que termine
    input_queue.put("STOP")
    worker_process.join()
    pygame.quit()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    g = Graph("demo")
    # Aumentar el número de nodos para probar el cuello de botella
    # g.create_erdos_renyi_graph(500, 1500) # Más nodos, menos aristas (sparse)
    g.create_gilbert_graph(
        500, 0.995
    )  # Grafos más densos son más problemáticos para N^2 repulsión
    # g.create_barabasi_graph(1000) # Pruebas con 1000 nodos o más
    print(len(g.nodes), len(g.edges))
    draw_graph(g)
