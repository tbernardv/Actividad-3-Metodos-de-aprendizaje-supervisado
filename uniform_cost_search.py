# uniform_cost_search.py
import heapq
from graph import get_neighbors

def uniform_cost_search(graph, start, goal, preferred_transport=None):
    # La cola de prioridad para mantener los nodos por explorar
    queue = [(0, start, [], None)]  # (costo acumulado, nodo actual, camino, modo de transporte actual)
    visited = set()  # Para evitar visitar nodos repetidos

    while queue:
        cost, node, path, transport_mode = heapq.heappop(queue)  # Extrae el nodo con el menor costo

        if node in visited:
            continue

        path = path + [(node, transport_mode)]  # Actualizamos el camino
        visited.add(node)

        if node == goal:
            return (cost, path)  # Si llegamos al destino, retornamos el costo y el camino

        # Expandimos los nodos vecinos
        for neighbor, transports in get_neighbors(node).items():
            if neighbor not in visited:
                # Filtrar por el medio de transporte preferido, si se especifica
                for mode, weight in transports.items():
                    if preferred_transport is None or mode == preferred_transport:
                        heapq.heappush(queue, (cost + weight, neighbor, path, mode))

    return None  # Si no se encuentra el destino
