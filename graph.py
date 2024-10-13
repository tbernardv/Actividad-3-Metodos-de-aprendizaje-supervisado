# graph.py

# Grafo con diferentes medios de transporte y sus respectivos costos
graph = {
    'A': {'B': {'bus': 2, 'metro': 1.5}, 'C': {'bus': 1}},
    'B': {'A': {'bus': 2}, 'D': {'metro': 3}, 'E': {'bus': 7}},
    'C': {'A': {'bus': 1}, 'F': {'metro': 5}},
    'D': {'B': {'metro': 3}},
    'E': {'B': {'bus': 7}, 'F': {'metro': 3, 'bicicleta': 2}},
    'F': {'C': {'metro': 5}, 'E': {'metro': 3, 'bicicleta': 2}}
}

# Función para obtener los vecinos de un nodo
def get_neighbors(node):
    return graph[node]

# Función para agregar una nueva conexión
def add_connection(node1, node2, transport_mode, cost):
    if node1 in graph:
        graph[node1][node2] = {transport_mode: cost}
    else:
        graph[node1] = {node2: {transport_mode: cost}}

# Función para eliminar una conexión
def remove_connection(node1, node2):
    if node1 in graph and node2 in graph[node1]:
        del graph[node1][node2]
