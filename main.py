# main.py
from graph import graph, add_connection, remove_connection
from uniform_cost_search import uniform_cost_search

def main():
    print("Bienvenido al sistema de búsqueda de rutas")
    print("Las estaciones disponibles son:", ", ".join(graph.keys()))

    start_node = input("Ingrese la estación de partida: ").strip().upper()
    goal_node = input("Ingrese la estación de destino: ").strip().upper()
    preferred_transport = input("Ingrese el medio de transporte preferido (bus, metro, bicicleta o dejar en blanco para cualquier): ").strip().lower() or None

    # Realizar la búsqueda
    result = uniform_cost_search(graph, start_node, goal_node, preferred_transport)

    if result:
        cost, path = result
        print(f"La mejor ruta de {start_node} a {goal_node} tiene un costo de {cost} y sigue el camino:")
        for station, transport in path:
            if transport:
                print(f"{station} (en {transport})")
            else:
                print(station)
    else:
        print(f"No se pudo encontrar una ruta de {start_node} a {goal_node}.")

if __name__ == "__main__":
    main()
