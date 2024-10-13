# visualization.py
import matplotlib.pyplot as plt
import networkx as nx
from graph import graph

def draw_graph():
    G = nx.Graph()

    for node, neighbors in graph.items():
        for neighbor, transports in neighbors.items():
            for mode, cost in transports.items():
                G.add_edge(node, neighbor, weight=cost, label=mode)

    pos = nx.spring_layout(G)  # Posiciones de los nodos
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Mapa del Sistema de Transporte")
    plt.show()

if __name__ == "__main__":
    draw_graph()
