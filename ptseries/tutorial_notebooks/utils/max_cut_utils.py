import matplotlib.pyplot as plt
import networkx as nx


def separate_in_two_subsets(G, subset1, subset2):
    """Draws the two subsets of the given graph in two different colors"""
    G2 = nx.Graph()

    edge_list1 = []
    edge_list2 = []
    for i, j in G.edges:
        if ((i in subset1) and (j in subset2)) or ((j in subset1) and (i in subset2)):
            edge_list2.append((i, j))
        else:
            edge_list1.append((i, j))

    print("number of cutting edges = ", len(edge_list2))

    G2.add_nodes_from(G.nodes)
    nodes_color = []
    for i in G.nodes:
        G2.add_node(i)
        if i in subset1:
            nodes_color.append("red")
        else:
            nodes_color.append("blue")

    G2.add_edges_from(edge_list1)
    G2.add_edges_from(edge_list2)

    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
    _, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G2, pos, node_size=700, node_color=nodes_color, ax=ax)

    # edges
    nx.draw_networkx_edges(G2, pos, edgelist=edge_list1, width=5, ax=ax)
    nx.draw_networkx_edges(G2, pos, edgelist=edge_list2, width=2, style="--", ax=ax)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif", ax=ax)

    ax.set_axis_off()
    ax.margins(0.08)
    plt.tight_layout()
    plt.show()
