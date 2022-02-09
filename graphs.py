import networkx as nx
import matplotlib.pyplot as plt
from itertools import product, combinations
import numpy as np


def get_graph_from_neighbours(n_list):
    g = nx.Graph()
    for i in range(len(n_list)):
        g.add_node(i)
    for nod in n_list:
        for nod2 in nod[1]:
            if (nod[0], nod2) not in g.edges and (nod2, nod[0]) not in g.edges:
                g.add_edge(nod[0], nod2)
    return g


def draw_graph(g, spin_nodes=0, save=False, filename=None):
    if type(spin_nodes) is int:
        spin_nodes = [spin_nodes]
    colour_map = []
    for n in range(len(g.nodes)):
        if n in spin_nodes:
            colour_map.append('red')
        else:
            colour_map.append('blue')

    pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos=pos, node_color=colour_map, with_labels=True)
    if save:
        plt.savefig(filename)
    plt.show()


def gen_linear_graph(nqubits):
    graph = nx.Graph()
    graph.add_nodes_from(range(nqubits))
    these_edges = [(node_ix, node_ix + 1) for node_ix in range(nqubits - 1)]
    graph.add_edges_from(these_edges)
    return graph


def gen_ring_graph(nqubits):
    graph = nx.Graph()
    graph.add_nodes_from(range(nqubits))
    these_edges = [(node_ix, (node_ix + 1) % nqubits) for node_ix in range(nqubits)]
    graph.add_edges_from(these_edges)
    return graph


def gen_star_graph(nqubits, central_qubit=0):
    graph = nx.Graph()
    nodes = range(nqubits)
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        product([central_qubit], [other_nodes for other_nodes in nodes if other_nodes != central_qubit]))
    return graph


def gen_fullyconnected_graph(nqubits):
    graph = nx.Graph()
    nodes = range(nqubits)
    graph.add_nodes_from(nodes)
    graph.add_edges_from(combinations(nodes, 2))
    return graph


def gen_crazy_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix * nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        these_edges = product(nodes_mat[layer_ix], nodes_mat[layer_ix + 1])
        graph.add_edges_from(these_edges)
    return graph


def gen_multiwire_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix * nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        these_edges = zip(nodes_mat[layer_ix], nodes_mat[layer_ix + 1])
        graph.add_edges_from(these_edges)
    return graph


def gen_square_lattice_graph(nrows, nlayers):
    graph = nx.Graph()
    nodes_mat = np.arange(nrows * nlayers).reshape((nlayers, nrows))
    for layer_ix in range(nlayers):
        for row_ix in range(nrows):
            graph.add_node(layer_ix * nrows + row_ix, layer=layer_ix)
    for layer_ix in range(nlayers - 1):
        # Horizontal edges
        these_edges = list(zip(nodes_mat[layer_ix], nodes_mat[layer_ix + 1]))
        graph.add_edges_from(these_edges)
    for layer_ix in range(nlayers):
        # Vertical edges
        these_edges = [tuple([nodes_mat[layer_ix, row_ix], nodes_mat[layer_ix, row_ix + 1]])
                       for row_ix in range(nrows - 1)]
        graph.add_edges_from(these_edges)
    return graph


def main():
    pass


if __name__ == '__main__':
    main()


