import networkx as nx
import matplotlib.pyplot as plt
from itertools import product, combinations
import numpy as np
from alcpack import local_complementation


def get_graph_from_neighbours(n_list):
    g = nx.Graph()
    for i in range(len(n_list)):
        g.add_node(i)
    for nod in n_list:
        for nod2 in nod[1]:
            if (nod[0], nod2) not in g.edges and (nod2, nod[0]) not in g.edges:
                g.add_edge(nod[0], nod2)
    return g


def graph_from_edges(edges):
    g = nx.Graph()
    n_nodes = max([node for edge in edges for node in edge])
    g.add_nodes_from(list(range(n_nodes)))
    g.add_edges_from(edges)
    return g


def draw_graph(g, spin_nodes=0, save=False, filename=None, from_edges=False):
    if from_edges:
        g_ = nx.Graph()
        n_nodes = max([m for e in g for m in e])
        g_.add_nodes_from(list(range(n_nodes)))
        g_.add_edges_from(g)
        g = g_
    if spin_nodes is None:
        spin_nodes = []
    elif type(spin_nodes) is int:
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


def gen_rgs_graph(n):
    """
    generate the repeater graph state from "Nature Communications volume 6, Article number: 6787 (2015)"
    :param n:
    :return:
    """
    g = gen_fullyconnected_graph(2 * n)
    extra_nodes = [i for i in range(2*n, 4 * n)]
    g.add_nodes_from(extra_nodes)
    g.add_edges_from([(node, node+2 * n) for node in range(2 * n)])
    return g


def x_meas_graph(g, v):
    """
    generate the new graph obtained by performing an x measurement on vertex v of the graph g
    :param g:
    :param v:
    :return:
    """
    a = [x for x in g.neighbors(v)][0]
    g2 = local_complementation(local_complementation(g, a), v)
    nu_edge_list = []
    for e in g2.edges:
        if v not in e:
            nu_edge_list.append(e)
    out_graph = nx.Graph()
    out_graph.add_nodes_from(g.nodes)
    out_graph.add_edges_from(nu_edge_list)
    return local_complementation(out_graph, a)


def y_meas_graph(g, v):
    g2 = local_complementation(g, v)
    nu_edge_list = []
    for e in g2.edges:
        if v not in e:
            nu_edge_list.append(e)
    out_graph = nx.Graph()
    out_graph.add_nodes_from(g.nodes)
    out_graph.add_edges_from(nu_edge_list)
    return out_graph


def z_meas_graph(g, v):
    """
    Perform a z measurement
    :param g:
    :param v:
    :return:
    """
    nu_edge_list = []
    for e in g.edges:
        if v not in e:
            nu_edge_list.append(e)
    out_graph = nx.Graph()
    out_graph.add_nodes_from(g.nodes)
    out_graph.add_edges_from(nu_edge_list)
    return out_graph


def gen_cascaded_graph(graph_list):
    """
    Generate the networkx graph corresponding to the cascade in graph_list
    :param graph_list:
    :return:
    """
    depth = len(graph_list)
    qubit_labels = [[0]]  # A list of lists, each representing qubit number at each layer
    edges = []
    current_max = 0

    for d in range(depth):
        # Find qubit numbers in this layer.
        new_layer_qbts = []
        new_layer_edges = []
        g_qbts = list(graph_list[d].nodes())
        g_qbts.remove(0)
        g_edges = list(graph_list[d].edges())
        for qbt_root in qubit_labels[d]:
            new_layer_qbts += [q + current_max for q in g_qbts]
            for e in g_edges:
                if e[0] == 0:
                    new_layer_edges.append((qbt_root, e[1] + current_max))
                else:
                    new_layer_edges.append((e[0] + current_max, e[1] + current_max))
            current_max = new_layer_qbts[-1]
        qubit_labels.append(new_layer_qbts)
        edges += new_layer_edges

    g = nx.Graph()
    g.add_edges_from(edges)
    return g, qubit_labels


def main():
    g3 = gen_ring_graph(3)
    g5 = gen_ring_graph(5)
    g, nodes = gen_cascaded_graph([g5, g5, g3])
    draw_graph(g)
    plt.show()


if __name__ == '__main__':
    main()
