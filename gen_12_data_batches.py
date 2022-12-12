from helpers import load_obj, save_obj
from multiprocessing import Pool
import argparse
import sys
import networkx as nx
from decoder_class import FastDecoder, CascadeDecoder
import bz2
from os.path import join
from graphs import graph_from_edges


def read_data_from_bz2(line_start, lines_stop, filename, path_to_file):
    full_path = join(path_to_file, filename)
    graphs_to_inspect = []
    f = bz2.BZ2File(full_path, "r")
    count = 0
    for line in f:
        if count < line_start:
            pass
        elif count < lines_stop:
            line = line.decode('UTF-8')
            x = line.split('\t')
            n_graphs = int(x[1])
            schmidt = x[5]
            edge_list_ix = 10
            for ix in range(len(x)):
                if x[ix] == 'yes' or x[ix] == 'no':
                    edge_list_ix = ix + 1

            edge_list = x[edge_list_ix]
            e2 = edge_list.replace(')(', ',').removeprefix('(').removesuffix(')').split(',')
            e3 = [_.split('-') for _ in e2]
            e4 = [(int(x[0]), int(x[1])) for x in e3]
            graphs_to_inspect.append((n_graphs, e4))
        else:
            break
        count += 1
    return graphs_to_inspect


def permute_input_qubit(edge_list, in_ix):
    if in_ix == 0:
        return edge_list
    else:
        def new_ix(ix):
            if ix != in_ix and ix != 0:
                return ix
            elif ix == in_ix:
                return 0
            elif ix == 0:
                return in_ix
        edge_list_permuted = [(new_ix(e[0]), new_ix(e[1])) for e in edge_list]
        return edge_list_permuted


def gen_non_isomorphic_graphs(edge_list_in):
    """
    From a graph represented by the edge_list_in, return the set of graphs that are inequivalent under permutation of the code (non-input) qubits
    :param edge_list_in:
    :return: list of edge_lists of the unique graphs
    """
    g = graph_from_edges(edge_list_in)
    n = g.number_of_nodes()

    # Give the graph edge attributes
    input = {0: 1}
    for node in range(1, n):
        input[node] = 0
    nx.set_node_attributes(g, input, "is_input")

    unique_graphs_list = [g]
    nm = nx.isomorphism.categorical_node_match("is_input", 0)
    for ix in range(1, n):
        g2 = nx.Graph()
        g2.add_nodes_from(list(range(n)))
        input = {0: 1}
        for node in range(1, n):
            input[node] = 0
        nx.set_node_attributes(g2, input, "is_input")
        es = permute_input_qubit(list(g.edges), ix)
        g2.add_edges_from(es)
        new = True
        for g1 in unique_graphs_list:
            if nx.isomorphism.is_isomorphic(g1, g2, node_match=nm):
                new = False
        if new:
            unique_graphs_list.append(g2)
    return [x.edges for x in unique_graphs_list]


def get_all_graphs(in_graph):
    e = in_graph[1]
    return [(in_graph[0], x) for x in gen_non_isomorphic_graphs(e)]


def get_data_chunk(ix, size):
    dat = load_obj(name=f'12_qubit_graphs_ordered_num_in_class', path=path_to_data)
    length = len(dat)
    if (ix + 1) * size < length:
        data_chunk = dat[ix * size:(ix+1) * size]
    elif ix * size < length:
        data_chunk = dat[ix * size:]
    else:
        raise ValueError(f"Array Index too large for data input, with {ix=}, batch size={size} and {length} graphs")
    return data_chunk


def graph_perf(edges):
    iso_out = []
    all_gs = gen_non_isomorphic_graphs(edges)
    for x in all_gs:
        g = graph_from_edges(x)
        dec = FastDecoder(g)
        spc = dec.get_dict(condensed=True)
        iso_out.append((x, spc))
    return iso_out


def graph_perf_pauli(edges):
    """
    :param graph_info: tuple (class_size, edgelist) of the representative graph of the equivalence class
    :return: tuple (edgelist, x, y, z results dictionaries)
    """
    iso_out = []
    all_gs = gen_non_isomorphic_graphs(edges)
    i = 0
    for x in all_gs:
        g = graph_from_edges(x)
        dec = CascadeDecoder(g)
        bases = ['x', 'y', 'z']
        xr, yr, zr = [dec.get_dict(basis=b, cascading=False, condensed=True) for b in bases]
        iso_out.append((x, xr, yr, zr))
    return iso_out


if __name__ == '__main__':
    # Parse the command line inputs
    parser = argparse.ArgumentParser(description="Pass variables for Loss tolerance SPF")
    parser.add_argument(
        "-arix",
        "--array_ix",
        help="array index of data chunk to analyse",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-p",
        "--path_to_data",
        help="path to the directory containing the graph data",
        type=str,
        default="",
    )
    parser.add_argument(
        "-df",
        "--datafile",
        help="name of datafile",
        type=str,
        default="",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="size of batches to examine",
        type=int,
        default=500,
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        help="output directory",
        type=str,
        default="",
    )
    parser.add_argument(
        "-ofs",
        "--offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-pb",
        "--pauli_basis",
        help="Do pauli basis measurements",
        action=argparse.BooleanOptionalAction
    )

    arguments = parser.parse_args(sys.argv[1:])
    array_ix = arguments.array_ix + arguments.offset
    path_to_data = arguments.path_to_data
    batch_size = arguments.batch_size
    output_dir = arguments.output_dir
    filename = arguments.datafile
    pauli_basis = arguments.pauli_basis
    if output_dir == "":
        output_dir = path_to_data

    if pauli_basis:
        func = graph_perf_pauli
        name_suffix = '_pauli'
    else:
        func = graph_perf
        name_suffix = ''

    # Load a chunk of the 12 qubit classes
    # Consider batch_size classes at a time
    # Load data directly from csv
    line_start = array_ix * batch_size
    line_stop = (1 + array_ix) * batch_size
    graph_data = read_data_from_bz2(line_start, line_stop, filename=filename, path_to_file=path_to_data)
    edge_lists = [d[1] for d in graph_data]

    with Pool() as p:
        out = p.map(func, edge_lists)

    # Flatten the output
    out_flat = [x for y in out for x in y]

    save_obj(out_flat, f'graph_performance{name_suffix}_batch{array_ix}', output_dir)

