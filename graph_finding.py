import os
import csv
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cascaded import AnalyticCascadedResult, ConcatenatedResultDicts, FastResult, CascadedResultPauli
from decoder_class import CascadeDecoder, FastDecoder
from graphs import gen_linear_graph, gen_ring_graph, gen_star_graph, draw_graph, graph_from_edges
from helpers import load_obj, save_obj, bisection_search, get_size
from random import randrange, choice, random, randint
from time import time
from itertools import combinations, groupby, permutations
from multiprocessing import Pool
import bz2
from tqdm import tqdm
from logical_fusions import get_fusion_peformance, fusion_threshold_from_dict
from tree_analytics import best_tree_data
from functools import partial


def tot_q(concat, type='cascaded'):
    """
    For a given concatenation, find the total number of required qubits
    Code qubits only
    """
    layer_tots = [1]
    for x in concat:
        layer_tots.append(layer_tots[-1] * (x-1))
    if type == 'cascaded':
        return sum(layer_tots) - 1
    elif type == 'concatenated':
        return layer_tots[-1]


def fit_best_trees(capture, deg, max_x=None):
    path_to_data = os.getcwd() + '/stef_data'
    for i in os.listdir(path_to_data):
        if os.path.isfile(os.path.join(path_to_data, i)) and i.startswith(
                f"TreeGraph_LTvsQubitNum_random_t{capture}_MaxLayers"):
            print(i)
            tree_data = np.loadtxt(open(os.path.join(path_to_data, i), "rb"), delimiter=",")
            print(tree_data)
            xdat, ydat = [], []
            for i in range(len(tree_data[0, :])):
                logx = np.log(tree_data[0, :][i])
                logy = np.log(tree_data[1, :][i])
                if max_x is None or logx < max_x:
                    xdat.append(logx)
                    ydat.append(logy)
            fit = np.polyfit(xdat, ydat, deg=deg)
    return fit


def plot_best_trees(capture):
    """From the datafiles in data plot the best tree graph performance for given loss"""
    path_to_data = os.getcwd() + '/stef_data'
    for i in os.listdir(path_to_data):
        if os.path.isfile(os.path.join(path_to_data, i)) and i.startswith(f"TreeGraph_LTvsQubitNum_random_t{capture}_MaxLayers"):
            tree_data = np.loadtxt(open(os.path.join(path_to_data, i), "rb"), delimiter=",")
            plt.plot(tree_data[0, :], tree_data[1, :], 's-')


def test_random_graphs(n, n_samples=4):
    def gnp_random_connected_graph(n, p):
        """
        Generates a random undirected graph, similarly to an Erdős-Rényi
        graph, but enforcing that the resulting graph is conneted
        """
        edges = combinations(range(n), 2)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        if p <= 0:
            return G
        if p >= 1:
            return nx.complete_graph(n, create_using=G)
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            random_edge = choice(node_edges)
            G.add_edge(*random_edge)
            for e in node_edges:
                if random() < p:
                    G.add_edge(*e)
        return G
    for _ in range(n_samples):
        weight = -1
        while 0>weight or 1<weight:
            weight = np.random.normal(loc=0.5, scale=0.5)
        print(weight)
        g = gnp_random_connected_graph(n, weight)
        nx.draw(g)
        plt.show()
        x = CascadeDecoder(g)
        dicts = [x.get_dict(b) for b in ('spc', 'x', 'y', 'z', 'xy')]
        print('got dictionaries')
        r = AnalyticCascadedResult([dicts], [0])
        etas = np.linspace(0, 1)
        plt.plot(etas, [r.get_spc_prob(t, 0) for t in etas])
    plt.plot((0, 1), (0, 1), 'k--')
    plt.show()


def analyse_graph(edges, n):
    g = nx.Graph()
    g.add_nodes_from(list(range(n)))
    g.add_edges_from(edges)

    x = CascadeDecoder(g)
    bases = ['spc', 'x', 'y', 'z', 'xy']
    spc, xr, yr, zr, xyr = [x.get_dict(basis=b) for b in bases]
    return edges, spc, xr, yr, zr, xyr


def gen_non_isomorphic_graphs(edge_list_in):
    """
    From a graph represented by the edge_list_in, return the set of graphs that are inequivalent under permutation of the code (non-input) qubits
    :param edge_list_in:
    :return: list of edge_lists of the unique graphs
    """
    g = nx.Graph()
    n = max([i for j in edge_list_in for i in j]) + 1
    g.add_nodes_from(list(range(n)))
    g.add_edges_from(edge_list_in)

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


def inspect_class(lc_class, n):
    c=1
    class_size = len(lc_class)
    result = []
    for graph in lc_class:
        print(f'inspecting graph {c}/{class_size}')
        c += 1
        result.append(analyse_graph(graph, n))
    return result


def get_best_perf(min_q, max_q, eta=0.99, find_threshold=False, prefix_num=None, printing=True, graph_data_12q=False, pauli=False, subdirname=None):
    """
    Find the best spc graphs for the top layer.
    Because we are only looking at one layer only consider one representative graph from each class
    Find threshold and subthreshold performance, and code distance
    """
    best_graphs_dict = {}
    for n in range(min_q, max_q + 1):
        if n > 9:
            if graph_data_12q:
                path = os.getcwd() + '/graph_data_12q'
            else:
                path = os.getcwd() + f'/graphs_batched_{n}q'
            if prefix_num is not None:
                if pauli:
                    prefix = f"graph_performance_pauli_batch{prefix_num}"
                else:
                    prefix = f"graph_performance_batch{prefix_num}"
            else:
                if pauli:
                    prefix = "graph_performance_pauli_batch"
                else:
                    prefix = "graph_performance_batch"
            filenames = [f[:-4] for f in os.listdir(path) if f.startswith(prefix)]
        else:
            if pauli:
                path = os.getcwd() + f'/data/pauli_data'
            else:
                path = os.getcwd() + f'/data/spc_data'
            if subdirname is not None:
                path += f'/{subdirname}'
            filenames = [f"{n}_qubit_performance"]

        best_subthresh_graph = None
        best_threshold_graph = None
        best_dist_graph = None
        max_subthresh = 0
        min_threshold = 1
        max_dist = 1

        ix = 0
        for file in filenames:
            list_of_dicts = load_obj(file, path)
            if printing:
                print(f"Graphs loaded, filesize={get_size(list_of_dicts)}")
            for graph in list_of_dicts:
                edges = graph[0]
                if pauli:
                    xr = graph[1]
                    yr = graph[2]
                    zr = graph[3]
                    results = [xr, yr, zr]
                else:
                    spcr = graph[1]
                    results = [spcr]
                subthresh, threshold, distance = [], [], []
                for basis_r in results:
                    r = FastResult(basis_r)

                    # Find threshold
                    def func(t):
                        return r.get_spc_prob(t) - t
                    try:
                        thresh = bisection_search((0.4, 0.98), func)
                    except ValueError:
                        # print('NO ROOT')
                        thresh = 1
                    threshold.append(thresh)
                    # Get subthreshold performance
                    spc = r.get_spc_prob(eta)
                    subthresh.append(spc)
                    l1e3 = 1 - r.get_spc_prob(1 - 0.001)
                    l1e5 = 1 - r.get_spc_prob(1 - 0.00001)
                    dist = (np.log10(l1e3) - np.log10(l1e5)) / 2
                    distance.append(dist)
                d = min(distance)
                st = min(subthresh)
                t = max(threshold)
                if d > max_dist:
                    best_dist_graph = (edges, results[:])
                    max_dist = d
                if t <= min_threshold:
                    min_threshold = t
                    best_threshold_graph = (edges, results[:])
                if st > max_subthresh:
                    best_subthresh_graph = (edges, results[:])
                    max_subthresh = st
            if printing:
                print(f'File number #{ix} complete, current bests: {max_subthresh=}, {min_threshold=}, {max_dist=}')
            ix += 1
        best_graphs_dict[n] = (best_subthresh_graph, max_subthresh, best_threshold_graph, min_threshold, best_dist_graph, max_dist)
        print(best_graphs_dict[n])
    return best_graphs_dict


def gen_data_points(n_samples=100, t=0.9, min_q=3, max_q=9, max_depth=4, biased=False, trees_only=False, lc_equiv=False,
                    best_top_layer_only=True, only_winning_graphs=True, concatenated=False):
    if best_top_layer_only:
        bests = get_best_perf(min_q, max_q, eta=t)
    if only_winning_graphs:
        fit_func = get_fit_func(t)
    n_qubit_lists = {}
    x = []
    y = []
    best_graphs = []
    for n in range(min_q, max_q + 1):
        path = os.getcwd() + '/LC_equiv_graph_data'
        if lc_equiv:
            name = f'{n}QubitResultDicts_incl_equiv'
        else:
            name = f'{n}QubitResultDicts'
        list_of_dicts = load_obj(name, path)
        n_qubit_lists[n] = list_of_dicts
    print([len(n_qubit_lists[i]) for i in range(min_q, max_q+1)])
    for i in range(n_samples):

        # pick a random number of qubits for each layer of the graph
        #TODO bias by the number of graphs there are at each qubit number?
        if biased:
            nu = [np.sqrt(x) for x in range(min_q, max_q+1)]
            tot = sum(nu)
            scaled = [x/tot for x in nu]
            ixs = list(np.random.choice(len(scaled), size=max_depth, replace=True, p=scaled))
            q_num_list = [min_q + ix for ix in ixs]
        else:
            q_num_list = [randrange(min_q, max_q+1) for _ in range(max_depth)]

        # Pick random graphs of each qubit number
        if trees_only and not best_top_layer_only:
            graph_dict_lists = [n_qubit_lists[q][0][1:] for q in q_num_list]  # The first graphs are ghz i.e. trees
        elif trees_only and best_top_layer_only:
            # print(bests[q_num_list[0]][1:])
            graph_dict_lists = [bests[q_num_list[0]][1:]] + [n_qubit_lists[q][0][1:] for q in q_num_list[1:]]
        elif not trees_only and best_top_layer_only:
            graph_dict_lists = [bests[q_num_list[0]][1:]] + [n_qubit_lists[q][randrange(0, len(n_qubit_lists[q]))][1:] for q in q_num_list[1:]]
        else:
            graph_dict_lists = [n_qubit_lists[q][randrange(0, len(n_qubit_lists[q]))][1:] for q in q_num_list]

        n_extra_layers = [i for i in range(max_depth)]
        if concatenated:
            for d in n_extra_layers:
                r = ConcatenatedResultDicts(graph_dict_lists, cascade_ix=list(range(d+1)))
                eff_loss = 1 - r.teleportation_prob(t)
                nq = tot_q(q_num_list[:d+1], type='concatenated')
                if not only_winning_graphs or eff_loss < fit_func(nq):
                    x.append(nq)
                    y.append(eff_loss)
                    if q_num_list[:d+1] not in best_graphs:
                        best_graphs.append(q_num_list[:d+1])

        else:
            r = AnalyticCascadedResult(graph_dict_lists)
            layer_y = []
            for d in n_extra_layers:
                eff_loss = 1 - r.get_spc_prob(t, depth=d + 1)
                nq = tot_q(q_num_list[:d+1])
                if not only_winning_graphs or eff_loss < fit_func(nq):
                    x.append(nq)
                    y.append(eff_loss)
                    if q_num_list[:d+1] not in best_graphs:
                        best_graphs.append(q_num_list[:d+1])

    plot_best_trees(t)
    plt.plot(x, y, '+')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of qubits')
    plt.ylabel('Effective loss')
    plt.xlim(1.9, 1000)
    plt.ylim(1e-11, 1-t + 0.01)
    plt.plot((0, 10000), (1-t, 1-t), 'k--')
    plt.title(r'$\eta$ = ' + str(t))
    plt.savefig('Concatenated_vs_trees')
    plt.show()
    print(best_graphs)
    print(set(y))


def testing_trees(n_samples=100, t=0.9, min_q=2, max_q=10, max_depth=6, enforce_pattern=False):
    res_dict = {}
    for n in range(min_q, max_q+1):
        print('calculating graph n = ', n)
        graph = gen_star_graph(n)

        x = CascadeDecoder(graph)
        spc, x, y, z, xy = [x.get_dict(basis=b) for b in ['spc', 'x', 'y', 'z', 'xy']]
        res_dict[n] = (spc, x, y, z, xy)

    x = []
    y = []
    dict_list = [res_dict[n] for n in range(min_q, max_q+1)]
    if enforce_pattern:
        n_samples=1
    for i in range(n_samples):
        if enforce_pattern:
            q_list = enforce_pattern
        else:
            # Generate random list of graph sizes
            q_list = [randrange(min_q, max_q+1) for _ in range(max_depth)]
        ix_list = [q-min_q for q in q_list]

        n_extra_layers = [i for i in range(max_depth)]
        r = AnalyticCascadedResult(dict_list, ix_list)
        # r = AnalyticCascadedResult(gtop[0], gbot[1], gbot[2], gbot[3], gbot[4])
        layer_y = []
        for d in n_extra_layers:
            eff_loss = 1 - r.get_spc_prob(t, depth=d + 1)
            y.append(eff_loss)
            layer_y.append(eff_loss)
            x.append(tot_q(q_list[:d+1]))
        print(q_list, layer_y, x)
    plot_best_trees(t)
    plt.plot(x, y, '+')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of qubits')
    plt.ylabel('Effective loss')
    plt.show()


def get_fit_func(eta):
    """
    Get an expression for the best trees curve so you can look at only the graphs that win
    """
    deg = 5
    fit = fit_best_trees(eta, deg=deg, max_x=9)

    def fit_func(n_q):
        x = np.log(n_q)
        y = 0
        for i in range(deg + 1):
            y += fit[i] * x ** (deg - i)
        return np.exp(y)
    return fit_func


def read_data_from_bz2(line_start, lines_stop, filename, path_to_file):
    full_path = os.path.join(path_to_file, filename)
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


def txt_to_edge_data(filename):
    """
    Read the files taken from http://www.ii.uib.no/~larsed/entanglement/ and return a list of the edges of graphs
    :param filename:
    :return:
    """
    output = []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        x = line.split('\t')
        # for a in x:
        #     print(a)
        n_graphs = int(x[1])
        schmidt = x[5]
        edge_list_ix = 10
        for ix in range(len(x)):
            if x[ix] == 'yes' or x[ix] == 'no':
                edge_list_ix = ix + 1

        edge_list = x[edge_list_ix]
        y = edge_list.split(')')
        edges = [(int(e[1+4*j]), int(e[3+4*j])) for e in y for j in range(len(e)//4)]
        # print(edges)
        output.append(edges)
    return output


def main():
    pass


def graph_perf_spc_only(graph_info):
    class_size = graph_info[0]
    edges = graph_info[1]
    g = graph_from_edges(edges)
    x = FastDecoder(g)
    spc = x.get_dict()
    return edges, spc, class_size


def graph_perf_pauli(graph_info):
    """
    :param graph_info: tuple (class_size, edgelist) of the representative graph of the equivalence class
    :return: tuple (edgelist, x, y, z results dictionaries)
    """
    class_size = graph_info[0]
    edges = graph_info[1]
    g = graph_from_edges(edges)
    dec = CascadeDecoder(g)
    bases = ['x', 'y', 'z']
    xr, yr, zr = [dec.get_dict(basis=b, cascading=False) for b in bases]
    return edges, xr, yr, zr


def graph_perf_cascading(graph_info):
    """
    :param graph_info: tuple (class_size, edgelist) of the representative graph of the equivalence class
    :return: tuple (edgelist, teleportation, x, y, z results dictionaries including cascading)
    """
    class_size = graph_info[0]
    edges = graph_info[1]
    g = graph_from_edges(edges)
    dec = CascadeDecoder(g)
    bases = ['spc', 'x', 'y', 'z']
    spcr, xr, yr, zr = [dec.get_dict(basis=b, cascading=True) for b in bases]
    return edges, spcr, xr, yr, zr


def graph_perf_fusion(graph_info):
    """
    :param graph_info: tuple (class_size, edgelist) of the representative graph of the equivalence class
    :return: tuple (edgelist, teleportation performance dictionary, fusion threshold for 6 ring resourced RHG lattice)
    """
    class_size = graph_info[0]
    edges = graph_info[1]
    g = graph_from_edges(edges)
    perf_dicts = get_fusion_peformance(g, decoder_type='ACF')
    thresh = fusion_threshold_from_dict(perf_dicts, pf=0.5)
    return edges, perf_dicts, thresh


def get_graph_perf_dicts(n, spc_only=True, mp=False, cascading=False, fusion=False, new_data_flag=False, new_dirname=None, save=True):
    """
    Analyse a set of graphs for some type of measurement
    :param n: number of qubtis
    :param spc_only: only analyse performance for arbitrary basis measurement
    :param mp: bool - do multiproccessing using Pool
    :param cascading: allow indirect Z measurements upon failure
    :param fusion: do logical fusion
    :param new_data_flag: save in new file
    :param new_dirname: new directory
    :return:
    """
    graph_data = load_obj(f'{n}_qubit_graphs_ordered_num_in_class', path='data/uib_data')
    outfile_name = f"{n}_qubit_performance"

    def save_data(data, outfile, dec_type=None):
        if dec_type == 'spc':
            suffix = 'spc_data'
        elif dec_type == 'pauli':
            suffix = 'pauli_data'
        elif dec_type == 'cascaded':
            suffix = 'cascaded'
        elif dec_type == 'fusion':
            suffix = 'fusion'
        else:
            raise ValueError
        if new_data_flag:
            path = os.getcwd() + f'/data/{suffix}/{new_dirname}'
            try:
                os.mkdir(path)
            except FileExistsError:
                pass
        else:
            path = os.getcwd() + f'/data/{suffix}'
        save_obj(data, outfile, path)

    if fusion:
        if mp:
            with Pool(9) as p:
                data_out = p.map(graph_perf_fusion, graph_data)
        else:
            data_out = [graph_perf_fusion(g) for g in graph_data]
        save_data(data_out, outfile_name, dec_type='fusion')
    elif spc_only:
        if mp:
            with Pool() as p:
                data_out = p.map(graph_perf_spc_only, graph_data)
        else:
            data_out = [graph_perf_spc_only(g) for g in graph_data]
        save_data(data_out, outfile_name, dec_type='spc')
    elif not cascading:
        if mp:
            with Pool() as p:
                data_out = p.map(graph_perf_pauli, graph_data)
        else:
            data_out = [graph_perf_pauli(g) for g in graph_data]
        save_data(data_out, outfile_name, dec_type='pauli')
    elif cascading:
        if mp:
            with Pool() as p:
                data_out = p.map(graph_perf_cascading, graph_data)
        else:
            data_out = [graph_perf_cascading(g) for g in graph_data]
        if save:
            save_data(data_out, outfile_name, dec_type='cascaded')


def get_distance(spcr, plot=True, show_plot=True):
    low_loss_etas = np.linspace(0.9, 0.9999)
    r = FastResult(spcr)
    eta_log = [r.get_spc_prob(t) for t in low_loss_etas]
    log_x = [np.log(1 - t) for t in low_loss_etas]
    log_y = [np.log(1 - tl) for tl in eta_log]
    grad = (log_y[-1] - log_y[0]) / (log_x[-1] - log_x[0])
    if plot:
        plt.plot(log_x, log_y)
    if show_plot:
        plt.show()
    return grad


def get_2d_data(graph, losses, noises, bases, save=False, plot=True):
    """
    :param graph:
    :param losses:
    :param noises:
    :param bases:
    :param save:
    :param plot:
    :param spf: if doing spf you need to consider the error on the output qubit separately, which is 3p, but that is also
    the case for direct transmission.
    :return:
    """
    n_dat_x = len(losses)
    n_dat_y = len(noises)
    dec = CascadeDecoder(graph)
    probs_arrays = {'spc': np.zeros((n_dat_x, n_dat_y)), 'x': np.zeros((n_dat_x, n_dat_y)), 'y': np.zeros((n_dat_x, n_dat_y)), 'z': np.zeros((n_dat_x, n_dat_y))}
    acc_arrays = {'spc': np.zeros((n_dat_x, n_dat_y)), 'x': np.zeros((n_dat_x, n_dat_y)), 'y': np.zeros((n_dat_x, n_dat_y)), 'z': np.zeros((n_dat_x, n_dat_y))}
    prod_arrays = {'spc': np.zeros((n_dat_x, n_dat_y)), 'x': np.zeros((n_dat_x, n_dat_y)), 'y': np.zeros((n_dat_x, n_dat_y)), 'z': np.zeros((n_dat_x, n_dat_y))}

    x, y = np.meshgrid(losses, noises)

    for basis in bases:
        dec.build_tree(basis, ec=True, cascading=False, printing=False)
        print(len(dec.successful_outcomes[basis]))

        for i in tqdm(range(n_dat_x)):
            for j in range(n_dat_y):
                prob, acc = dec.success_prob_outcome_list(1-losses[i], 4 * noises[j], basis=basis, ec=True)
                if basis == 'spc':
                    acc *= (1 - 3 * noises[j])
                probs_arrays[basis][j, i] = prob
                acc_arrays[basis][j, i] = acc

                prod_arrays[basis][j, i] = (1 - prob * acc)
        # print(x)
        # print(y)
        # print(prod_arrays[basis])
        # print(x[0, 0], y[0, 0], prod_arrays[basis][0, 0])
        print(acc_arrays['spc'])
        if plot:
            plt.contourf(x, y, prod_arrays[basis])
            plt.colorbar()
            plt.xlabel('loss')
            plt.ylabel('x,y,z error probability')
            plt.title(f'{basis=}')
            plt.show()
    if save:
        save_obj((x, y, probs_arrays['spc'], acc_arrays['spc'], prod_arrays['spc']), 'dec_5ring_spf_perf_ec_lt', os.getcwd())


def indices_best_graphs(nu=True, save=False):
    bp_name = 'best_pauli_graphs_5-11_qubits'
    bspf_name = 'best_spf_graphs_5-11_qubits'
    inds_name = 'indices_of_bests_5-11'
    if nu:
        bp_name += '_dated_11_10'
        bspf_name += '_dated_11_10'
        inds_name += '_dated_11_10'
    bp = load_obj(bp_name, os.getcwd() + '/best_graphs')
    bspf = load_obj(bspf_name, os.getcwd() + '/best_graphs')
    st_ixs = {'spf': {k: None for k in bspf.keys()}, 'pauli': {k: None for k in bp.keys()}}
    t_ixs = {'spf': {k: None for k in bspf.keys()}, 'pauli': {k: None for k in bp.keys()}}
    draw_graph(bspf[6][0][0], from_edges=True)

    for k in bp.keys():
        edges_subthresh_spf = bspf[k][0][0]
        edges_thresh_spf = bspf[k][2][0]
        edges_subthresh_p = bp[k][0][0]
        edges_thresh_p = bp[k][2][0]
        if k < 10:
            n_fix = 1
        elif k == 10:
            n_fix = 26
        elif k == 11:
            n_fix = 82
        else:
            raise ValueError
        for fix in range(n_fix):
            if k < 10:
                nq_gs = load_obj(f'{k}_qubit_graphs_ordered_num_in_class', os.getcwd() + '/data/uib_data')
            else:
                nq_gs = load_obj(f'graph_data_batch{fix}', os.getcwd() + f'/graphs_batched_{k}q')
            for i in range(len(nq_gs)):
                if nq_gs[i][1] == edges_subthresh_spf:
                    st_ixs['spf'][k] = (fix, i)
                if nq_gs[i][1] == edges_thresh_spf:
                    t_ixs['spf'][k] = (fix, i)
                if nq_gs[i][1] == edges_subthresh_p:
                    st_ixs['pauli'][k] = (fix, i)
                if nq_gs[i][1] == edges_thresh_p:
                    t_ixs['pauli'][k] = (fix, i)
    full = {'subthresh': st_ixs, 'thresh': t_ixs}
    if save:
        save_obj(full, inds_name, os.getcwd())
    # print(st_ixs)
    # print(t_ixs)


def concat_perf(list_of_concats, etas, do_perms=True):
    """
    test all concatenation orders of the graphs whose results dictionaries are given in list of concats
    :param perf_dicts: list of list of tuples [[(n_qubits, performance dict), (...), ...], ...]
    :param eta:
    :return: data dictionary whose outer keys are transmissions and inner keys are bases
    """
    n_graphs = len(list_of_concats[0])
    bases = ['spc', 'x', 'y', 'z']
    data_conc = {t: {b: [] for b in bases} for t in etas}
    data_casc = []
    #iterate through all the concatenations in list_of_concats
    for perf_dicts in list_of_concats:
        # generate all possible permutation orders
        if do_perms:
            ix_orderings = list(permutations(range(n_graphs)))
        else:
            ix_orderings = [list(range(n_graphs))]
        for ix_ordering in ix_orderings:
            concat_of_perfs = [perf_dicts[i][1] for i in ix_ordering]
            # print(len(concat_of_perfs))
            # print(len(concat_of_perfs[0]))
            # print(concat_of_perfs)
            rcon = ConcatenatedResultDicts(concat_of_perfs)
            for eta in etas:
                rcon.calc_params(eta)
                for depth in range(n_graphs):
                    nq_conc = tot_q([perf_dicts[ix][0] for ix in ix_ordering[depth:]], type='concatenated')
                    nq_casc = tot_q([perf_dicts[ix][0] for ix in ix_ordering[depth:]], type='concatenated')
                    out_conc = []
                    out_casc = []
                    for b in bases:
                        data_conc[eta][b].append((nq_conc, rcon.meff_prob(eta, depth, basis=b, calc_params=False)))
                    # data_conc[eta].append((nq_conc, out_conc))
    return data_conc


def homog_cascades_pauli_spf_bests(max_n=10):
    bases = ['spc', 'x', 'y', 'z']
    spc_graphs_to_test = load_obj('best_spf_graphs_5-11_qubits', os.getcwd() + '/best_graphs')
    pauli_graphs = load_obj('best_pauli_graphs_5-11_qubits', os.getcwd() + '/best_graphs')

    def data_list_from_raw(obj):
        data = []
        for nq, perf in obj.items():
            if nq < max_n + 1:
                e = perf[0][0]
                graph = nx.Graph()
                graph.add_nodes_from(list(range(max([a for b in e for a in b]))))
                graph.add_edges_from(e)
                dec = CascadeDecoder(graph)
                graph_expr_dicts = (graph.number_of_nodes(), [dec.get_dict(b, cascading=False) for b in bases])
                data.append(graph_expr_dicts)
        return data
    data = data_list_from_raw(spc_graphs_to_test) + data_list_from_raw(pauli_graphs)
    return data


def casc_perf(list_of_concats, etas, do_perms=True, spc_only=True):
    """
    test all cascade orders of the graphs whose results dictionaries are given in list of concats
    :param perf_dicts: list of list of tuples [[(n_qubits, performance dict), (...), ...], ...]
    :param eta:
    :return:
    """
    n_graphs = len(list_of_concats[0])
    if spc_only:
        bases = ['spc']
    else:
        bases = ['spc', 'x', 'y', 'z']
    data_casc = {t: {b: [] for b in bases} for t in etas}

    # iterate through all the cacades in list_of_concats
    for perf_dicts in list_of_concats:
        # generate all possible permutation orders
        if do_perms:
            ix_orderings = list(permutations(range(n_graphs)))
        else:
            ix_orderings = [list(range(n_graphs))]
        for ix_ordering in ix_orderings:
            concat_of_perfs = [perf_dicts[i][1] for i in ix_ordering]
            # print(len(concat_of_perfs))
            # print(len(concat_of_perfs[0]))
            # print(concat_of_perfs)
            rcasc = CascadedResultPauli(concat_result_list=concat_of_perfs, ec=False, from_dict=True)
            for eta in etas:
                rcasc.get_all_params(eta, depolarising_noise=0, ec=False)
                for depth in range(n_graphs):
                    nq_casc = tot_q([perf_dicts[ix][0] for ix in ix_ordering[depth:]], type='cascaded')
                    for b in bases:
                        data_casc[eta][b].append((nq_casc, rcasc.get_spc_prob(maxdepth=n_graphs-depth)))

    return data_casc


def random_cascades(minq, maxq, n_shots=1000, bests_only=False, max_depth=5):
    output_data = []
    num_gs = {4: 4, 5: 10, 6: 31, 7: 110, 8: 497, 9: 2845}
    graph_perf = {nq: (load_obj(f'{nq}_qubit_performance', os.getcwd() + '/data/spc_data'),
                       load_obj(f'{nq}_qubit_performance', os.getcwd() + '/data/pauli_data')) for nq in num_gs.keys()}
    # print(graph_perf[4][1])
    for _ in range(n_shots):
        nq_list = [randint(minq, maxq) for _ in range(max_depth)]
        graph_ixs = [randint(0, num_gs[j] - 1) for j in nq_list]
        graph_perf_dicts = [(nq_list[i], [graph_perf[nq_list[i]][0][graph_ixs[i]][1]] + list(graph_perf[nq_list[i]][1][graph_ixs[i]][1:]))
                            for i in range(max_depth)]
        output_data.append(graph_perf_dicts)
        # print(nq_list)
        # print(graph_ixs)
        # print(graph_perf_dicts)
        # print(graph_perf_dicts[0])
    return output_data


def best_cascade_graphs():
    ################## Get the best pauli and spf graphs up to 11 qubits under cascades  ###########################
    # for the 11 qubit graph need to re-run the decoder as pickle file corrupted
    from script_blue_crystal_lt_ec import graph_perf_cascading

    best_graphs_casc_pauli = {'subthresh': {n: None for n in range(5, 12)},
                              'thresh': {n: None for n in range(5, 12)}}
    best_graphs_casc_spf = {'subthresh': {n: None for n in range(5, 12)}, 'thresh': {n: None for n in range(5, 12)}}

    indices = load_obj('indices_of_bests_5-11', os.getcwd())

    for n in range(5, 12):
        if n < 10:
            for g_type in ['subthresh', 'thresh']:
                ixp = indices[g_type]['pauli'][n][1]
                ixspf = indices[g_type]['spf'][n][1]
                best_graphs_casc_pauli[g_type][n] = load_obj(f'{n}_qubit_performance', os.getcwd() + '/data/cascaded')[ixp]
                best_graphs_casc_spf[g_type][n] = load_obj(f'{n}_qubit_performance', os.getcwd() + '/data/cascaded')[ixspf]
        elif n == 10:
            for g_type in ['subthresh', 'thresh']:
                fixp = indices[g_type]['pauli'][n][0]
                ixp = indices[g_type]['pauli'][n][1]
                fixspf = indices[g_type]['spf'][n][0]
                ixspf = indices[g_type]['spf'][n][1]
                best_graphs_casc_pauli[g_type][n] = \
                load_obj(f'graph_performance_cascaded_batch{fixp}', os.getcwd() + f'/graphs_batched_{n}q')[ixp]
                best_graphs_casc_spf[g_type][n] = \
                load_obj(f'graph_performance_cascaded_batch{fixspf}', os.getcwd() + f'/graphs_batched_{n}q')[ixspf]
        elif n == 11:
            for g_type in ['subthresh', 'thresh']:
                fixp = indices[g_type]['pauli'][n][0]
                ixp = indices[g_type]['pauli'][n][1]
                fixspf = indices[g_type]['spf'][n][0]
                ixspf = indices[g_type]['spf'][n][1]
                pauli_g_edges = load_obj(f'graph_data_batch{fixp}', os.getcwd() + f'/graphs_batched_{n}q')[ixp][1]
                spf_g_edges = load_obj(f'graph_data_batch{fixspf}', os.getcwd() + f'/graphs_batched_{n}q')[ixspf][1]
                print(f'Getting {g_type} performance for spf, 11q')
                best_graphs_casc_pauli[g_type][n] = graph_perf_cascading(pauli_g_edges)
                print(f'Getting {g_type} performance for pauli, 11q')
                best_graphs_casc_spf[g_type][n] = graph_perf_cascading(spf_g_edges)

    full = {'spf': best_graphs_casc_spf, 'pauli': best_graphs_casc_pauli}
    save_obj(full, 'cascaded_perf_best_graphs_5-11', os.getcwd() + '/best_graphs')

#################################################################################################################


if __name__ == '__main__':
    pass


    ######################## THE FOLLOWING IS FOR DOING MULTIPROCCESSING OF CONCATENATED GRAPH SEARCH ############

    # minq = 4
    # maxq = 9
    # n_shot_in = 1000
    # n_shot_out = 10
    # ks = list(range(n_shot_out))
    # max_depth = 5
    # etas = (0.7, 0.8, 0.9, 0.95, 0.99)
    # bases = ['spc', 'x', 'y', 'z']
    # out_dicts_global = {eta: {b: [] for b in bases} for eta in etas}
    # t = time()
    # with Pool(n_shot_out) as p:
    #     out = p.map(partial(per_round, min_q=minq, max_q=maxq, etas=etas, n_shots=n_shot_in, max_depth=max_depth), ks)
    # print(time() - t)
    # for item in out:
    #     for key in item:
    #         for b in bases:
    #             # print(key, out_dicts_global[key])
    #
    #             out_dicts_global[key][b] += item[key][b]
    #
    # for eta in etas:
    #     best_tree_data(eta, 13, 5, plot=True, show=False, from_file=True)
    #     for basis in ['spc']:
    #         data_full = out_dicts_global[eta][basis]
    #         # print(data_full)
    #         plt.plot([b[0] for b in data_full], [1 - b[1] for b in data_full], '+')
    #         plt.title(f'{basis=}, {eta=}')
    #     plt.show()

    ################################################################################################################




