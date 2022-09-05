import os
import csv
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cascaded import AnalyticCascadedResult, ConcatenatedResultDicts, FastResult
from decoder_class import CascadeDecoder, FastDecoder
from graphs import gen_linear_graph, gen_ring_graph, gen_star_graph, draw_graph
from helpers import load_obj, save_obj, bisection_search, get_size
from random import randrange, choice, random
from time import time
from itertools import combinations, groupby
from multiprocessing import Pool
import bz2
from tqdm import tqdm


def tot_q(concat, type='cascaded'):
    """For a given concatenation, find the total number of required qubits"""
    layer_tots = [1]
    for x in concat:
        layer_tots.append(layer_tots[-1] * (x-1))
    if type == 'cascaded':
        return sum(layer_tots)
    elif type == 'concatenated':
        return 1 + layer_tots[-1]


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


def import_graphs(n_q, plot=False, equiv='Ci'):
    """
    'Ci' gives a single representative from each locally equivalent class
    'Li' gives all graphs
    """
    maxInt = sys.maxsize

    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    from helpers import save_obj
    a = input('Warning! this will overwrite saved data. Press x to continue')
    if a != 'x':
        return
    path = os.getcwd()
    path_to_file = path + '/LC_equiv_graph_data/'
    graph_list = []
    with open(path_to_file + f'{n_q}qubitorbitsCi.csv', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            lc_eq_class = []
            x = row[4].split("), ")
            for y in x:
                edge_list = []
                new = y.replace("(", "")
                n2 = new.replace(")", "")
                z = n2.split(", ")
                for ixs in z:
                    edge_list.append((int(ixs[0])-1, int(ixs[2])-1))
                lc_eq_class.append(edge_list)
            graph_list.append(lc_eq_class)
    save_obj(graph_list, f'{n_q}QubitCi', path + '/LC_equiv_graph_data')
    if plot:
        for eq_class in graph_list:
            for g in eq_class:
                graph = nx.Graph()
                graph.add_nodes_from([i for i in range(0, n_q)])
                graph.add_edges_from(g)
                nx.draw(graph)
                plt.show()


def graph_performance(n, lc_equiv=False, save_perf_dicts=False, mp=False, cascading=True, permute_input=False):
    from helpers import load_obj, save_obj
    path = os.getcwd() + '/LC_equiv_graph_data'

    if n == 2:
        g = nx.Graph()
        g.add_nodes_from([0, 1])
        g.add_edges_from(([(0, 1)]))
        x = CascadeDecoder(g)
        bases = ['spc', 'x', 'y', 'z', 'xy']
        spc, xr, yr, zr, xyr = [x.get_dict(basis=b, cascading=cascading) for b in bases]
        results = []
        res = []
        res.append(g.edges)
        if save_perf_dicts:
            for mr in spc, xr, yr, zr, xyr:
                res.append(mr)
            results.append(res)
            save_obj(results, f'{n}QubitResultDicts', path)
            return

    graph_data = load_obj(f'{n}QubitCi', path)
    n_classes = len(graph_data)
    c = 1
    results = []

    if not lc_equiv:
        for lc_class in graph_data:
            print(f'inspecting graph {c}/{n_classes}, number of qubits = {n}')
            c += 1
            edge_list = lc_class[0]  # Just choose the first graph
            if permute_input:
                for in_ix in range(n):
                    edges_permuted = permute_input_qubit(edge_list, in_ix)
                    results.append(analyse_graph(edges_permuted, n))
            else:
                results.append(analyse_graph(edge_list, n))
        if save_perf_dicts:
            save_obj(results, f'{n}QubitResultDicts', path)
        return results
    else:
        t0 = time()
        if mp:
            from multiprocessing import Pool
            pool = Pool()
            mp_res = []
            classnum = 0
            for lc_class in graph_data:
                classnum += 1
                print(f'Class {classnum}/{len(graph_data)}')
                mp_res.append(pool.apply_async(inspect_class, args=(lc_class, n)))
            results_nest = [p.get() for p in mp_res]
            results = [item for sublist in results_nest for item in sublist]
        else:
            classnum = 0
            for lc_class in graph_data:
                classnum += 1
                print(f'Class {classnum}/{len(graph_data)}')
                results += inspect_class(lc_class, n)
        t1 = time()
        print(t1-t0)
        if save_perf_dicts:
            save_obj(results, f'{n}QubitResultDicts_incl_equiv', path)
        return results


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


def plotting(n):
    from helpers import load_obj
    depths = [3]
    path = os.getcwd() + '/LC_equiv_graph_data'
    list_of_dicts = load_obj(f'{n}QubitResultDicts', path)
    all_outs = []
    for graph in list_of_dicts:
        edges, spcr, xr, yr, zr, xyr = graph
        r = AnalyticCascadedResult(spcr, xr, yr, zr, xyr)
        etas = np.linspace(0, 1)
        for depth in depths:
            eta_eff = [r.get_spc_prob(eta, depth) for eta in etas]
            all_outs.append(eta_eff)
            plt.plot(etas, eta_eff)
    plt.plot((0, 1), (0, 1), 'k--')
    plt.show()
    data = np.array(all_outs)
    maxs = np.amax(data, axis=0)
    plt.plot(np.linspace(0, 1), maxs)
    plt.plot((0, 1), (0, 1), 'k--')
    plt.show()


def get_best_perf(min_q, max_q, eta=0.99, find_threshold=False, prefix_num=None, printing=True, graph_data_12q=False):
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
                prefix = f"graph_performance_batch{prefix_num}"
            else:
                prefix = "graph_performance_batch"
            filenames = [f[:-4] for f in os.listdir(path) if f.startswith(prefix)]
        else:
            path = os.getcwd() + f'/data/spc_data'
            filenames = [f"{n}_qubit_performance"]

        best_subthresh_graph = None
        best_threshold_graph = None
        best_dist_graph = None
        max_subthresh = 0
        min_threshold = 1
        max_dist = 1

        ix = 0
        for file in tqdm(filenames):
            list_of_dicts = load_obj(file, path)
            if printing:
                print(f"Graphs loaded, filesize={get_size(list_of_dicts)}")
            for graph in list_of_dicts:
                edges = graph[0]
                spcr = graph[1]
                r = FastResult(spcr)

                # Find threshold
                def func(t):
                    return r.get_spc_prob(t) - t
                try:
                    thresh = bisection_search((0.5, 0.95), func)
                except ValueError:
                    # print('NO ROOT')
                    thresh = 1
                if thresh <= min_threshold:
                    min_threshold = thresh
                    best_threshold_graph = graph
                # Get subthreshold performance
                spc = r.get_spc_prob(eta)
                l1e3 = 1 - r.get_spc_prob(1 - 0.001)
                l1e5 = 1 - r.get_spc_prob(1 - 0.00001)
                distance = (np.log10(l1e3) - np.log10(l1e5)) / 2
                if distance > max_dist:
                    best_dist_graph = graph
                    max_dist = distance

                if spc > max_subthresh:
                    best_subthresh_graph = graph
                    max_subthresh = spc
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
    # gen_data_points(1000, 0.9, biased=False, trees_only=False, max_q=9, best_top_layer_only=True)
    gen_data_points(50000, 0.9, biased=False, min_q=3, trees_only=False, max_q=11, best_top_layer_only=False, max_depth=5, concatenated=True)
    print('exiting')
    exit()
    # plotting(7)


def graph_perf_spc_only(graph_info):
    class_size = graph_info[0]
    edges = graph_info[1]
    g = nx.Graph()
    n = max([i for edge in edges for i in edge])
    g.add_nodes_from(list(range(n)))
    g.add_edges_from(edges)
    x = FastDecoder(g)
    spc = x.get_dict()
    return edges, spc, class_size


def graph_perf_pauli(graph_info):
    class_size = graph_info[0]
    edges = graph_info[1]
    g = nx.Graph()
    n = max([i for edge in edges for i in edge])
    g.add_nodes_from(list(range(n)))

    # print(edges)
    g.add_edges_from(edges)
    dec = CascadeDecoder(g)
    bases = ['x', 'y', 'z']
    xr, yr, zr = [dec.get_dict(basis=b, cascading=False) for b in bases]
    return edges, xr, yr, zr


def get_graph_perf_dicts(n, spc_only=True, mp=False):
    """

    :param path_to_graph_list:
    :param spc_only:
    :return:
    """
    graph_data = load_obj(f'{n}_qubit_graphs_ordered_num_in_class', path='data/uib_data')
    if spc_only:
        if mp:
            with Pool() as p:
                data_out = p.map(graph_perf_spc_only, graph_data)
        else:
            data_out = [graph_perf_spc_only(g) for g in graph_data]

        save_obj(data_out, f"{n}_qubit_performance", "data/spc_data")
    else:
        if mp:
            with Pool() as p:
                data_out = p.map(graph_perf_pauli, graph_data)
        else:
            data_out = [graph_perf_pauli(g) for g in graph_data]
        save_obj(data_out, f"{n}_qubit_performance", "data/pauli_data")


def scatter_perf_class_size(n, eta=0.99, spc_only=True):
    data = load_obj(f'{n}_qubit_performance', "data/spc_data")
    log_loss = []
    class_sizes = []
    for item in data:
        r = FastResult(item[1])
        log_loss.append(np.log10(1-r.get_spc_prob(eta)))
        class_sizes.append(item[2])
    plt.scatter(log_loss, class_sizes)
    plt.xlabel(f'Logical loss, log scale')
    plt.ylabel('Class size')
    plt.title(f"{n} qubits, physical loss {np.round(1-eta, 5)}")
    plt.show()


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


if __name__ == '__main__':
    for n in range(4, 10):
        get_graph_perf_dicts(n, spc_only=False, mp=True)




