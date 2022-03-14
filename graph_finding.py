import os
import csv
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cascaded import AnalyticCascadedResult
# from CodesFunctions.decode_success_funcs import plot_best_trees, tot_q
from decoder_class import CascadeDecoder
from graphs import gen_linear_graph, gen_ring_graph, gen_star_graph
from helpers import load_obj
from random import randrange, choice, random
from time import time
from itertools import combinations, groupby


def tot_q(concat):
    """For a given concatenation, find the total number of required qubits"""
    layer_tots = [1]
    for x in concat:
        layer_tots.append(layer_tots[-1] * (x-1))
    return sum(layer_tots)


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
            plt.plot(tree_data[0, :], tree_data[1, :], '--')


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


def graph_performance(n, lc_equiv=False, save_perf_dicts=False, mp=False):
    from helpers import load_obj, save_obj
    path = os.getcwd() + '/LC_equiv_graph_data'

    if n == 2:
        g = nx.Graph()
        g.add_nodes_from([0, 1])
        g.add_edges_from(([(0, 1)]))
        x = CascadeDecoder(g)
        bases = ['spc', 'x', 'y', 'z', 'xy']
        spc, xr, yr, zr, xyr = [x.get_dict(basis=b) for b in bases]
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
            res = []
            edge_list = lc_class[0]  # Just choose the first graph
            res.append(edge_list)
            g = nx.Graph()
            g.add_nodes_from([i for i in range(n)])
            g.add_edges_from(edge_list)

            x = CascadeDecoder(g)
            bases = ['spc', 'x', 'y', 'z', 'xy']
            spc, xr, yr, zr, xyr = [x.get_dict(basis=b) for b in bases]
            for mr in spc, xr, yr, zr, xyr:
                res.append(mr)
            results.append(res)
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
        return results, t1-t0


def inspect_class(lc_class, n):
    c=1
    class_size = len(lc_class)
    result = []
    for graph in lc_class:
        print(f'inspecting graph {c}/{class_size}')
        c += 1
        res = []
        edge_list = graph
        res.append(edge_list)
        g = nx.Graph()
        g.add_nodes_from([i for i in range(n)])
        g.add_edges_from(edge_list)
        x = CascadeDecoder(g)
        bases = ['spc', 'x', 'y', 'z', 'xy']
        spc, xr, yr, zr, xyr = [x.get_dict(basis=b) for b in bases]
        for mr in spc, xr, yr, zr, xyr:
            res.append(mr)
        result.append(res)
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


def get_best_perf(min_q, max_q, eta):
    """
    Find the best spc graphs for the top layer.
    Becasue we are only looking at one layer only consider one representative graph from each class
    """
    from helpers import load_obj
    path = os.getcwd() + '/LC_equiv_graph_data'
    best_graphs_dict = {}
    for n in range(min_q, max_q + 1):
        list_of_dicts = load_obj(f'{n}QubitResultDicts', path)
        best_spc = None
        max_spc = 0
        for graph in list_of_dicts:
            edges, spcr, xr, yr, zr, xyr = graph
            r = AnalyticCascadedResult(spcr, xr, yr, zr, xyr)
            spc = r.get_spc_prob(eta, 1)
            if spc > max_spc:
                best_spc = graph
                max_spc = spc
        best_graphs_dict[n] = graph
    return best_graphs_dict


def gen_data_points(n_samples=100, t=0.9, min_q=3, max_q=9, max_depth=4, biased=False, trees_only=False, lc_equiv=False,
                    best_top_layer_only=True, only_winning_graphs=True):
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
            q_num_list = [randrange(min_q, max_q) for _ in range(max_depth)]

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
    plt.show()
    print(best_graphs)


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


def main():

    # for n in [8]:
    #     r, t = graph_performance(n, lc_equiv=True, save_perf_dicts=True, mp=True)
    # exit()
    # exit()
    # from helpers import load_obj, save_obj
    # path = os.getcwd() + '/LC_equiv_graph_data'
    #
    # # Get 3 qubit graph performance
    # ring3 = gen_ring_graph(3)
    # lin3 = gen_linear_graph(3)
    # print(ring3.edges)
    # # exit()
    # from decoder_class import CascadeDecoder
    # q3res = []
    # for graph in (ring3, lin3):
    #     r_list = [ring3.edges]
    #     x = CascadeDecoder(graph)
    #     for b in ['spc', 'x', 'y', 'z', 'xy']:
    #         r_list.append(x.get_dict(basis=b))
    #     q3res.append(r_list)
    # save_obj(q3res, '3QubitResultDicts', path)
    # exit()

    from time import time

    # gen_data_points(1000, 0.9, biased=False, trees_only=False, max_q=9, best_top_layer_only=True)
    gen_data_points(1000, 0.95, biased=False, trees_only=False, max_q=9, best_top_layer_only=False, max_depth=4)
    print('exiting')
    exit()



    # plotting(7)


if __name__ == '__main__':
    main()
    n = 5
    g = gen_ring_graph(n)
    nx.draw(g)
    plt.show()
    x = CascadeDecoder(g)
    # xy_dict = x.get_dict('xy')
    # print(xy_dict)
    #
    #
    # exit()
    dicts = [x.get_dict(b) for b in ('spc', 'x', 'y', 'z', 'xy')]
    # print(dicts[1])
    # print(dicts[2])
    # print(dicts[4])
    print('got dictionaries')
    r = AnalyticCascadedResult([dicts], [0] * 5)
    etas = np.linspace(0, 1)
    for depth in range(1, 6):
        plt.plot(etas, [r.get_spc_prob(t, depth) for t in etas])
    plt.plot(etas, etas, 'k--')
    plt.legend([1, 2, 3, 4, 5])
    plt.xlabel('Physical transmission probability')
    plt.ylabel('Teleportation success rate')
    plt.title(f'Cascaded {n} qubit ring')
    plt.show()
    exit()


    # main()
    # testing_trees(1, 0.9, max_q=3, min_q=3, max_depth=13)
    # # for n in [2]:
    # #     graph_performance(n, save_perf_dicts=True)
    # exit()
    test_random_graphs(10, n_samples=2)
    exit()
    main()
