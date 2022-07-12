from multiprocessing import Pool, cpu_count
from decoder_class import CascadeDecoder
from os import getcwd
from networkx import Graph
from time import time
from helpers import load_obj, save_obj
from numpy import array_split


def graph_performance(n, lc_equiv=False, save_perf_dicts=False, mp=False, cascading=True):
    from helpers import load_obj, save_obj
    path = getcwd() + '/LC_equiv_graph_data'


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
            g = Graph()
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


def graph_pef_on_batch(graph_list):
    results = []
    c = 1
    for k in graph_list:
        print(f'Inspecting graph {c}')
        c+=1
        results.append(graph_perf(k))
    return results


def graph_perf(edges):
    g = Graph()
    g.add_nodes_from(list(range(7)))
    # print(edges)
    g.add_edges_from(edges)

    x = CascadeDecoder(g)
    bases = ['spc', 'x', 'y', 'z', 'xy']
    spc, xr, yr, zr, xyr = [x.get_dict(basis=b) for b in bases]
    return edges, spc, xr, yr, zr, xyr


if __name__ == '__main__':
    t0 = time()
    n = 8
    lc_equiv = False
    save_perf_dicts = False
    mp = False
    cascading = False
    save = False
    n_threads = cpu_count()

    path = getcwd() + '/LC_equiv_graph_data'
    graph_data = load_obj(f'{n}QubitCi', path)
    n_classes = len(graph_data)
    all_graph_ixs = list(range(n_classes))
    if n_classes % n_threads:
        per_thread = n_classes // n_threads + 1
    else:
        per_thread = n_classes / n_threads
    input_lists = [list(x) for x in array_split(all_graph_ixs, n_threads)]
    print(n_classes, per_thread, n_threads)
    print(input_lists)
    graph_lists = [[graph_data[ix][0] for ix in input_lists[i]] for i in range(n_threads)]  # Take only the first from each equivalence class
    print(graph_lists)

    with Pool(n_threads) as p:
        out = p.map(graph_pef_on_batch, graph_lists)
    t1 = time()
    print(out)
    print(t1 - t0)
