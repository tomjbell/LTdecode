from multiprocessing import Pool, cpu_count
from decoder_class import CascadeDecoder, FastDecoder
from os import getcwd
from networkx import Graph
from time import time
from helpers import load_obj, save_obj
from numpy import array_split
from logical_fusions import AdaptiveFusionDecoder
from graph_finding import permute_input_qubit, txt_to_edge_data, gen_non_isomorphic_graphs


def graph_performance_fast(n, chunk_size=None):
    # Do batches of 100 graphs
    graph_list = txt_to_edge_data(f'{n}_qubit_graphs_data_uib.txt')
    all_graphs_data = []
    for e in graph_list:
        all_graphs_data += gen_non_isomorphic_graphs(e)

    graph_data = all_graphs_data
    if chunk_size is None:
        cpus = cpu_count()
        n_graphs = len(graph_data)
        all_graph_ixs = list(range(n_graphs))
        input_lists = [list(x) for x in array_split(all_graph_ixs, cpus)]
        graph_lists = [[graph_data[ix] for ix in input_lists[i]] for i in range(cpus)]  # Take only the first from each equivalence class
    else:
        graph_lists = [graph_data[i:i + chunk_size] for i in range(0, len(graph_data), chunk_size)]

    return graph_lists


def graph_perf_on_batch(graph_list):
    results = []
    c = 1
    for k in graph_list:
        print(f'Inspecting graph {c}')
        c+=1
        results.append(graph_perf(k))
    return results


def graph_perf_logical_fusion(edges):
    g = Graph()
    g.add_edges_from(edges)
    x = AdaptiveFusionDecoder(g)
    x.build_tree()
    return x.get_threshold(pfail=0.5)


def graph_perf(edges):
    g = Graph()
    n = max([i for edge in edges for i in edge])
    g.add_nodes_from(list(range(n)))

    # print(edges)
    g.add_edges_from(edges)

    x = FastDecoder(g)
    spc = x.get_dict()
    return edges, spc


if __name__ == '__main__':
    n=10

    # Do first 11 x 100
    n_cpu = cpu_count() - 1
    graph_lists = graph_performance_fast(n, chunk_size=20)
    n_chunks = len(graph_lists)
    if n_chunks % n_cpu:
        n_batches = n_chunks // n_cpu + 1
    else:
        n_batches = n_chunks / n_cpu

    for i in range(1, n_batches):
        if i < n_batches - 1:
            batch = graph_lists[i * n_cpu:(i+1) * n_cpu]
        else:
            batch = graph_lists[(n_batches - 1) * n_cpu:]

        with Pool(11) as p:
            out = p.map(graph_perf_on_batch, batch)

        save_obj(out, f'{n}QubitResultsDicts_PermuteInputFastDecoder_batch{i}', getcwd() + '/graph_perf_10q')
