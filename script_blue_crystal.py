import argparse
import sys
from multiprocessing import Pool, cpu_count
from helpers import load_obj, save_obj
from networkx import Graph
from decoder_class import FastDecoder
from generate_input_files import generate_data_files


def graph_perf_on_batch(graph_list):
    results = []
    # c = 1
    for k in graph_list:
        # print(f'Inspecting graph {c}')
        # c+=1
        results.append(graph_perf(k))
    return results


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

    arguments = parser.parse_args(sys.argv[1:])
    array_ix = arguments.array_ix
    path_to_data = arguments.path_to_data

    data = load_obj(name=f'graph_data_batch{array_ix}', path=path_to_data)
    edge_lists = [d[1] for d in data]

    with Pool() as p:
        out = p.map(graph_perf, edge_lists)

    save_obj(out, f'graph_performance_batch{array_ix}', path_to_data)

