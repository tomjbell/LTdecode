import argparse
import sys
from multiprocessing import Pool, cpu_count
from helpers import load_obj, save_obj
from networkx import Graph
from decoder_class import FastDecoder, CascadeDecoder
import numpy as np
from error_correction import best_checks_max_clique, pauli_error_decoder
from logical_fusions import get_fusion_performance, fusion_threshold_from_dict


def graph_from_edges(edges):
    g = Graph()
    n_nodes = max([node for edge in edges for node in edge])
    g.add_nodes_from(list(range(n_nodes)))
    g.add_edges_from(edges)
    return g


def is_error_tolerant_multiple_bases(edge_list):
    ps = np.linspace(0.00001, 0.25)
    g = graph_from_edges(edge_list)

    ec_flags = {'x': False, 'y': False, 'z': False}
    for basis in ['x', 'y', 'z']:
        g_decoder = CascadeDecoder(g)
        t, m1, s1, s2 = g_decoder.decode(get_first_strat=True, first_traversal=True, mc=True, eff_meas_basis=basis, pathfinding=False)
        # print(m1.to_str())
        checks = best_checks_max_clique(g, m1)
        # print([c.to_str() for c in checks])
        prob, conf = pauli_error_decoder([m1], checks, ps)

        # Check if error is suppressed
        for i in range(len(ps)):
            if prob[i] > (1 - 2 * ps[i]) * 1.001:  # Require at least .1% better (avoid machine precision errors)
                ec_flags[basis] = True
                # print(prob[i], ps[i])
                break
    return edge_list, ec_flags


def graph_perf_on_batch(graph_list):
    results = []
    # c = 1
    for k in graph_list:
        # print(f'Inspecting graph {c}')
        # c+=1
        results.append(graph_perf(k))
    return results


def graph_perf(edges):
    g = graph_from_edges(edges)
    x = FastDecoder(g)
    spc = x.get_dict()
    return edges, spc


def graph_perf_pauli(edges):
    g = graph_from_edges(edges)
    dec = CascadeDecoder(g)
    bases = ['x', 'y', 'z']
    xr, yr, zr = [dec.get_dict(basis=b, cascading=False) for b in bases]
    return edges, xr, yr, zr


def graph_perf_cascading(edges):
    g = graph_from_edges(edges)
    dec = CascadeDecoder(g)
    bases = ['spc', 'x', 'y', 'z']
    spcr, xr, yr, zr = [dec.get_dict(basis=b, cascading=True) for b in bases]
    return edges, spcr, xr, yr, zr


def graph_perf_fusion(edges):
    g = graph_from_edges(edges)
    perf_dicts = get_fusion_performance(g, decoder_type='ACF')
    thresh = fusion_threshold_from_dict(perf_dicts, pf=0.5, take_min=False)
    return edges, perf_dicts, thresh


if __name__ == '__main__':
    # Parse the command line inputs
    parser = argparse.ArgumentParser(description="Pass variables for Loss tolerance measurement")
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
        "-od",
        "--output_dir",
        help="output directory",
        type=str,
        default="",
    )
    parser.add_argument(
        "-s",
        "--spc_basis",
        help="Do arbitrary basis measurement",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-t",
        "--test",
        help="If test reduce the number of graphs to 28 (n_cores)",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-pb",
        "--pauli_basis",
        help="Do pauli basis measurements",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-ec",
        "--error_correction",
        help="Check the error correcting properties of the graph code in the relevant basis",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-ts",
        "--test_size",
        help="How many graphs to analyse in the test job",
        type=int,
        default=28,
    )
    parser.add_argument(
        "-c",
        "--cascading",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-f",
        "--fusion",
        help="test logical fusion performance",
        action=argparse.BooleanOptionalAction
    )


    arguments = parser.parse_args(sys.argv[1:])
    array_ix = arguments.array_ix
    path_to_data = arguments.path_to_data
    output_dir = arguments.output_dir
    spc_basis = arguments.spc_basis
    pauli_basis = arguments.pauli_basis
    is_test = arguments.test
    do_error_correction = arguments.error_correction
    test_size = arguments.test_size
    cascading = arguments.cascading
    fusion = arguments.fusion
    assert not (spc_basis and pauli_basis)
    assert (pauli_basis or spc_basis or cascading or fusion)
    if fusion:
        assert not (cascading or spc_basis or pauli_basis)


    if output_dir == "":
        output_dir = path_to_data

    data = load_obj(name=f'graph_data_batch{array_ix}', path=path_to_data)
    edge_lists = [d[1] for d in data]
    if is_test:
        edge_lists = edge_lists[:test_size]
    if spc_basis:
        func = graph_perf
    elif pauli_basis:
        if do_error_correction:
            func = is_error_tolerant_multiple_bases
        else:
            func = graph_perf_pauli
    elif cascading:
        func = graph_perf_cascading
    elif fusion:
        func = graph_perf_fusion

    else:
        raise ValueError('No basis supplied')

    if spc_basis:
        name = f'graph_performance_batch{array_ix}'
    elif pauli_basis:
        if do_error_correction:
            name = f'graph_performance_pauli_ec_batch{array_ix}'
        else:
            name = f'graph_performance_pauli_batch{array_ix}'
    elif cascading:
        assert not (spc_basis or pauli_basis or do_error_correction)
        name = f'graph_performance_cascaded_batch{array_ix}'
    elif fusion:
        name = f'graph_performance_fusion_batch{array_ix}'

    with Pool() as p:
        out = p.map(func, edge_lists)
    print(name, output_dir)
    save_obj(out, name=name, path=output_dir)

