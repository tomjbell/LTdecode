from graphs import gen_ring_graph, gen_cascaded_graph, draw_graph, gen_fullyconnected_graph
from stab_formalism import stabilizers_from_graph, Stabilizer
import matplotlib.pyplot as plt
from cascaded import ConcatenatedResult
from decoder_class import CascadeDecoder
import numpy as np


def main():
    g5 = gen_ring_graph(5)
    g5_depth_2, nodes = gen_cascaded_graph([g5, g5])
    draw_graph(g5_depth_2)
    # plt.show()
    print(g5_depth_2.nodes())
    stab = Stabilizer(stabilizers_from_graph(g5))
    stab = Stabilizer(stabilizers_from_graph(g5_depth_2))
    # stab.x_meas(0)
    for q in nodes[1]:
        stab.x_meas(q)
    unmeasured_stab = [x for x in stab.generators if x.weight() > 1]
    measured_stab = [x for x in stab.generators if x.weight() == 1]
    print([s.to_str() for s in stab.generators])

    nu_gens = []
    for md in measured_stab:
        qubit_with_weight = list(set(md.z_ix).union(set(md.x_ix)))[0]
        print(qubit_with_weight)
        for other in unmeasured_stab:
            if other.xs[qubit_with_weight]:
                print(f'updated {qubit_with_weight}')

                other.update_xs(qubit_with_weight, 0)

    print([s.to_str() for s in unmeasured_stab])


def test_concat_triangles():
    """
    Check that concatenation of 3 qubit rings is the same as analysing the resultant graph directly
    :return:
    """
    g3 = gen_ring_graph(3)
    g, nodes = gen_cascaded_graph([g3, g3])
    draw_graph(g)
    plt.show()
    stab = Stabilizer(stabilizers_from_graph(g))
    for v in nodes[1]:
        stab.x_meas(v)
    unmeasured_stab = [x for x in stab.generators if x.weight() > 1]
    print([s.to_str() for s in unmeasured_stab])

    measured_stab = [x for x in stab.generators if x.weight() == 1]
    print([s.to_str() for s in stab.generators])

    nu_gens = []
    for md in measured_stab:
        qubit_with_weight = list(set(md.z_ix).union(set(md.x_ix)))[0]
        print(qubit_with_weight)
        for other in unmeasured_stab:
            if other.xs[qubit_with_weight]:
                print(f'updated {qubit_with_weight}')

                other.update_xs(qubit_with_weight, 0)

    print([s.to_str() for s in unmeasured_stab])
    # This is the fully connected 5 qubit graph, so lets explore the concatenated performance
    decoder = CascadeDecoder(g3)
    bases = ['x', 'y', 'z', 'xy', 'spc']
    for basis in bases:
        decoder.build_tree(basis=basis, ec=False)
    transmissions = np.linspace(0, 1)

    conc = ConcatenatedResult([decoder.successful_outcomes], cascade_ix=[0, 0])
    plt.plot(transmissions, [conc.teleportation_prob(t) for t in transmissions])

    decoder_full = CascadeDecoder(gen_fullyconnected_graph(5))
    decoder_full.build_tree(cascading=False)
    logical_transmission = [decoder_full.success_prob_outcome_list(t, depolarizing_noise=0, ec=False) for t in transmissions]
    plt.plot(transmissions, logical_transmission, 'o')
    plt.plot((0, 1), (0, 1), 'k--')
    plt.show()


def concat2pent():
    from networkx import Graph
    g = Graph()
    g.add_nodes_from(list(range(17)))
    g.add_edges_from([(0, 1), (0, 4), (0, 5), (0, 8), (0, 9), (0, 12), (0, 13), (0, 16), (1, 2), (1, 5), (1, 8),
                      (1, 13), (1, 16), (2, 3), (3, 4), (4, 5), (4, 8), (4, 13), (4, 16), (5, 6), (6, 7), (7, 8),
                      (9, 10), (9, 13), (9, 16), (10, 11), (11, 12), (12, 13), (12, 16), (13, 14), (14, 15), (15, 16)])
    draw_graph(g)


if __name__ == '__main__':
    test_concat_triangles()
