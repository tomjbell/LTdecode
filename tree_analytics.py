import numpy as np
import matplotlib.pyplot as plt


# Theoretical success probability for tree graphs from Phys. Rev. Lett. 97, 120501 (2006)

# Z_ind_tree(t, k, branch_list) is the probability of an indirect Z measurement on a qubit at layer k
# of a tree-graph described by branchings branch_list, in case the trasmittivity is t.
def Z_ind_tree(t, k, branch_list):
    max_layer_depth = len(branch_list)
    #     print(branch_list, k)
    if k < max_layer_depth:
        this_b = branch_list[k]
    else:
        return 0
    if k == (max_layer_depth - 1):
        b_next = 0
    else:
        b_next = branch_list[k + 1]
    return 1 - ((1 - (t * ((t + (1 - t) * Z_ind_tree(t, k + 2, branch_list)) ** b_next))) ** this_b)


# Z_tree(t, k, branch_list) is the probability of Z measurement (direct or indirect) on a qubit at layer k
# of a tree-graph described by branchings branch_list, in case the trasmittivity is t.
def Z_tree(t, k, branch_list):
    return t + (1 - t) * Z_ind_tree(t, k, branch_list)

# p_succ_tree(t, branch_list) is the total probability for decoding a tree-graph
#
def p_succ_tree(t, branch_list):
    if not branch_list:
        return 0
    elif len(branch_list) == 1:
        b0 = branch_list[0]
        p = t ** b0
    else:
        z_inds = [Z_ind_tree(t, k, branch_list) for k in (1, 2)]
        b0 = branch_list[0]
        b1 = branch_list[1]
        p = ((t + (1 - t) * z_inds[0]) ** b0 - ((1 - t) * z_inds[0]) ** b0) * (t + (1 - t) * z_inds[1]) ** b1
    return p


def tree_q_num(b_list):
    tot = 1
    num_last_layer = 1
    for b in b_list:
        num_last_layer *= b
        tot += num_last_layer
    return tot


def test_trees():
    pass


if __name__ == '__main__':

    transmissions = np.linspace(0, 1)
    # Analytic tree results
    for k in range(1, 20):
        # print(Z_ind_tree(0.9, k, b_list))
        out = [p_succ_tree(t, [4, 20, 5] * k) for t in transmissions]
        plt.plot(transmissions, out)
    plt.show()
    exit()

    nq = 5
    max_depth = 5
    transmissions = np.linspace(0, 1)
    from graphs import gen_star_graph, gen_ring_graph
    from decoder_class import CascadeDecoder
    from cascaded import CascadedResultPauli
    bases = ['spc', 'x', 'y', 'z', 'xy', 'z_direct']
    g = gen_ring_graph(nq)
    decoder = CascadeDecoder(g)
    for basis in bases:
        decoder.build_tree(basis=basis, ec=False, cascading=True)
    result = CascadedResultPauli([decoder.successful_outcomes], cascade_ix=[0] * max_depth)
    print(decoder.successful_outcomes)

    # Analytic tree results
    # for k in range(1, 6):
    #     # print(Z_ind_tree(0.9, k, b_list))
    #     out = [p_succ_tree(t, [nq-1] * k) for t in transmissions]
    #     plt.plot(transmissions, out)

    depolarising_noise = 0.
    out = [[] for _ in range(max_depth)]
    for t in transmissions:
        result.get_all_params(t, depolarising_noise, ec=False)
        for n_graphs in range(1, max_depth + 1):
            spc_prob = result.get_spc_prob(n_graphs)
            out[n_graphs - 1].append(spc_prob)
    for i in range(max_depth):
        plt.plot(transmissions, out[i])
    plt.plot((0, 1), (0, 1), 'k--')
    # plt.title('Cascaded SPF')
    # plt.show()


    # plt.plot((0, 1), (0, 1), 'k--')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend([f'{depth=}' for depth in range(1, 6)])
    plt.show()
    #
    # branches = [2, 1, 1]
    # plt.plot(transmissions, [p_succ_tree(t, branches) for t in transmissions])
    # plt.plot((0, 1), (0, 1), 'k--')
    # plt.show()
    # print(tree_q_num(branches))
    # print(p_succ_tree(0.9, branches))
    # print(Z_ind_tree(0.9, 0, [3, 1]))
