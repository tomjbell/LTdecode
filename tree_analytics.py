import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from helpers import save_obj, load_obj
from os import getcwd


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
    return tot - 1


def get_tree_data(eta=0.9, max_branching=10, max_depth=5):
    data = []
    i = 0
    # branching ratio to vary between 1 and max_branching
    for b_rat in product(list(range(1, max_branching)), repeat=max_depth):
        i += 1
        n_q = tree_q_num(b_rat)
        eta_log = p_succ_tree(eta, b_rat)
        data.append((n_q, eta_log))
    return data


def best_tree_data(eta, max_branch, max_depth, plot=False, show=True, save=False, from_file=False):
    if from_file:
        bests = load_obj(f'best_trees_{eta}_max_branch_{max_branch}_depth_{max_depth}', getcwd()+'/best_graphs')
    else:
        data_full = []
        for r in range(1, max_depth+1):
            dn = get_tree_data(eta, max_branch, r)
            data_full += dn
        data_s = sorted(data_full, key=lambda x: x[0])
        bests = []
        low_q_winner = 0
        for d in data_s:
            if d[1] > low_q_winner:
                low_q_winner = d[1]
                bests.append(d)
        if save:
            save_obj(bests, f'best_trees_{eta}_max_branch_{max_branch}_depth_{max_depth}', getcwd()+'/best_graphs')
    if plot:
        plt.plot([b[0] for b in bests], [1 - b[1] for b in bests], '+')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of qubits')
        plt.ylabel('Effective loss')
    if show:
        plt.show()


if __name__ == '__main__':
    # for eta in [0.7, 0.8, 0.9, 0.95, 0.99]:
    #     best_tree_data(eta, 13, 5, plot=True, show=True, from_file=False, save=False)
    pass
