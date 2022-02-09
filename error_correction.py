from itertools import combinations, product
from pauli_class import pauli_prod
import numpy as np
from math import factorial


def pauli_error_decoder(targets, checks, ps, max_weight=2, ignore=(0,), lost_qubits=None, printing=False):
    """
    This can take a arbitrary number of target paulis and an arbitrary number of check paulis
    """
    if not targets:
        return [no_ec_flip_rate([len(t.support) for t in targets], p) for p in ps], None
    if lost_qubits is None:
        lost_qubits = []
    # Initialise variables that we will need, and data structures for storing the results
    results = []
    n_q = targets[0].n
    q_with_errors = [i for i in range(n_q)]
    for i in list(ignore) + lost_qubits:
        q_with_errors.remove(i)
    n_error_q = len(q_with_errors)
    paulis_measured = targets + checks

    # Generate functions to find the probability of a given flip pattern
    pauli_union = paulis_measured[0].copy()
    for i in range(1, len(paulis_measured)):
        pauli_union = pauli_union.union(paulis_measured[i])

    # What measurement do we do on each qubit? Lets us to find the probability of flipped outcomes given the error model
    meas_type_dict = {}  # 0 = I, 1 = X, 2 = Y, 3 = Z
    for ix in q_with_errors:
        if ix not in pauli_union.support:
            meas_type_dict[ix] = 0
        elif pauli_union.xs == 1:
            if pauli_union.zs == 1:
                meas_type_dict[ix] = 2
            else:
                meas_type_dict[ix] = 1
        else:
            meas_type_dict[ix] = 3

    def prob_of_error_pattern(ixs, x, y, z):
        """ Find the probability of an anticommuting error on qubits ixs, given an error model px, py, pz
        """
        pflips = {0: 0, 1: y + z, 2: x + z, 3: x + y}
        num_e_types = [0, 0, 0, 0]
        num_no_e_types = [0, 0, 0, 0]
        for ix in q_with_errors:
            if ix in ixs:
                num_e_types[meas_type_dict[ix]] += 1
            else:
                num_no_e_types[meas_type_dict[ix]] += 1
        prob = np.prod([pflips[j] ** num_e_types[j] * (1 - pflips[j]) ** num_no_e_types[j] for j in range(4)])
        return prob
    conf_with_p = []
    for p in ps:
        # Initially assume errors equally likely
        px = py = pz = p
        # Iterate through every combination of measurement outcome flips
        # For each, find the probability of occurrence, the flips on the target stabilizers, and the syndrome.
        syndrome_prob_dicts = {}
        synd_tot_probs = {}
        win_prob = 0
        for flip_pat in product((0, 1), repeat=n_error_q):
            # 1 Find the probability
            error_ixs = [q_with_errors[i] for i in range(n_error_q) if flip_pat[i]]
            prob = prob_of_error_pattern(error_ixs, px, py, pz)
            # 2 Find the syndrome
            syndrome = tuple([sum([x in ch.support for x in error_ixs]) % 2 for ch in checks])
            # 3 Find the error pattern
            error = tuple([sum([x in t.support for x in error_ixs]) % 2 for t in targets])
            if printing:
                print(f'{flip_pat=}')
                print(f'{error_ixs=}')
                print(f'{prob=}')
                print(f'{syndrome=}')
                print(f'{error=}')

            if syndrome not in syndrome_prob_dicts.keys():
                syndrome_prob_dicts[syndrome] = {x: 0 for x in product((0, 1), repeat=len(targets))}
                synd_tot_probs[syndrome] = 0
            syndrome_prob_dicts[syndrome][error] += prob
            synd_tot_probs[syndrome] += prob
        confidence = {}
        for syndrome in syndrome_prob_dicts.keys():
            max_value = max(syndrome_prob_dicts[syndrome].values())  # maximum value
            confidences_dict = {k: v/synd_tot_probs[syndrome] for k, v in syndrome_prob_dicts[syndrome].items()}
            confidence[syndrome] = confidences_dict
            max_keys = [k for k, v in syndrome_prob_dicts[syndrome].items() if v == max_value]
            # print(syndrome, max_keys)
            win_prob += max_value
        # print(f'{win_prob=}')
        results.append(win_prob)
        conf_with_p.append(confidence)
    return results, conf_with_p


def no_ec_flip_rate(weights, p):
    """
    get the probability of at least one measurement having a flipped outcome
    :param weights: list. The number of qubits in the support of each measurement.
    :param p: float. The probability of each individual pauli error for a dephasing channel lambda = 4p
    """
    pflip = 2 * p  # 2 anticommuting operators
    qflip = 1 - pflip
    win = 1
    for weight in weights:
        success_prob = 1 - sum([factorial(weight) / factorial(k) / factorial(weight - k) * pflip ** k * qflip ** (weight - k) for k in
                   range(weight + 1) if k % 2 == 1])
        win *= success_prob
    return 1 - win


def best_checks(graph, target, banned_qs=None, printing=False, stab_nt=None):
    """
    :param graph: graph to calculate stabilizers from
    :param target: the target operator that we want to check
    :param banned_qs: if any qubits have been lost, or are the input or outputs they cannot be measured
    :param printing: print status updates for debugging
    """
    if banned_qs is None:
        banned_qs = [0]
    nq = graph.number_of_nodes()
    if printing:
        print(target.to_str())
    # find compatible stabilizers
    if stab_nt is None:
        from stab_formalism import stabilizers_from_graph, gen_stabs_from_generators
        gens = stabilizers_from_graph(graph)
        # We are only interested in the non-trivial stabilizers - trivial stabilizers with non-overlapping supports give us no new info
        stab_t, stab_nt = gen_stabs_from_generators(gens, split_triviality=True)
    compat_stabs = [s for s in stab_nt if (not set(banned_qs).intersection(set(s.support)) and target.commutes_each(s, [i for i in range(nq)]))]
    if len(compat_stabs) == 0:
        return []

    # generate a list of all the possible strategies - each strategy may involve multiple stabilizers.
    num_stabs = len(compat_stabs)
    stab_dict = {i:compat_stabs[i] for i in range(num_stabs)}
    input_strats = [(ix,) for ix in range(num_stabs)]
    all_strats = input_strats[:]
    set_empty = False
    while not set_empty:
        next_set = get_next_possibles(input_strats, num_stabs, stab_dict)
        if len(next_set) == 0:
            break
        else:
            all_strats += next_set
            input_strats = next_set
    # Get the largest subgroup
    longest_checklist_ix = np.argmax(np.array([len(i) for i in all_strats]))
    longest_checks = all_strats[longest_checklist_ix]
    check = [stab_dict[longest_checks[i]] for i in range(len(longest_checks))]
    # find a generating set
    gen_checks = generating_set(check)
    if printing:
        print([p.to_str() for p in check])
        print([p.to_str() for p in gen_checks])
    return gen_checks


def get_next_possibles(n_poss, tot_n_singles, paulis_dict):
    """ Find the sets of compatible pauli measurements with n+1 different stabilizers, from the sets of possible n stabilizer
    measurements, and the set of single stabilizers
    :param n_poss: are tuples of the indices labelling the different stabilizers e.g. (1, 4) -> s1 and s4
    :param tot_n_singles: is the number of different stabilizers
    :param paulis_dict: maps the index to the corresponding pauli operator"""
    out_list = []
    for ixs in n_poss:
        pauli = paulis_dict[ixs[0]]
        for ix in ixs[1:]:
            p2 = pauli.union(paulis_dict[ix])
            pauli = p2

        # See if it is compatible with the other paulis
        for j in range(max(ixs)+1, tot_n_singles):
            if pauli.commutes_each(paulis_dict[j], [i for i in range(pauli.n)]):
                out_list.append(ixs + (j,))
    return out_list


def generating_set(plist):
    """
    Find a minimal generating set for the pauli subgroup containing the elements in plist
    """
    gen_set = [0]
    ixs = [i for i in range(1, len(plist))]
    min_not_spanned = len(plist) - 1

    def spans(gens_ix, group):
        not_spanned = set(group)
        for r in range(1, len(gens_ix)+1):
            for g_combos in combinations(gens_ix, r):
                to_remove = set()
                for s in not_spanned:
                    if pauli_prod([plist[x] for x in g_combos]).equivalent(s):
                        to_remove.add(s)
                not_spanned -= to_remove
        return len(not_spanned)

    if spans(gen_set, plist) == 0:
        return [plist[x] for x in gen_set]
    for ix in ixs:

        num_not_spanned = spans(gen_set + [ix], plist)
        if num_not_spanned < min_not_spanned:
            gen_set.append(ix)
            min_not_spanned = num_not_spanned
        # print(min_not_spanned)
        if min_not_spanned == 0:
            return [plist[x] for x in gen_set]
    raise ValueError


def main():
    pass


if __name__ == '__main__':
    main()
