from itertools import combinations, product
from pauli_class import Pauli, Strategy, pauli_prod


def gen_bipartitions(vals):
    combos = []
    for r in range(1, len(vals)//2 + 1):
        seen = set()
        for x in combinations(vals, r):
            seen.add(x)
            y = tuple(set(vals) - set(x))
            if y not in seen:
                combos.append((x, y))
    return combos


def gen_stabs_new(stab_gens):
    """
    New implementation to do triviality testing - possibly a slow way of doing it
    :param stab_gens:
    :return:
    """
    stab_grp = {}
    triv = []
    non_triv = []
    n_gens = len(stab_gens)
    nq = stab_gens[0].n
    q_list = list(range(nq))
    gen_ixs = list(range(n_gens))

    for r in range(1, n_gens+1):
        for gen_combo in combinations(gen_ixs, r):
            new_pauli = pauli_prod([stab_gens[i] for i in gen_combo])
            stab_grp[gen_combo] = new_pauli

    def is_trivial(p):
        trivial = False
        q_list = list(p.support)
        x_str = p.xs
        z_str = p.zs
        # print(x_str)
        for bp in gen_bipartitions(q_list):
            bp_size = len(bp[0])
            p2 = p.copy()
            for ix in bp[1]:
                p2.update_xs(ix, 0)
                p2.update_zs(ix, 0)
            for p3 in stab_grp.values():
                if len(p3.support) == bp_size:
                    if p2.equivalent(p3):
                        return True
        return False

    for k, v in stab_grp.items():
        if is_trivial(v):
            triv.append(v)
        else:
            non_triv.append(v)
    return triv, non_triv


def gen_stabs_from_generators(stab_gens, split_triviality=False):
    """
    Create a dictionary to track the indices of the stabilizer generators that you have included
    if split_triviality, return a list of the trivial and non-trivial stabilizers separately
    """
    stab_grp = {}
    n_gens = len(stab_gens)
    for i in range(n_gens):
        stab_grp[frozenset([i])] = stab_gens[i]
    while max([len(key) for key in stab_grp]) < n_gens:
        for key, val in stab_grp.items():
            nu_dict = {}
            for j in range(n_gens):
                if key.union([j]) not in stab_grp:
                    nu_dict[key.union([j])] = val.mult(stab_gens[j])
            stab_grp = {**stab_grp, **nu_dict}
    if split_triviality:
        triv, non_triv = [], []
        for key, val in stab_grp.items():
            k = list(key)
            tot_support = set()
            ix = 0
            trivial = True
            if len(k) == 1:
                trivial = False
            while ix < len(k) and trivial:
                intersections = tot_support & set(stab_gens[k[ix]].support)
                if len(intersections) != 0:
                    trivial = False
                tot_support = tot_support.union(set(stab_gens[k[ix]].support))
                ix += 1
            if trivial:
                triv.append(val)
            else:
                non_triv.append(val)
        return triv, non_triv
    else:
        full_stab_grp = [val for key, val in stab_grp.items()]
        return full_stab_grp


def get_meas_pattern(zls, xls, targets):
    """
    Just get the first pair of operators that anticommute on the target and commute elsewhere
    """
    n = zls[0].n
    m = []
    for t in targets:
        print(t)
        z_tried = 0
        pair_found = False
        for op in zls:
            z_tried += 1
            print(f'z ops tried: {z_tried}')
            #  Find a measurement that doesn't require a measurement on your other target qubits
            other_targets_in = False
            for s in targets:
                if s != t:
                    if op.zs[s] == 1 or op.xs[s] == 1:
                        other_targets_in = True
            if not other_targets_in:
                if op.zs[t] == 1 or op.xs[t] == 1:
                    x_ops_tried = 0
                    for op2 in xls:
                        x_ops_tried += 1
                        print(f'x ops tried = {x_ops_tried}')
                        commuting_ix = [i for i in range(n)]
                        commuting_ix.remove(t)
                        if op.commutes_each(op2, commuting_ix) and not op.commutes_each(op2, [t]):
                            z_meas = [i for i in range(n) if op.zs[i] == 1 or op2.zs[i] == 1]
                            x_meas = [i for i in range(n) if op.xs[i] == 1 or op2.xs[i] == 1]
                            z_meas.remove(t)
                            x_meas.remove(t)
                            nu_m = Pauli(z_meas, x_meas, n, (op.i_exp + op2.i_exp) % 4)
                            if is_compatible(nu_m, m):
                                m.append(nu_m)
                                print('op found')
                                print('breaking inner')
                                pair_found = True
                                break
            if pair_found:
                print(op.to_str(), op2.to_str())
                print('breaking outer')
                break
    return m


def is_compatible(nu_m, current_m, lost_qubits=None, no_supersets=True):
    """
    is the measurement nu_m compatible with the other measurement current_m?
    does it measure Zi where m has Xi or vice versa, but ZiXi and ZiXi is allowed
    the only useful compatible measurements are those whose support does not include one of the qubits in the support
    of the original measurement. i.e. the support of the original is not a subset of the support of the new.
    If the new measurement is a superset of the current, you would never switch to it
    """
    if lost_qubits is not None:  # If it relies on qubits that have already been lost, it won't work
        if len(set(lost_qubits) & set(nu_m.support)) != 0:
            return False
    if no_supersets and set(current_m.support).issubset(set(nu_m.support)):
        return False
    for i in range(nu_m.n):
        if ((current_m.zs[i] == 1 and nu_m.xs[i] == 1) and not (current_m.xs[i] == 1 and nu_m.zs[i] == 1)) or \
                ((current_m.xs[i] == 1 and nu_m.zs[i] == 1) and not (current_m.zs[i] == 1 and nu_m.xs[i] == 1)):
            return False
    return True


def stabilizers_from_graph(g):
    """
    from the networkx graph, use the edge list to find the neighbours of each qubit, and use this to generate the stabilizer group
    """
    neighbour_list = []
    for node in g.nodes:
        neighbour_list.append((node, [n for n in g.neighbors(node)]))
    stab_gens = [Pauli(node[1], [node[0]], g.number_of_nodes(), 0) for node in neighbour_list]
    return stab_gens


def get_measurements(stab_grp, t, get_individuals=False, new_implementation=False):
    """
    Find the set of possible measurements that will teleport the state from the input qubit (canonically zero) to the
    target output t
    """
    num_q = stab_grp[0].n
    commuting_ix = [i for i in range(1, num_q)]
    commuting_ix.remove(t)

    if new_implementation:
        bases = [p for p in product(('x', 'y', 'z'), repeat=2)]

        def get_other_bases(b):
            return [b2 for b2 in bases if b2[0] != b[0] and b2[1] != b[1] and b < b2]

        #Split in to x y z type and compare with stabilizers of the other types?
        # stab_pairs_new = []
        measurements = []
        m_type_dict = {b: [] for b in bases}
        for s in stab_grp:
            if (s.zs[0] or s.xs[0]) and (s.zs[t] or s.xs[t]):
                m_type_dict[(s.get_meas_type(0), s.get_meas_type(t))].append(s)
        for k1 in m_type_dict.keys():
            for k2 in get_other_bases(k1):
                for s1 in m_type_dict[k1]:
                    for s2 in m_type_dict[k2]:
                        if s1.commutes_each(s2, commuting_ix):
                            # stab_pairs_new.append((s1, s2))
                            z_meas = [i for i in range(1, num_q) if s1.zs[i] == 1 or s2.zs[i] == 1]
                            x_meas = [i for i in range(1, num_q) if s1.xs[i] == 1 or s2.xs[i] == 1]
                            z_meas.remove(t)
                            x_meas.remove(t)
                            s1_ = s1.copy()
                            s2_ = s2.copy()
                            for s in (s1_, s2_):
                                for qix in (0, t):
                                    s.update_xs(qix, 0)
                                    s.update_zs(qix, 0)
                            measurements.append((Pauli(z_meas, x_meas, num_q, 0), s1_, s2_))

        # Return only unique measurements
        # Alternative unique element finding (FASTER)
        a_dict = {tuple(p[0].xs + p[0].zs): p for p in measurements}
        nu_meas = [a_dict[key] for key in a_dict.keys()]
        strats = [Strategy(m[0], t, s1=m[1], s2=m[2]) for m in nu_meas]
        strats.sort(key=lambda x: x.pauli.weight(), reverse=False)
        return strats
    else:

        # reduce to the stabilizers that are non-identity on input and output
        stab_grp_reduced = [s for s in stab_grp if (s.zs[0] or s.xs[0]) and (s.zs[t] or s.xs[t])]
        n_stab_reduced = len(stab_grp_reduced)
        stab_pairs = []
        for i, j in combinations([i for i in range(len(stab_grp_reduced))], 2):
            if stab_grp_reduced[i].commutes_each(stab_grp_reduced[j], commuting_ix) and not (stab_grp_reduced[i].commutes_each(stab_grp_reduced[j], [t]) or stab_grp_reduced[i].commutes_each(stab_grp_reduced[j], [0])):
                stab_pairs.append((i, j))

        measurements = []
        if get_individuals:
            # Record both of the individual stabilizers and the union of them
            for pair in stab_pairs:
                op3 = stab_grp_reduced[pair[0]].copy()
                op4 = stab_grp_reduced[pair[1]].copy()

                  # Set the operators on the input and output qubits to identity
                for q in (0, t):
                    op3.update_zs(q, 0)
                    op3.update_xs(q, 0)
                    op4.update_zs(q, 0)
                    op4.update_xs(q, 0)
                a = op3.union(op4)
                measurements.append([op3, op4, a])

            m_dict = {tuple(m[2].xz_mat.flatten()): m for m in measurements}
            unique_meas = [value for value in m_dict.values()]
            strats = [Strategy(m[2], t, m[0], m[1]) for m in unique_meas]
            return strats

        else:
            # Now take the union of those two operators to give the possible measurement patterns
            for pair in stab_pairs:
                op1 = stab_grp_reduced[pair[0]]
                op2 = stab_grp_reduced[pair[1]]
                z_meas = [i for i in range(1, num_q) if op1.zs[i] == 1 or op2.zs[i] == 1]
                x_meas = [i for i in range(1, num_q) if op1.xs[i] == 1 or op2.xs[i] == 1]
                z_meas.remove(t)
                x_meas.remove(t)

                measurements.append(Pauli(z_meas, x_meas, num_q, 0))
            # Return only unique measurements
            # Alternative unique element finding (FASTER)
            a_dict = {tuple(p.xs + p.zs): p for p in measurements}
            nu_meas = [a_dict[key] for key in a_dict.keys()]
            strats = [Strategy(m, t) for m in nu_meas]
            # print(f'{len(strats)=}')
            strats.sort(key=lambda x: x.pauli.weight(), reverse=False)

            return strats


if __name__ == '__main__':
    pass


def gen_strats_from_stabs(stabs, n_qubits, targets=None, get_individuals=False, new_implementation=True):
    """
    For a networkx graph, use the adjacency list to find the generators of the stabilizer group
    Assuming the input qubit is qubit 0,
    """
    # Get the set of unique measurement operators that can teleport
    strats = []
    if targets is None:
        targets = [i for i in range(1, n_qubits)]
    for out in targets:
        meas = get_measurements(stabs, out, get_individuals=get_individuals, new_implementation=new_implementation)
        strats += meas
    return strats


def gen_strats_from_stabs_mp(stabs, n_qubits, targets=None):
    """
    TODO multiprocessing so each target qubit is calculated in parallel
    For a networkx graph, use the adjacency list to find the generators of the stabilizer group
    Assuming the input qubit is qubit 0,
    """

    # Get the set of unique measurement operators that can teleport
    strats = []
    if targets is None:
        targets = [i for i in range(1, n_qubits)]
    def strts(t):
        strats = []
        pairs, meas = get_measurements(stabs, t)
        return meas
    return strats


def gen_strats_from_graph(graph, no_y=False, targets=None):
    """
    For a networkx graph, use the adjacency list to find the generators of the stabilizer group
    Assuming the input qubit is qubit 0,
    """
    num_qubits = graph.number_of_nodes()
    # For a given Input graph, find it's stabilizer group, and the set of trivial and non-trivial stabilizers
    stab_gen = stabilizers_from_graph(graph)
    stab_grp_t, stab_grp_nt = gen_stabs_new(stab_gen)
    # Get the set of unique measurement operators that can teleport
    strats = []
    if targets is None:
        targets = [i for i in range(1, num_qubits)]
    for out in targets:
        pairs, meas = get_measurements(stab_grp_nt, out)
        strats += meas
    return strats


def gen_logicals_from_stab_grp(stabs, n_qubits):
    x, y, z = (Pauli([], [0], n_qubits, 0), Pauli([0], [0], n_qubits, 0), Pauli([0], [], n_qubits, 0))
    all_x_log, all_y_log, all_z_log = [[m.mult(s) for s in stabs] for m in (x, y, z)]
    x_log, y_log, z_log = [[l for l in m_list if 0 not in l.support] for m_list in (all_x_log, all_y_log, all_z_log)]
    x_or_y_log = x_log + y_log
    return x_log, y_log, z_log, x_or_y_log


class Stabilizer:
    def __init__(self, generators):
        self.generators = generators
        self.n = generators[0].n

    def x_meas(self, qubit, outcome=1):
        if outcome == 1:
            i = 0
        elif outcome == -1:
            i = 2
        else:
            raise ValueError
        p = Pauli(z_ix=[], x_ix=[qubit], n=self.n, i_pow=i)
        nu_gen_list = []
        first_anticommuting_gen = None
        for gen in self.generators:
            if p.commutes_global(gen):
                nu_gen_list.append(gen)
            else:
                if first_anticommuting_gen is None:
                    nu_gen_list.append(p)
                    first_anticommuting_gen = gen
                else:
                    nu_gen_list.append(first_anticommuting_gen.mult(gen))
        self.generators = nu_gen_list[:]


def main():
    pass


if __name__ == '__main__':
    main()
