from pauli_class import Pauli


def emit(zl, xl, s, ix, interstitial=0):
    """
    Update the logical operators and stabilizer group associated with a graph state by emission of a single photon,
    in a machine gun protocol style approach
    """
    # current number of qubits
    n = zl[0].n
    # Add new qubit
    for pauli_list in (zl, xl, s):
        for p in pauli_list:
            p.add_qubit()
    # Add the trivial stabilizer that stabilizes the new qubit (n+1)th
    s.append(Pauli([n], [], n+1, 0))
    # Transform the stabilizers and logical operators by conjugation with the operator H_emitter CX(emitter, new_photon)
    for pauli_list in (zl, xl, s):
        for p in pauli_list:
            if interstitial == 0:
                p.conjugate_w_cxh(control=ix, target=n)
            elif interstitial == 1:
                p.conjugation_by_CX(control=ix, target=n)
            elif interstitial == 2:
                p.conjugation_by_S_CX(control=ix, target=n)
    return zl, xl, s




def get_graphs(n_rounds, n_emitters=1):
    """
    Looking for graph states by applying different intestitial operators
    We generate the full stabilizer group, then identify the terms which look like stabilizer generators, and use these
    to construct the corresponding graph.
    """
    op_dict = {0: 'H', 1: 'I', 2: 'S'}
    for pattern in product([0, 1, 2], repeat=n_rounds):
        print(f'INTERSTITIALS: {[op_dict[p] for p in pattern]}')
        z_logicals = [Pauli([i], [], n_emitters, 0) for i in range(n_emitters)]
        x_logicals = [Pauli([], [i], n_emitters, 0) for i in range(n_emitters)]

        stabilizers = [Pauli([], [i], n_emitters, 0) for i in range(n_emitters)]
        for i in range(n_rounds):
            for emitter in range(n_emitters):
                z_logicals, x_logicals, stabilizers = emit(z_logicals, x_logicals, stabilizers, emitter, interstitial=pattern[i])
        print('Generators:')
        for s in stabilizers:
            print(s.to_str())
        print('\n')
        full_stab_grp = gen_stabs_from_generators(stabilizers)
        for s in full_stab_grp:
            # print(s.to_str())
            if num_1s(s.xs) == 1:
                print('This looks like a generator:')
                print(s.to_str())
                # print('\n')
        gens = [s for s in full_stab_grp if num_1s(s.xs) == 1]
        print([k.to_str() for k in gens])
        neighbours = [0] * len(gens)
        print(neighbours)
        for gen in gens:
            q = gen.xs.index(1)
            print(q)
            neighbs = [y for y in range(gen.n) if gen.zs[y] == 1]
            neighbours[q] = (q, neighbs)
        print(neighbours)
        g = get_graph_from_neighbours(neighbours)
        draw_graph(g, spin_nodes=[x for x in range(n_emitters)])
