import random
import numpy as np
import matplotlib.pyplot as plt
from CodesFunctions.graphs import gen_linear_graph, gen_ring_graph, gen_tree_graph
from cascaded import AnalyticCascadedResult, get_func_from_results, analytic_cascaded, cascaded_analytic_funcs
from time import time
import multiprocessing as mp


def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count


def main():
    np.random.RandomState(100)
    arr = np.random.randint(0, 10, size=[100000, 5])
    data = arr.tolist()
    print(data[:5])



    print('hi')
    pool = mp.Pool(mp.cpu_count())
    t0 = time()
    results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
    t1 = time()
    pool.close()
    # print(results[:10])
    print(t1 - t0)
    results = [howmany_within_range(row, 4, 8) for row in data]
    t2 = time()
    print(t2-t1)


def pauli_error_5ring(px, py, pz, b='x', n_samples=100):
    """Try a Pauli error-detecting b-basis measurement on qubit 0 of the 5-ring"""
    from pauli_class import Pauli
    if b == 'x':
        s = (Pauli(z_ix=(1, 4), x_ix=[0], n=5, i_pow=0), Pauli(z_ix=(2, 3), x_ix=(0, 2, 3), n=5, i_pow=0))
        target = Pauli(z_ix=[], x_ix=[0], n=5, i_pow=0)
    elif b == 'z':
        s = (Pauli(z_ix=(0, 2), x_ix=[1], n=5, i_pow=0), Pauli(z_ix=(0, 3), x_ix=[4], n=5, i_pow=0))
        target = Pauli(z_ix=[0], x_ix=[], n=5, i_pow=0)

    else:
        raise NotImplementedError

    err_logical = 0
    for _ in range(n_samples):
    # Build error:
        xerr = [i for i in range(5) if random.random() < px]
        yerr = [i for i in range(5) if random.random() < py]
        zerr = [i for i in range(5) if random.random() < pz]

        xerrs = list(set(xerr).symmetric_difference(set(yerr)))
        zerrs = list(set(zerr).symmetric_difference(set(yerr)))
        error = Pauli(z_ix=zerrs, x_ix=xerrs, n=5, i_pow=0)
        syndrome = (s[0].commutes_global(error), s[1].commutes_global(error))
        # print(syndrome, s[0].to_str(), s[1].to_str(), error.to_str())
        if syndrome == (False, False):
            error_syndrome = True
        else:
            error_syndrome = False
        # Was there really an error? Does our syndrome agree?
        target_error = not target.commutes_global(error)
        if (not target_error and error_syndrome) or (target_error and not error_syndrome):
            err_logical += 1/n_samples
    return err_logical


if __name__ == '__main__':
    from pauli_class import Pauli
    from graphs import draw_graph
    from error_correction import best_checks

    n=7
    p = 0.05
    g = gen_tree_graph([2, 2, 1])
    from stab_formalism import gen_stabs_from_generators, stabilizers_from_graph
    t0 = time()

    s = stabilizers_from_graph(g)
    nt, t = gen_stabs_from_generators(s, split_triviality=True)
    t1 = time()
    print(t1-t0)
    exit()



    # import multiprocessing
    #
    # p = multiprocessing.Pool()
    # t0 = time()
    # # result = p.apply_async(func).get()
    # result = func()
    # t1 = time()
    # print(result)
    # print(t1-t0)
    #
    #
    # exit()
    from CodesFunctions.graphs import gen_ring_graph
    from decoder_class import CascadeDecoder
    for nq in [3, 4,5 , 6, 7, 8, 9, 10]:
        t0 = time()
        ring = gen_ring_graph(nq)
        x = CascadeDecoder(ring)
        dicts = [x.get_dict(b) for b in ('spc', 'x', 'y', 'z', 'xy')]
        t1 = time()
        print(t1 - t0)

        r = AnalyticCascadedResult(dicts[0], dicts[1], dicts[2], dicts[3], dicts[4])
        etas = np.linspace(0., 1)
        plt.plot(etas, [r.get_spc_prob(eta, depth=1) for eta in etas])
    plt.plot((0, 1), (0, 1), 'k--')
    plt.show()




