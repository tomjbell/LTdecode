import pickle
from os import getcwd
import numpy as np


# def partial_trace(mat, qubits_to_keep, n):
#     q = Qobj(mat, dims=[[2]*n, [2]*n])
#     q2 = q.ptrace(qubits_to_keep)
#     return np.array(q2)
def bisection_search(interval, func, depth=10, y0=None, y1=None):
    k0, k1 = interval[0], interval[1]
    mid = 0.5 * (k0 + k1)
    if depth == 0:
        return (k0 + k1)/2
    else:
        if y0 is None:
            y0 = func(k0)
            y1 = func(k1)
        ymid = func(mid)
        if y0 * ymid < 0:
            return bisection_search([k0, mid], func, depth=depth-1, y0=y0, y1=ymid)
        elif ymid * y1 < 0:
            return bisection_search([mid, k1], func, depth=depth-1, y0=ymid, y1=y1)
        else:
            raise ValueError('No root in interval')


def save_obj(obj, name, path):
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path, suffix='.pkl'):
    with open(path + '/' + name + suffix, 'rb') as f:
        return pickle.load(f)


def expr_to_prob(expr, t):
    tot = 0
    for key, val in expr.items():
        tot += val * t ** key[0] * (1-t)**key[1]
    return tot


def add(b1, b2):
    # Bit-wise addition
    return (b1 + b2)%2


def num_1s(a_str):
    out = 0
    for i in a_str:
        if i == 1:
            out += 1
    return out


def bin_to_str(pauli):
    out_str = []
    for i in range(len(pauli[0])):
        if pauli[0][i] == 1:
            out_str.append(f'Z{i}')
        if pauli[1][i] == 1:
            out_str.append(f'X{i}')
    return '_'.join(out_str)


def bin_to_num(b):
    out = 0
    n = len(b)
    for i in range(n):
        out += b[n-1-i] * 2**i
    return out


if __name__ == '__main__':
    x = load_obj('8QubitResultDicts', getcwd() + '/LC_equiv_graph_data')
    print(x[0])
    y = load_obj('10QubitResultDicts', getcwd() + '/LC_equiv_graph_data')
    print(y[0])