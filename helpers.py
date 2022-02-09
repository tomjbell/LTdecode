import pickle
from qutip import Qobj, ptrace
import numpy as np


def partial_trace(mat, qubits_to_keep, n):
    q = Qobj(mat, dims=[[2]*n, [2]*n])
    q2 = q.ptrace(qubits_to_keep)
    return np.array(q2)


def save_obj(obj, name, path):
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path):
    with open(path + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


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
    print(bin_to_num([0, 1, 1, 0, 0, 1]), bin_to_num([0, 1, 1]), bin_to_num([0, 0, 1]))