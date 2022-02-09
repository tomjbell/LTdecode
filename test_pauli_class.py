from unittest import TestCase
from pauli_class import Pauli
from random import randrange, randint


class TestPauli(TestCase):
    def test_copy(self):
        n = 5
        # Generate some Paulis to test
        a_pauli_list = [Pauli(z_ix=[], x_ix=[], n=n)]
        for _ in range(10):
            x, z = randint(0, 2**(n-1)), randint(0, 2**(n-1))
            print(x, z)
            xs, zs = [1 & (x >> j) for j in range(n)], [1 & (z >> k ) for k in range(n)]
            print(xs, zs)
            x_ixs, z_ixs = [j for j in range(n) if xs[j]], [k for k in range(n) if zs[k]]
            a_pauli_list.append(Pauli(x_ix=x_ixs, z_ix=z_ixs, n=n))
        print([p.to_str() for p in a_pauli_list])


        self.fail()
