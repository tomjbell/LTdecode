import numpy as np


def add(b1, b2):
    # Bit-wise addition
    return (b1 + b2) % 2


class Pauli:
    def __init__(self, z_ix, x_ix, n, i_pow=0):
        """
        binary representation of a pauli operator on n qubits
        assume a bunch of z operators to the left, if not toggle the sign using the exponent of i
        """
        self.zs = [0] * n
        for ix in z_ix:
            self.zs[ix] = 1
        self.z_ix = z_ix
        self.xs = [0] * n
        for ix in x_ix:
            self.xs[ix] = 1
        self.x_ix = x_ix
        self.i_exp = i_pow
        self.n = n
        self.support = [i for i in range(n) if self.zs[i] == 1 or self.xs[i] == 1]
        self.xz_mat = np.array([self.xs, self.zs]).T

    def set_support(self):
        self.support = [i for i in range(self.n) if self.zs[i] == 1 or self.xs[i] == 1]

    def copy(self):
        other = Pauli([i for i in range(self.n) if self.zs[i]==1], [j for j in range(self.n) if self.xs[j]==1], self.n, self.i_exp)
        return other

    def update_zs(self, ix, val):
        # print(self.zs)
        self.zs[ix] = val
        self.z_ix = [i for i in range(self.n) if self.zs[i] == 1]
        self.set_support()
        self.xz_mat = np.array([self.xs, self.zs]).T

    def update_xs(self, ix, val):
        self.xs[ix] = val
        self.x_ix = [i for i in range(self.n) if self.xs[i] == 1]
        self.set_support()
        self.xz_mat = np.array([self.xs, self.zs]).T

    def mult(self, other):
        """
        right multiply by other pauli (self @ other)
        """
        nu_i = self.i_exp + other.i_exp
        assert self.n == other.n
        nu_z = [(self.zs[i] + other.zs[i])%2 for i in range(self.n)]
        nu_x = [(self.xs[i] + other.xs[i])%2 for i in range(self.n)]
        #  Commute the Zs on the right through the Xs to the left
        for i in range(len(other.zs)):
            if other.zs[i] == 1:
                if self.xs[i] == 1:
                    nu_i += 2
        z_indices = [i for i, k in enumerate(nu_z) if k == 1]
        x_indices = [i for i, k in enumerate(nu_x) if k == 1]
        return Pauli(z_indices, x_indices, self.n, nu_i % 4)

    def to_str(self):
        out_str = []
        if self.i_exp == 1:
            out_str.append('i')
        elif self.i_exp == 2:
            out_str.append('-')
        elif self.i_exp == 3:
            out_str.append('-i')
        for i in range(self.n):
            if self.zs[i] == 1:
                out_str.append(f'Z{i}')
        for i in range(self.n):
            if self.xs[i] == 1:
                out_str.append(f'X{i}')
        return '_'.join(out_str)

    def weight(self):
        return sum([(self.zs[j] or self.xs[j]) for j in range(self.n)])

    def add_qubit(self):
        self.zs.append(0)
        self.xs.append(0)
        self.n += 1

    def conjugate_w_cxh(self, control, target):
        # def conjugation_by_H_CX(pauli, control, target):
        # Will need to worry about signs when a z -> x or vice versa
        minus_sign = 0
        out_z_str = self.zs[:]
        out_x_str = self.xs[:]
        if self.zs[control] == 1:
            out_z_str[control] = 0
            out_x_str[control] = add(out_x_str[control], 1)
        if self.zs[target] == 1:
            out_x_str[control] = add(out_x_str[control], 1)
            if control > target:
                minus_sign = 1  # Because you have to commute it past
        if self.xs[control] == 1:
            out_x_str[control] = 0
            out_z_str[control] = add(out_z_str[control], 1)
            out_x_str[target] = add(out_x_str[target], 1)
        if self.xs[target] == 1:
            pass
        self.zs = out_z_str
        self.xs = out_x_str
        self.i_exp = (self.i_exp + 2*minus_sign) % 4

    def conjugation_by_S_CX(self, control, target):
        z_str = self.zs[:]
        x_str = self.xs[:]
        delta_i = 0
        out_z_str = z_str[:]
        out_x_str = x_str[:]
        if self.zs[control] == 1:
            pass
        if self.zs[target] == 1:
            out_z_str[control] = add(out_z_str[control], 1)
        if self.xs[control] == 1:
            out_z_str[control] = add(out_z_str[control], 1)
            out_x_str[target] = add(out_x_str[target], 1)
            delta_i += 3
        if self.xs[target] == 1:
            pass
        self.zs = out_z_str
        self.xs = out_x_str
        self.i_exp = (self.i_exp + delta_i) % 4

    def conjugation_by_CX(self, control, target):
        z_str = self.zs[:]
        x_str = self.xs[:]
        delta_i = 0
        out_z_str = z_str[:]
        out_x_str = x_str[:]
        if self.zs[control] == 1:
            pass
        if self.zs[target] == 1:
            out_z_str[control] = add(out_z_str[control], 1)
        if self.xs[control] == 1:
            out_x_str[target] = add(out_x_str[target], 1)
        if self.xs[target] == 1:
            pass
        self.zs = out_z_str
        self.xs = out_x_str

    def union(self, other):
        """
        Take the union of two commuting pauli operators, i.e. x on a particular qubit if either one of them or both have
        x on, identity if they are both identity, etc.
        """
        assert(self.commutes_each(other, qbts=[i for i in range(self.n)]))
        z_union = list(set(self.z_ix).union(set(other.z_ix)))
        x_union = list(set(self.x_ix).union(set(other.x_ix)))
        i_pow = (self.i_exp + other.i_exp) % 4
        return Pauli(z_ix=z_union, x_ix=x_union, n=self.n, i_pow=i_pow)

    def commutes_each(self, other, qbts):
        """
        Check if paulis commute on each and every of the qbt-th qubits return 0 if yes, 1 if no
        """
        r = 0
        for q in qbts:
            if self.zs[q] == 1 and other.xs[q] == 1:
                r += 1
            if self.xs[q] == 1 and other.zs[q] == 1:
                r += 1
            if r % 2 == 1:
                return False
        return True

    def commutes_every(self, other):
        """Does this Pauli operator commute with other on every qubit of their joint support?
        Use the matrix representation of the paulis"""
        sig_x = np.array([[0, 1], [1, 0]])
        b = [x%2 for x in np.diag(self.xz_mat @ sig_x @ other.xz_mat.T)]
        return sum(b) == 0

    def anticommuting_ix(self, other):
        """Find the qubit indices where the two pauli operators anticommute"""
        anticoms = []
        for q in range(self.n):
            r = 0
            if self.zs[q] == 1 and other.xs[q] == 1:
                r += 1
            if self.xs[q] == 1 and other.zs[q] == 1:
                r += 1
            if r % 2 == 1:
                anticoms.append(q)
        return anticoms

    def commutes_global(self, other):
        """Do the two Pauli operators commute"""
        xz_arr = np.array(self.zs + self.xs)
        xz_oth = np.array(other.zs + other.xs)
        zxT = np.kron(np.array([[0, 1], [1, 0]]), np.eye(self.n)) @ xz_oth
        return bool(((xz_arr @ zxT) + 1) % 2)

    def equivalent(self, other):
        """
        Are two Pauli operators equivalent?
        Up to a phase
        """
        if self.xs == other.xs and self.zs == other.zs:
            return True
        return False

    def contains_other(self, other, exclude=None):
        """
        Does this pauli P1 contain another pauli operator P2, such that P1 can be written P1 = P2 tensor P3, where P2
         and P3 have non-overlapping supports
         Allow for some qubits to be excluded from the comparison, e.g. for logical fusion measurements where we can
         recover all pauli measurements on the fused qubits
        """
        flag = True
        if exclude is not None:
            qubits_to_compare = list(set(other.support) - set(exclude))
        else:
            qubits_to_compare = other.support
        for q in qubits_to_compare:
            if other.zs[q] != self.zs[q] or other.xs[q] != self.xs[q]:
                flag = False
                break
        return flag

    def get_meas_type(self, ix):
        if self.xs[ix]:
            if self.zs[ix]:
                return 'y'
            else:
                return 'x'
        elif self.zs[ix]:
            return 'z'


def pauli_prod(list_of_paulis):
    out = list_of_paulis[0].copy()
    for x in range(1, len(list_of_paulis)):
        out = out.mult(list_of_paulis[x])
    return out


def multi_union(p_list):
    """
    Take the union of this pauli operator with all others in the list others
    """
    out = p_list[0].copy()
    for other in p_list[1:]:
        tmp = out.union(other)
        out = tmp.copy()
    return out


class Strategy:
    def __init__(self, p, t=None, s1=None, s2=None):
        # TODO include an ordering here so that different measurement orders are different strategies - may have different tolerances
        self.pauli = p
        self.t = t
        self.s1 = s1
        self.s2 = s2

    def to_measure(self, q_lost, pauli_done, other_strats, trivial_imp=True):
        """
        Return the order in which to attempt measurements. Want to try to boost success prob by ensuring lots of backups
        If there are no compatible measurements to make (because of loss), this is not a good strategy - something has gone wrong
        If the measurements have all already been made, we are done (although this should have already been caught)
        """
        if len(set(self.pauli.support) & set(q_lost)) != 0:
            raise ValueError(f'{q_lost=}, {self.pauli.to_str()=}')

        if trivial_imp:
            # Trivial implementation, just return lowest number
            outstanding = list(set(self.pauli.support) - set(pauli_done.support))
            return outstanding
        else:
            raise NotImplementedError

    def copy(self):
        if self.s1 is not None:
            s1_ = self.s1.copy()
        else:
            s1_ = None
        if self.s2 is not None:
            s2_ = self.s2.copy()
        else:
            s2_ = None
        return Strategy(self.pauli.copy(), self.t, s1_, s2_)


def main():
    pass


if __name__ == '__main__':
    main()
