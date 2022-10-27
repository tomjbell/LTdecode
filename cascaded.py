from helpers import expr_to_prob


class FastResult:
    def __init__(self, spc_dict):
        self.spc_dict = spc_dict

    def get_spc_prob(self, eta):
        if len(list(self.spc_dict.keys())[0]) == 8:
            return sum([self.spc_dict[k] * eta ** sum([k[i] for i in (0, 2, 4, 6)]) *
                    (1 - eta) ** sum([k[i] for i in (1, 3, 5, 7)]) for k in self.spc_dict.keys()])
        elif len(list(self.spc_dict.keys())[0]) == 12:
            return sum([self.spc_dict[k] * eta ** sum([k[i] for i in (0, 3, 6, 9)]) *
                        (1 - eta) ** sum([k[i] for i in (2, 5, 8, 11)]) * 0 ** sum([k[i] for i in (1, 4, 7, 10)]) for k in self.spc_dict.keys()])
        else:
            return sum([self.spc_dict[k] * eta ** sum(k[0::2]) * (1 - eta) ** sum(k[1::2]) for k in self.spc_dict.keys()])


class AnalyticCascadedResult:
    def __init__(self, concat_of_perf_dicts, cascade_ix=None):
        """
        :param concat_of_perf_dicts: A list of the spc, x, y, z express dictionaries describing their polynomial success function
        :param cascade_ix: The order in which the graphs corresponding to the express dicts are cascaded
        This allows inhomogeneous cascades
        """
        if cascade_ix is None:
            self.layer_map = lambda x: x
            self.max_depth = len(concat_of_perf_dicts)
        else:
            self.layer_map = lambda x: cascade_ix[x]
            self.max_depth = len(cascade_ix)
        try:
            self.spc_dicts = [x[0] for x in concat_of_perf_dicts]
        except IndexError:
            print(concat_of_perf_dicts)
            print([len(x) for x in concat_of_perf_dicts])
        self.xeffs = [x[1] for x in concat_of_perf_dicts]
        self.yeffs= [x[2] for x in concat_of_perf_dicts]
        self.zeffs = [x[3] for x in concat_of_perf_dicts]
        self.xyeffs = [x[4] for x in concat_of_perf_dicts]

    def get_func(self, expr):
        def succ_prob(xx, xzi, xf, zz, zzi, zf, yy, yzi, yf, aa, azi, af):
            return sum([expr[k] * xx ** k[0] * xzi ** k[1] * xf ** k[2] * zz ** k[3] * zzi ** k[4] * zf ** k[5]
                        * yy ** k[6] * yzi ** k[7] * yf ** k[8] * aa ** k[9] * azi ** k[10] * af ** k[11] for k in
                        expr.keys()])
        return succ_prob

    def xx(self, n, t):
        """find the probability of an attempted x measurement of a qubit with n graphs below it returning an x
        measurement"""
        if n == 0:
            return t
        else:
            xx_ = self.xx(n-1, t)
            yy_ = self.yy(n-1, t)
            zz_ = self.zz(n-1, t)
            zi_ = self.zi(n-1, t)
            func = self.get_func(self.xyeffs[self.layer_map(self.max_depth - n)])
            return t * func(xx_, zi_, 1 - xx_ - zi_, zz_, zi_, 1 - zz_ - zi_, yy_, zi_, 1 - yy_ - zi_, 0, 0, 0)

    def yy(self, n, t):
        if n == 0:
            return t
        else:
            xx_ = self.xx(n - 1, t)
            yy_ = self.yy(n - 1, t)
            zz_ = self.zz(n - 1, t)
            zi_ = self.zi(n - 1, t)
            func = self.get_func(self.xyeffs[self.layer_map(self.max_depth - n)])
            return t * func(xx_, zi_, 1 - xx_ - zi_, zz_, zi_, 1 - zz_ - zi_, yy_, zi_, 1 - yy_ - zi_, 0, 0, 0)

    def zz(self, n, t):
        return t

    def zi(self, n, t):
        if n == 0:
            return 0
        else:
            xx_ = self.xx(n - 1, t)
            yy_ = self.yy(n - 1, t)
            zz_ = self.zz(n - 1, t)
            zi_ = self.zi(n - 1, t)
            func = self.get_func(self.zeffs[self.layer_map(self.max_depth - n)])
            return (1 - t) * func(xx_, zi_, 1 - xx_ - zi_, zz_, zi_, 1 - zz_ - zi_, yy_, zi_, 1 - yy_ - zi_, 0, 0, 0)

    def get_spc_prob(self, t, depth=1):
        self.max_depth = depth
        xx = self.xx(depth-1, t)
        yy = self.yy(depth-1, t)
        zz = t
        zi = self.zi(depth-1, t)
        xf = 1 - xx - zi
        yf = 1 - yy - zi
        zf = 1 - zz - zi
        spc_top_func = self.get_func(self.spc_dicts[self.layer_map(0)])
        return spc_top_func(xx, zi, xf, zz, zi, zf, yy, zi, yf, xx, zi, xf)


class CascadedResultPauli:
    def __init__(self, concat_result_list, cascade_ix=None, ec=True, from_dict=False):
        """
        :param concat_result_list: concatenation of results lists, each one is a dictionary with keys spc, x, y, z, xy, direct_z
        and each value is a list of results
        :param epsilon_0:
        :param eta_0:
        """
        self.from_dict = from_dict
        if cascade_ix is None:
            self.layer_map = lambda x: x
            self.max_depth = len(concat_result_list)
        else:
            self.layer_map = lambda x: cascade_ix[x]
            self.max_depth = len(cascade_ix)
        # print(f'{[self.layer_map(_) for _ in range(5)]=}')
        self.results_lists = concat_result_list
        self.pxx = [None] * self.max_depth
        self.pxz = [0] * self.max_depth
        self.pyy = [0] * self.max_depth
        self.pyz = [0] * self.max_depth
        self.pzz = [0] * self.max_depth
        self.pz_z = [0] * self.max_depth
        self.pzz_ = [0] * self.max_depth
        self.pz_z_ = [0] * self.max_depth
        self.ex = [0] * self.max_depth
        self.ey = [0] * self.max_depth
        self.ez = [0] * self.max_depth
        self.ezi = [0] * self.max_depth
        self.epsilon = 0
        self.eta = 1
        self.ec = ec

    def get_var(self, depth):
        # print(depth)
        # print(self.pxx[depth], self.pxz[depth], self.pzz[depth], self.pz_z[depth], self.pzz_[depth], self.pz_z_[depth])
        return self.pxx[depth], self.pxz[depth], self.pyy[depth], self.pyz[depth], self.pzz[depth], self.pz_z[depth], \
               self.pzz_[depth], self.pz_z_[depth], self.ex[depth], self.ey[depth], self.ez[depth], self.ezi[depth]

    def f_disentangle(self, pxx, pxz, pzz, k):
        if self.from_dict:
            dx = self.results_lists[self.layer_map(k)]['x']
            dy = self.results_lists[self.layer_map(k)]['y']
            return max([p_from_dict(dict_b, pxx, pxz, pzz) for dict_b in (dx, dy)])
        else:
            return max([sum([r.outcome_prob(pxx, pxz, pzz) for r in self.results_lists[self.layer_map(k)]['x']]),
                        sum([r.outcome_prob(pxx, pxz, pzz) for r in self.results_lists[self.layer_map(k)]['y']])])

    def f_z_indirect(self, pxx, pxz, pzz, k):
        if self.from_dict:
            dz = self.results_lists[self.layer_map(k)]['z']
            return p_from_dict(dz, pxx, pxz, pzz)
        else:
            return sum([r.outcome_prob(pxx, pxz, pzz) for r in self.results_lists[self.layer_map(k)]['z']])

    def f_spc(self, pxx, pxz, pzz, k):
        if self.from_dict:
            ds = self.results_lists[self.layer_map(k)]['spc']
            return p_from_dict(ds, pxx, pxz, pzz)
        else:
            return sum([r.outcome_prob(pxx, pxz, pzz) for r in self.results_lists[self.layer_map(k)]['spc']])

    def get_all_params(self, eta, depolarising_noise, ec=True):
        self.ec = ec
        self.eta = eta
        self.epsilon = depolarising_noise/2
        for k in reversed(range(self.max_depth)):
            # print(f'{k=}')
            params_k = self.calculate_params(k)
            self.pxx[k], self.pxz[k], self.pyy[k], self.pyz[k], self.pzz[k], self.pz_z[k],\
            self.pzz_[k], self.pz_z_[k], self.ex[k], self.ey[k], self.ez[k], self.ezi[k] = params_k

    def calculate_params(self, depth):
        if depth == self.max_depth - 1:
            return self.eta, 0, self.eta, 0, 0, 0, self.eta, 1-self.eta, self.epsilon, self.epsilon, self.epsilon, self.epsilon
        else:
            pxx, pxz, pyy, pyz, pzz, pz_z, pzz_, pz_z_, ex, ey, ez, ezi = self.get_var(depth+1)
            # print(f'\n{pxz==pz_z=}\n')

            indirect_z_prob = self.f_z_indirect(pxx, pxz, pzz + pzz_, depth + 1)
            disentangle_prob = self.f_disentangle(pxx, pxz, pzz + pzz_, depth + 1)
            # print(f'{sum([pzz, pzz_])=}')
            # print(f'{depth=}, {indirect_z_prob=}')
            yy = xx = self.eta * disentangle_prob
            yzi = xzi = (1 - self.eta) * indirect_z_prob
            zz = self.eta * indirect_z_prob
            z_z = (1 - self.eta) * indirect_z_prob
            zz_ = self.eta * (1 - indirect_z_prob)
            z_z_ = 1 - zz_ - z_z - zz

            if self.ec:
                # TODO write this to correctly calculate cascaded errors
                idx = 0
                # print('##############################')
                # print(xx)
                # print(f'{depth=}')
                for result in self.results_lists[self.layer_map(depth+1)]['xy']:
                    no_flip = result.no_flip_prob_cascade([1-ex, 1-ey, 1-ez, 1-ezi])
                    accuracy_both = no_flip * (1 - self.epsilon)
                    print(accuracy_both)
                    idx += accuracy_both * result.outcome_prob(pxx, pxz, pzz + pzz_) * self.eta / xx
                idy = idx
                for r in self.results_lists[self.layer_map(depth+1)]['z']:
                    no_flip = r.no_flip_prob_cascade([1 - ex, 1 - ey, 1 - ez, 1 - ezi])
                    prob = r.outcome_prob(pxx, pxz, pzz + pzz_)
                    # print(prob, no_flip, prob * no_flip * (1-self.eta)/z_z)
                idzi = sum([result.no_flip_prob_cascade([1-ex, 1-ey, 1-ez, 1-ezi]) * result.outcome_prob(pxx, pxz, pzz + pzz_) * (1-self.eta) / z_z for result in self.results_lists[self.layer_map(depth+1)]['z']])
                idz_direct = sum([result.no_flip_prob_cascade([1-ex, 1-ey, 1-ez, 1-ezi]) * result.outcome_prob(pxx, pxz, pzz + pzz_) / self.eta for result in self.results_lists[self.layer_map(depth+1)]['z_direct']])

                idz = idz_direct
                print(f'{depth=}')
                print(self.eta, self.epsilon)
                print(f'{idx=}, {idzi=}, {idz=}')
                print('\n')
            else:
                idx = idy = idz = idzi = 1
            return xx, xzi, yy, yzi, zz, z_z, zz_, z_z_, 1-idx, 1-idy, 1-idz, 1-idzi

    def get_spc_prob(self, maxdepth):
        """
        :param eta:
        :param epsilon:
        :param maxdepth: the total number of graphs in the cascade
        :return:
        """
        k = self.max_depth-maxdepth
        # print(self.pxx)
        if self.pxx[k] is None:
            raise ValueError('call get_all_params() first')
        # print('parameters to use = ' + str(k))
        # print(self.pxx, self.pxz, self.pzz, self.pz_z, self.pzz_, self.pz_z_)
        # print([self.pzz[i] + self.pzz_[i] for i in range(self.max_depth)])
        # print([self.f_z_indirect(self.pxx[n], self.pxz[n], self.pzz[n] + self.pzz_[n], k=n) for n in range(self.max_depth)])
        #TODO Figure out how to return the accuracy of the pathfinding
        # Include the error in the disentangling measurement of the target

        spc_prob = self.f_spc(self.pxx[k], self.pxz[k], self.pzz[k] + self.pzz_[k], k)
        if self.ec:
            total_acc = 0
            for result in self.results_lists[self.layer_map(k)]['spc']:
                disentangle_t_acc = 1 - self.ex[k]
                spc_pauli_acc = result.no_flip_prob_cascade([1 - self.ex[k], 1 - self.ey[k], 1 - self.ez[k], 1 - self.ezi[k]])
                total_acc += (disentangle_t_acc * spc_pauli_acc + (1 - disentangle_t_acc) * (1 - spc_pauli_acc)) * result.outcome_prob(self.pxx[k], self.pxz[k], self.pzz[k] + self.pzz_[k])
            # acc = sum([result.no_flip_prob_cascade([1 - self.ex[k], 1 - self.ey[k], 1 - self.ez[k], 1 - self.ezi[k]])
            # * result.outcome_prob(self.pxx[k], self.pxz[k], self.pzz[k] + self.pzz_[k]) for result in self.results_lists[self.layer_map(k)]['spc']]) / spc_prob
            return spc_prob, total_acc/spc_prob
        else:
            return spc_prob


class ConcatenatedResult:
    def __init__(self, concat_result_list, cascade_ix=None, ec=False):
        """

        :param concat_result_list: A list of outcome dictionaries, where each dictionary has the outcomes of
        decoding in all bases for the graph at that depth, from which we can calculate effective error rates etc
        :param cascade_ix: The order of the graphs in the cascade, so that we can search different cascades without
        passing around additional performance dictionaries
        :param ec:
        """
        if cascade_ix is None:
            self.layer_map = lambda x: x
            self.max_depth = len(concat_result_list)
        else:
            self.layer_map = lambda x: cascade_ix[x]
            self.max_depth = len(cascade_ix)
        # print(f'{[self.layer_map(_) for _ in range(5)]=}')
        self.ec = ec
        self.results_lists = concat_result_list
        self.px = [None] * self.max_depth
        self.pz = [0] * self.max_depth
        self.py = [0] * self.max_depth
        self.pa = [0] * self.max_depth
        self.ex = [0] * self.max_depth
        self.ey = [0] * self.max_depth
        self.ez = [0] * self.max_depth
        self.ea = [0] * self.max_depth

        self.eta = 1
        self.error = None

    def get_var(self, depth):
        if self.ec:
            return self.px[depth], self.py[depth], self.pz[depth], self.pa[depth], self.ex[depth], self.ey[depth], \
                   self.ez[depth], self.ea[depth]
        else:
            return self.px[depth], self.py[depth], self.pz[depth], self.pa[depth]

    def f_m_indirect(self, basis, pxx, pyy, pzz, paa, k, ex=None, ey=None, ez=None, ea=None):
        if basis != 'spc':
            ea = 0
            paa = 0
        if self.ec:
            res_perfs = []  # This will be a list of tuples of the success probability and accuracy of each element in the results list
            for r in self.results_lists[self.layer_map(k)][basis]:
                prob = r.outcome_prob(pxx, 0, pzz, diff_y=True, pyy=pyy)
                acc = r.no_flip_prob_concat((1-ex, 1-ey, 1-ez, 1-ea))
                res_perfs.append((prob, acc))
            prob_tot = sum([x[0] for x in res_perfs])
            return prob_tot, sum([x[0] * x[1] for x in res_perfs]) / prob_tot
        else:
            return sum([r.outcome_prob(pxx, 0, pzz, paa=paa, diff_y=True, pyy=pyy) for r in
                        self.results_lists[self.layer_map(k)][basis]])

    def calc_params(self, eta, error_prob=0):
        """
        Find all parameters of the concatenation, i.e. the logical error and loss rates at all levels
        :param eta: Physical transmission rate
        :param error_prob: The probability of each of the pauli errors (X, Y, Z) at the physical level
                           In the paper this is called lambda
        :return:
        """
        self.eta = eta
        self.error = error_prob
        for k in reversed(range(self.max_depth)):
            # print(f'{k=}')
            params_k = self.calculate_params(k)
            if self.ec:
                self.px[k], self.py[k], self.pz[k], self.pa[k], self.ex[k], self.ey[k], self.ez[k], self.ea[k] = params_k
            else:
                self.px[k], self.py[k], self.pz[k], self.pa[k] = params_k

    def calculate_params(self, depth):
        if depth == self.max_depth - 1:
            if self.ec:
                return self.eta, self.eta, self.eta, self.eta, 2 * self.error, 2 * self.error, 2 * self.error, 2 * self.error
            else:
                return self.eta, self.eta, self.eta, self.eta
        else:
            if self.ec:
                px, py, pz, pa, ex, ey, ez, ea = self.get_var(depth+1)
                x, erx = self.f_m_indirect('x', px, py, pz, pa, depth + 1, ex, ey, ez, ea)
                y, ery = self.f_m_indirect('y', px, py, pz, pa, depth + 1, ex, ey, ez, ea)
                z, erz = self.f_m_indirect('z', px, py, pz, pa, depth + 1, ex, ey, ez, ea)
                a, era = self.f_m_indirect('spc', px, py, pz, pa, depth + 1, ex, ey, ez, ea)

                return x, y, z, a, erx, ery, erz, era
            else:
                px, py, pz, pa = self.get_var(depth+1)
                x = self.f_m_indirect('x', px, py, pz, pa, depth + 1)
                y = self.f_m_indirect('y', px, py, pz, pa, depth + 1)
                z = self.f_m_indirect('z', px, py, pz, pa, depth + 1)
                a = self.f_m_indirect('spc', px, py, pz, pa, depth + 1)
                return x, y, z, a

    def meff_prob(self, basis, eta, error_p=0, depth=0, calc_params=True):
        """
        Note that here the depth refers to the depth how many layers there are above the qubit, so max depth
        is actually a single layer graph, depth 0 is the deepest concatenation
        :param eta:
        :param depth:
        :return:
        """
        if calc_params:
            self.calc_params(eta, error_prob=error_p)
        if self.ec:
            return self.f_m_indirect(basis, self.px[depth], self.py[depth], self.pz[depth], self.pa[depth], depth,
                                     self.ex[depth], self.ey[depth], self.ez[depth], self.ea[depth])
        else:
            return self.f_m_indirect(basis, self.px[depth], self.py[depth], self.pz[depth], self.pa[depth], depth)


class ConcatenatedResultDicts:
    """
    For use with the performance data that has already been generated and saved as dictionaries
    """
    def __init__(self, concat_result_dicts, cascade_ix=None, ec=False):
        """ Want results dictionaries in the order spc, x, y, z, xy"""
        if cascade_ix is None:
            self.layer_map = lambda x: x
            self.max_depth = len(concat_result_dicts)
        else:
            self.layer_map = lambda x: cascade_ix[x]
            self.max_depth = len(cascade_ix)
        # print(f'{[self.layer_map(_) for _ in range(5)]=}')
        self.results_dicts = concat_result_dicts
        self.px = [None] * self.max_depth
        self.pz = [0] * self.max_depth
        self.py = [0] * self.max_depth
        self.pa = [0] * self.max_depth
        self.eta = 1

    def get_var(self, depth):
        return self.px[depth], self.py[depth], self.pz[depth], self.pa[depth]

    def get_func(self, expr):
        if len(list(expr.keys())[0]) == 12:
            def succ_prob(xx, xzi, xf, zz, zzi, zf, yy, yzi, yf, aa, azi, af):
                return sum([expr[k] * xx ** k[0] * xzi ** k[1] * xf ** k[2] * zz ** k[3] * zzi ** k[4] * zf ** k[5]
                            * yy ** k[6] * yzi ** k[7] * yf ** k[8] * aa ** k[9] * azi ** k[10] * af ** k[11] for k in
                            expr.keys()])
        elif len(list(expr.keys())[0]) == 8:
            def succ_prob(xx, xzi, xf, zz, zzi, zf, yy, yzi, yf, aa, azi, af):
                return sum([expr[k] * xx ** k[0] * xf ** k[1] * zz ** k[2] * zf ** k[3]
                            * yy ** k[4] * yf ** k[5] * aa ** k[6] * af ** k[7] for k in
                            expr.keys()])
        else:
            print(len(expr.keys()[0]))
            raise ValueError
        return succ_prob

    def f_disentangle(self, pxx, pzz, k):
        fdis = self.get_func(self.results_dicts[self.layer_map(k)][4])
        return fdis(pxx, 0, 1 - pxx, pzz, 0, 1 - pzz, pxx, 0, 1 - pxx, 0, 0, 0)

    def f_x_indirect(self, pxx, pyy, pzz, k):
        fx = self.get_func(self.results_dicts[self.layer_map(k)][1])
        return fx(pxx, 0, 1 - pxx, pzz, 0, 1 - pzz, pyy, 0, 1-pyy, 0, 0, 0)

    def f_y_indirect(self, pxx, pyy, pzz, k):
        fy = self.get_func(self.results_dicts[self.layer_map(k)][2])
        return fy(pxx, 0, 1 - pxx, pzz, 0, 1 - pzz, pyy, 0, 1-pyy, 0, 0, 0)

    def f_z_indirect(self, pxx, pyy, pzz, k):
        fzi = self.get_func(self.results_dicts[self.layer_map(k)][3])
        return fzi(pxx, 0, 1 - pxx, pzz, 0, 1 - pzz, pyy, 0, 1 - pyy, 0, 0, 0)

    def f_spc(self, pxx, pyy, pzz, paa, k):
        fdis = self.get_func(self.results_dicts[self.layer_map(k)][0])
        return fdis(pxx, 0, 1 - pxx, pzz, 0, 1 - pzz, pyy, 0, 1 - pyy, paa, 0, 1-paa)

    def calc_params(self, eta):
        self.eta = eta
        for k in reversed(range(self.max_depth)):
            # print(f'{k=}')
            params_k = self.calculate_params(k)
            self.px[k], self.py[k], self.pz[k], self.pa[k] = params_k

    def calculate_params(self, depth):
        if depth == self.max_depth - 1:
            return self.eta, self.eta, self.eta, self.eta
        else:
            px, py, pz, pa = self.get_var(depth+1)
            # print(f'\n{pxz==pz_z=}\n')

            z = self.f_z_indirect(px, py, pz, depth + 1)
            x = self.f_x_indirect(px, py, pz, depth + 1)
            y = self.f_y_indirect(px, py, pz, depth + 1)
            a = self.f_spc(px, py,  pz, pa, depth+1)
            return x, y, z, a

    def meff_prob(self, eta, depth=0, basis='x', calc_params=True):
        if calc_params:
            self.calc_params(eta)
        if basis == 'x':
            return self.f_x_indirect(self.px[depth], self.py[depth], self.pz[depth], depth)
        elif basis == 'y':
            return self.f_y_indirect(self.px[depth], self.py[depth], self.pz[depth], depth)
        elif basis == 'z':
            return self.f_z_indirect(self.px[depth], self.py[depth], self.pz[depth], depth)
        elif basis == 'spc':
            return self.f_spc(self.px[depth], self.py[depth], self.pz[depth], self.pa[depth], depth)
        else:
            raise ValueError


def p_from_dict(res_dict, pxx, pxz, pzz):
    return sum([res_dict[key] * pxx ** (key[0] + key[6] + key[9]) * pxz ** (key[1] + key[7] + key[10]) * (1 - pxx - pxz) ** (key[2] + key[8] + key[11])
                    * pzz ** key[3] * pxz ** key[4] * (1 - pzz - pxz) ** key[5] for key in res_dict.keys()])

