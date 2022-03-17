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
        self.spc_dicts = [x[0] for x in concat_of_perf_dicts]
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
        """find the probability of an attempted x measurement of a qubit with n rings below it returning an x
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

    def get_spc_prob(self, t, depth):
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
    def __init__(self, concat_result_list, cascade_ix=None):
        """
        :param concat_result_list: concatenation of results lists, each one is a dictionary with keys spc, x, y, z, xy,
        and each value is a list of results
        :param epsilon_0:
        :param eta_0:
        """
        if cascade_ix is None:
            self.layer_map = lambda x: x
            self.max_depth = len(concat_result_list)
        else:
            self.layer_map = lambda x: cascade_ix[x]
            self.max_depth = len(cascade_ix)
        # print(f'{[self.layer_map(_) for _ in range(5)]=}')
        self.results_lists = concat_result_list
        self.pxx = [0] * self.max_depth
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

    def get_var(self, depth):
        # print(depth)
        # print(self.pxx[depth], self.pxz[depth], self.pzz[depth], self.pz_z[depth], self.pzz_[depth], self.pz_z_[depth])
        return self.pxx[depth], self.pxz[depth], self.pyy[depth], self.pyz[depth], self.pzz[depth], self.pz_z[depth], \
               self.pzz_[depth], self.pz_z_[depth], self.ex[depth], self.ey[depth], self.ez[depth], self.ezi[depth]

    def prob_outcome(self, pxx, pxz, pzz, counts):
        aa = xx = yy = pxx
        zz = pzz
        azi = xzi = yzi = zzi = pxz
        af = xf = yf = 1 - xx - xzi
        zf = 1 - zz - zzi
        return xx ** counts[0] * xzi ** counts[1] * xf ** counts[2] * zz ** counts[3] * zzi ** counts[4] \
                    * zf ** counts[5] * yy ** counts[6] * yzi ** counts[7] * yf ** counts[8] * aa ** counts[9] \
                    * azi ** counts[10] * af ** counts[11]

    def f_disentangle(self, pxx, pxz, pzz, k):
        out_prob = 0
        for result in self.results_lists[self.layer_map(k)]['xy']:
            counts = [val for val in result.counter.values()]
            out_prob += self.prob_outcome(pxx, pxz, pzz, counts)
        return out_prob

    def f_z_indirect(self, pxx, pxz, pzz, k):
        out_prob = 0
        for result in self.results_lists[self.layer_map(k)]['z']:
            counts = [val for val in result.counter.values()]
            out_prob += self.prob_outcome(pxx, pxz, pzz, counts)
        return out_prob

    def f_spc(self, pxx, pxz, pzz, k):
        out_prob = 0
        for result in self.results_lists[self.layer_map(k)]['spc']:
            counts = [val for val in result.counter.values()]
            out_prob += self.prob_outcome(pxx, pxz, pzz, counts)
        return out_prob

    def get_all_params(self):
        for k in reversed(range(self.max_depth)):
            # print(f'{k=}')
            params_k = self.calculate_params(k)
            self.pxx[k], self.pxz[k], self.pyy[k], self.pyz[k], self.pzz[k], self.pz_z[k],\
            self.pzz_[k], self.pz_z_[k], self.ex[k], self.ey[k], self.ez[k], self.ezi[k] = params_k

    def calculate_params(self, depth):
        if depth == self.max_depth - 1:
            return self.eta, 0, self.eta, 0, 0, 0, self.eta, 1-self.eta, self.epsilon, self.epsilon, self.epsilon, 0
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

            # TODO write this to correctly calculate cascaded errors
            idx = 1 - ex
            idy = 1 - ey
            idz = 1 - ez
            idzi = 1 - ezi
            return xx, xzi, yy, yzi, zz, z_z, zz_, z_z_, ex, ey, ez, ezi

    def get_spc_prob(self, eta, epsilon, maxdepth):
        """

        :param eta:
        :param epsilon:
        :param maxdepth: the total number of graphs in the cascade
        :return:
        """
        self.eta = eta
        self.epsilon = epsilon
        self.get_all_params()
        k = self.max_depth-maxdepth
        # print('parameters to use = ' + str(k))
        # print(self.pxx, self.pxz, self.pzz, self.pz_z, self.pzz_, self.pz_z_)
        # print([self.pzz[i] + self.pzz_[i] for i in range(self.max_depth)])
        # print([self.f_z_indirect(self.pxx[n], self.pxz[n], self.pzz[n] + self.pzz_[n], k=n) for n in range(self.max_depth)])
        return self.f_spc(self.pxx[k], self.pxz[k], self.pzz[k] + self.pzz_[k], k)

