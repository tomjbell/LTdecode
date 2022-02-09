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
