import networkx as nx
import numpy as np
from pauli_class import Pauli, Strategy
from stab_formalism import stabilizers_from_graph, gen_stabs_from_generators, gen_logicals_from_stab_grp, \
    gen_strats_from_stabs, gen_stabs_new
from itertools import combinations, product
from graphs import gen_ring_graph, draw_graph
import matplotlib.pyplot as plt
from helpers import bisection_search
from copy import deepcopy
from random import random


class LFStrategy:
    def __init__(self, pauli, lx=None, lz=None, anticommuting_ix=None):
        self.pauli = pauli
        self.x = lx
        self.z = lz
        self.to_fuse = anticommuting_ix

    def copy(self):
        if self.x is not None:
            x_ = self.x.copy()
        else:
            x_ = None
        if self.z is not None:
            z_ = self.z.copy()
        else:
            z_ = None
        return Strategy(self.pauli.copy(), x_, z_, self.to_fuse.copy())


class FusionDecoder:
    """
    Superclass for the different types of fusion decoder, used for generating success probabilities and thresholds for FBQC
    """
    def __init__(self, graph=None):
        if graph is None:
            pass
        else:
            self.nq = graph.number_of_nodes()
            self.graph = graph
            stab_generators = stabilizers_from_graph(graph)
            self.stab_grp_t, self.stab_grp_nt = gen_stabs_new(stab_generators)

    def best_perasure_with_w(self, eta, pfail, take_min=True):
        ws = np.linspace(0, 1)
        outs = []
        for w in ws:
            outs.append(self.get_probs_from_outcomes(eta=eta, pfail=pfail, w=w))
        avgs = [0.5 * (o['xrec'] + o['zrec']) for o in outs]
        mins = [min([o['xrec'], o['zrec']]) for o in outs]
        if take_min:
            max_ix = np.argmax(mins)
        else:
            max_ix = np.argmax(avgs)
        # print(outs[max_ix], ws[max_ix])
        return outs[max_ix], ws[max_ix]

    def get_probs_from_outcomes(self, eta, pfail, max_w):
        """
        Needs to be overwritten below
        :param eta:
        :param pfail:
        :return:
        """
        raise NotImplementedError
        return None, None

    def get_threshold(self, pfail=0.5, pthresh=0.88, w=None, optimising_eta=0.96, print_max_w=False, take_min=True, no_w=False):
        if not no_w:
            if w is None:
                max_av, max_w = self.best_perasure_with_w(optimising_eta, pfail, take_min=take_min)
            else:
                max_w = w
            if print_max_w:
                print(f'{max_w=}')
        else:
            max_w = None

        def func_to_optimise(t):
            probs = self.get_probs_from_outcomes(t, pfail, max_w)
            if take_min:
                t = min([probs['xrec'], probs['zrec']]) - pthresh
            else:
                t = 0.5 * (probs['xrec'] + probs['zrec']) - pthresh
            return t

        try:
            threshold = bisection_search([0.9, 1], func_to_optimise)
        except ValueError:
            threshold = 1
        # print(1-threshold)
        return threshold

    def plot_threshold(self, n_points=27, pthresh=0.88, optimising_eta=0.96, show=True, w=None, line='--', take_min=True, print_max_w=False, no_w=False):
        pfails = np.linspace(0.001, 0.5, n_points)
        data = [1-self.get_threshold(p, pthresh=pthresh, w=w, optimising_eta=optimising_eta, take_min=take_min, print_max_w=print_max_w, no_w=no_w) for p in pfails]

        # print(data)
        xclean = []
        yclean = []
        for i in range(len(pfails)):
            if data[i] > 0:
                xclean.append(pfails[i])
                yclean.append(data[i])
        plt.plot(xclean, yclean, line, color='k')
        pfdiscr = [0.5 ** i for i in range(2, 7)]
        plt.plot(pfdiscr, [1-self.get_threshold(p, pthresh=pthresh, w=w, optimising_eta=optimising_eta, take_min=take_min, print_max_w=print_max_w, no_w=no_w) for p in pfdiscr], color='k', marker='o', linestyle='')
        if show:
            plt.xlabel('pfail')
            plt.ylabel('loss threshold')
            plt.show()


class TransversalFusionDecoder(FusionDecoder):
    def __init__(self, graph):
        super(TransversalFusionDecoder, self).__init__(graph)
        self.logical_operators = gen_logicals_from_stab_grp(self.stab_grp_nt, self.nq)
        self.fusion_outcomes = None
        self.fb_dicts = None

    def get_xz_logicals(self):
        """
        Get the x and z logical operators in the form of lists of the x meas, ymeas and z meas qubits
        :return:
        """
        log_all = {'x': self.logical_operators[0], 'z': self.logical_operators[2]}
        log_out = {'x': [], 'z': []}
        for m_type in ['x', 'z']:
            for p in log_all[m_type]:
                # Get rid of anything with Y measurement - this is unnecessary!
                # for q in p.support:
                #     if p.get_meas_type(q) == 'y':
                #         good = False
                xs = set(p.x_ix)
                zs = set(p.z_ix)
                good = True
                if good:
                    log_out[m_type].append((xs - zs, xs & zs, zs - xs, p))
        return log_out

    def decode(self, monte_carlo=False):
        """
        Follow the FBQC approach to finding FBQC loss tolerance thresholds
        xx measurements are independent of zz measurements
        randomize the type of fusion so that zz is achieved on fusion failure with probability w, xx with (1-w)
        Explore the full possible set of measurement outcomes, i.e. picking from successful fusion, x recovery, z recovery and erasure
        :param w:
        :param pfail: Physical fusion failure probability
        :param monte_carlo: monte carlo sim?
        :return:
        """
        logical_operators = self.get_xz_logicals()
        if monte_carlo:
            pass
        else:
            fusion_outcomes = []
            # Generate all sets of possible outcomes, [loss, fused, fail recover x, fail_recover y, fail recover z]
            # Probabilities [(1 - eta ** 1/pfail), eta**1/pfail * (1 - pfail), eta**1/pfail * (1-w)pfail, eta ** 1/pfail * wpfail]
            for outcomes in product(['loss', 'fused', 'x', 'y', 'z'], repeat=self.nq-1):
                lost = set([i + 1 for i, x in enumerate(outcomes) if x == "loss"])
                fused = set([i + 1 for i, x in enumerate(outcomes) if x == "fused"])
                xrec = set([i + 1 for i, x in enumerate(outcomes) if x == "x"])
                yrec = set([i + 1 for i, x in enumerate(outcomes) if x == "y"])
                zrec = set([i + 1 for i, x in enumerate(outcomes) if x == "z"])
                xm = fused.union(xrec)
                ym = fused.union(yrec)
                zm = fused.union(zrec)
                probs = (lost, fused, xrec, yrec, zrec)
                ops_measured = {'x': False, 'z': False, 'probs': probs}
                for m_type in ('x', 'z'):
                    for operator in logical_operators[m_type]:
                        if operator[0].issubset(xm) and operator[1].issubset(ym) and operator[2].issubset(zm):
                            ops_measured[m_type] = True
                            break
                # Only append if one or other is measured
                if ops_measured['x'] or ops_measured['z']:
                    fusion_outcomes.append(ops_measured)
            self.fusion_outcomes = fusion_outcomes

    def get_fixed_basis_dicts(self):
        """
        Generate the results for fusion, xx and zz recovered logical measurements - for use in FBQC stuff
        :return:
        """
        dicts = {'fusion': {}, 'xrec': {}, 'zrec': {}}
        for rec_bases in product(['x', 'y', 'z'], repeat=self.nq-1):
            for mt_dict in dicts.values():
                mt_dict[rec_bases] = {}
            xf = set([i + 1 for i, x in enumerate(rec_bases) if x == "x"])
            yf = set([i + 1 for i, x in enumerate(rec_bases) if x == "y"])
            zf = set([i + 1 for i, x in enumerate(rec_bases) if x == "z"])
            m_type_done = None
            for f_res in self.fusion_outcomes:
                if f_res['z']:
                    if f_res['x']:
                        m_type_done = 'fusion'
                    else:
                        m_type_done = 'zrec'
                elif f_res['x']:
                    m_type_done = 'xrec'
                if m_type_done is not None:
                    l, f, x, y, z = f_res['probs']
                    if x.issubset(xf) and y.issubset(yf) and z.issubset(zf):
                        nl, nf, nx, ny, nz = [len(a) for a in [l, f, x, y, z]]
                        key = (nl, nf, nx + ny + nz)
                        if key in dicts[m_type_done][rec_bases]:
                            dicts[m_type_done][rec_bases][key] += 1
                        else:
                            dicts[m_type_done][rec_bases][key] = 1
        self.fb_dicts = dicts

    def best_fixed_basis(self, eta, pfail):
        if self.fb_dicts is None:
            self.get_fixed_basis_dicts()
        na = 1 / pfail
        best_config = None
        max_p = 0

        def prob(expr_dict):
            return sum([v * (1 - eta ** na) ** k[0] * ((1 - pfail) * eta ** na) ** k[1] * (pfail * eta ** na) ** k[2] for k, v in expr_dict.items()])
        for key, val in self.fb_dicts.items():
            p = prob(val)
            if p > max_p:
                max_p = p
                best_config = key
        return best_config, max_p

    def fusion_prob(self, etas, p_fail):
        return [self.best_fixed_basis(eta, p_fail)[1] for eta in etas]

    def get_probs_from_outcomes(self, eta, pfail, w):
        """
        Calculate the probability of obtaining a particular pattern given that you are randomly selecting between
        failing in the X and Z bases with weighting w
        :param eta:
        :param pfail:
        :param w:
        :return:
        """
        def prob(nlost, nfused, nxrec, nyrec, nzrec):
            a = 1/pfail
            return ((1 - eta ** a) ** nlost) * (eta ** a * (1 - pfail)) ** nfused *\
                   (eta ** a * (1 - w) * pfail) ** nxrec * (eta ** a * w * pfail) ** nzrec * 0 ** nyrec
        prob_dict = {'fusion': 0, 'xrec': 0, 'zrec': 0}
        for f_res in self.fusion_outcomes:
            nl, nf, nx, ny, nz = [len(a) for a in f_res['probs']]
            p = prob(nl, nf, nx, ny, nz)
            if f_res['x']:
                prob_dict['xrec'] += p
                if f_res['z']:
                    prob_dict['zrec'] += p
                    prob_dict['fusion'] += p
            elif f_res['z']:
                prob_dict['zrec'] += p
        return prob_dict


class DiffBasisTransversalDecoder(FusionDecoder):
    """
    This strategy we choose the failure basis for the fusion measurements independently for each qubit
    THIS DECODER IS NOT FULLY IMPLEMENTED
    """
    def __init__(self, graph, picker='maxmin'):
        super(DiffBasisTransversalDecoder, self).__init__(graph)
        self.xyz_logical_operators = gen_logicals_from_stab_grp(self.stab_grp_nt, self.nq)
        self.xz_logicals = self.prep_logical_operators()
        self.failure_bases = None
        self.fusion_outcomes = None

    def prep_logical_operators(self):
        """
        Prepare logical operators as sets of indices with x, y, z measurements
        :return: dictionary of logical x and z operators, where each is of the form (x_qubits, y_qubits, z_qubits)
        """
        log_all = {'x': self.xyz_logical_operators[0], 'z': self.xyz_logical_operators[2]}
        log_out = {'x': [], 'z': []}
        for m_type in ['x', 'z']:
            for p in log_all[m_type]:
                x_ix = set(p.x_ix)
                z_ix = set(p.z_ix)
                log_out[m_type].append((x_ix - z_ix, x_ix & z_ix, z_ix - x_ix))
        return log_out

    def choose_failure_bases(self, picker='maxmin', printing=False):
        """
        Create a dictionary that tells us how we fail on each qubit - precompile this dictionary
        The performance of the decoder will be dependent on how well this function chooses optimal failure bases.
        :param picker: 'maxmin', 'maxone' or 'sum'
        :return:
        """
        failure_bases = {'x': set(), 'y': set(), 'z': set()}
        # print([p.to_str() for p in self.xyz_logical_operators[0]])
        # print([p.to_str() for p in self.xyz_logical_operators[2]])

        for code_q in range(1, self.nq):
            n_type_x_logical = {'x': 0, 'y': 0, 'z': 0}
            n_type_z_logical = {'x': 0, 'y': 0, 'z': 0}
            for x_log in self.xyz_logical_operators[0]:
                if code_q in x_log.support:
                    n_type_x_logical[x_log.get_meas_type(code_q)] += 1
            for z_log in self.xyz_logical_operators[2]:
                if code_q in z_log.support:
                    n_type_z_logical[z_log.get_meas_type(code_q)] += 1

            # Decide whether we want to maximise the minimum of xlog and z_log, maximise one, or maximise the sum
            if picker == 'maxmin':
                mins = {k: min([n_type_x_logical[k], n_type_z_logical[k]]) for k in ['x', 'y', 'x']}
                basis = max(mins, key=mins.get)
            elif picker == 'maxone':
                maxs = {k: max([n_type_x_logical[k], n_type_z_logical[k]]) for k in ['x', 'y', 'x']}
                basis = max(maxs, key=maxs.get)
            elif picker == 'sum':
                sums = {k: n_type_x_logical[k] + n_type_z_logical[k] for k in ['x', 'y', 'x']}
                basis = max(sums, key=sums.get)
            else:
                raise ValueError
            if printing:
                print(f'{n_type_x_logical=}')
                print(f'{n_type_z_logical=}')
                print(f'Best basis for qubit {code_q} with strategy {picker} is {basis}')
            failure_bases[basis].add(code_q)
        print(failure_bases)
        return failure_bases

    def decode(self, picker='maxmin'):
        """
        Initialise all sets of possibilities and see if we recover fusion, logical x or logical z
        :return:
        """
        self.failure_bases = self.choose_failure_bases(picker=picker)
        fusion_outcomes = []
        logical_operators = self.xz_logicals
        for outcomes in product(['loss', 'fused', 'fail'], repeat=self.nq - 1):
            lost = [i + 1 for i, x in enumerate(outcomes) if x == "loss"]
            fused = [i + 1 for i, x in enumerate(outcomes) if x == "fused"]
            fail = [i + 1 for i, x in enumerate(outcomes) if x == "fail"]
            xm = set(fused).union(set(fail) & self.failure_bases['x'])
            ym = set(fused).union(set(fail) & self.failure_bases['y'])
            zm = set(fused).union(set(fail) & self.failure_bases['z'])
            # print(outcomes, xm, ym, zm)

            probs = (len(lost), len(fused), len(fail))
            ops_measured = {'x': False, 'z': False, 'probs': probs}
            for m_type in ('x', 'z'):
                for operator in logical_operators[m_type]:
                    if operator[0].issubset(xm) and operator[1].issubset(ym) and operator[2].issubset(zm):
                        ops_measured[m_type] = True
                        break
            # Only append if one or other is measured
            if ops_measured['x'] or ops_measured['z']:
                fusion_outcomes.append(ops_measured)
        self.fusion_outcomes = fusion_outcomes
        print(fusion_outcomes)

    def get_probs_from_outcomes(self, eta, pfail, w=None):
        def prob(nlost, nfused, nfail):
            a = 1/pfail
            return ((1 - eta ** a) ** nlost) * (eta ** a * (1 - pfail)) ** nfused * (eta ** a * pfail) ** nfail
        prob_dict = {'fusion': 0, 'xrec': 0, 'zrec': 0}
        for f_res in self.fusion_outcomes:
            nl, nf, nfail = f_res['probs']
            p = prob(nl, nf, nfail)
            if f_res['x']:
                prob_dict['xrec'] += p
                if f_res['z']:
                    prob_dict['zrec'] += p
                    prob_dict['fusion'] += p
            elif f_res['z']:
                prob_dict['zrec'] += p
        return prob_dict


class AdaptiveFusionDecoder(FusionDecoder):
    def __init__(self, graph, meas_outcome_prefix=None):
        super(AdaptiveFusionDecoder, self).__init__(graph)
        self.q_lost = []
        self.fusion_failures = {'x':[], 'z': []}
        self.fusion_losses = []
        self.failure_q = None
        self.qubit_to_fuse = None
        self.fused_qubits = []
        self.pauli_done = Pauli([], [], self.nq, 0)
        self.strategies_remaining = None
        self.p_log_remaining = [None, None]
        if meas_outcome_prefix is None:
            self.success_pattern = []
        else:
            self.success_pattern = meas_outcome_prefix
        self.fusion_complete = False
        self.xlog_measured = False
        self.zlog_measured = False
        self.counter = {'x': 0, 'xf': 0, 'y': 0, 'yf': 0, 'z': 0, 'zf': 0, 'fusion': 0, 'ffx': 0, 'ffz': 0, 'floss':0}
        self.current_target = None
        self.strat = None
        self.target_pauli = None
        self.target_measured = False
        self.successful_outcomes = {'fusion': [], 'xrec': [], 'zrec': []}
        self.status_dict = {}
        self.finished = False
        self.results = []
        self.results_dicts = {}

    def set_params(self, successes, first_pass=False, pick_failure_basis=False):
        if first_pass:
            self.fusion_complete = False
            self.xlog_measured = False
            self.zlog_measured = False
            if pick_failure_basis:
                self.counter = {'x': 0, 'xf': 0, 'y': 0, 'yf': 0, 'z': 0, 'zf': 0, 'fusion': 0, 'ffx': 0, 'ffy': 0, 'ffz': 0, 'floss': 0}  # In this case all the bases are equivalent, but record them for debugging
                self.fusion_failures = {'x': [], 'y': [], 'z': []}
            else:
                self.counter = {'x': 0, 'xf': 0, 'y': 0, 'yf': 0, 'z': 0, 'zf': 0, 'fusion': 0, 'ffx': 0, 'ffz': 0, 'floss': 0}
                self.fusion_failures = {'x': [], 'z': []}
            self.q_lost = []
            self.fusion_losses = []
            self.fused_qubits = []
            self.pauli_done = Pauli([], [], self.nq, 0)
            self.failure_q = None
            self.qubit_to_fuse = None
            self.finished = False
        else:
            self.strategies_remaining, self.p_log_remaining, self.strat, self.current_target, self.q_lost, self.fusion_complete, \
            self.xlog_measured, self.zlog_measured, self.fused_qubits, self.fusion_failures, self.fusion_losses, self.pauli_done, \
            self.target_measured, self.counter, self.failure_q, self.qubit_to_fuse, self.finished = deepcopy(self.status_dict[tuple(successes)])
            self.success_pattern = successes.copy()
            self.target_pauli = self.strat.pauli

    def decode(self, starting_point=None, first_traversal=False, mc=False, pfail=0.5, w=0.5, eta=1.0, printing=False):
        if starting_point is None:
            assert first_traversal and mc
        self.set_params(starting_point, first_pass=first_traversal)

        # Identify the set of possible measurement strategies
        if first_traversal:
            self.strategies_remaining = gen_strats_from_stabs(self.stab_grp_nt, self.nq, get_individuals=True)
            logical_operators = gen_logicals_from_stab_grp(self.stab_grp_nt, self.nq)
            self.p_log_remaining = [[Strategy(p) for p in logical_operators[i]] for i in (0, 2)]
            self.strat = self.strategy_picker_new()
            self.target_pauli = self.strat.pauli
            self.current_target = self.strat.t
            self.cache_status()

        if printing:
            self.print_status()

        while not self.finished:
            # 1 Identify the best strategy and try to measure the target qubit
                # Only try to do fusion if we haven't done one yet, and there are still fusion strategies remaining
            while (not self.target_measured) and self.strategies_remaining:
                if self.qubit_to_fuse is None and self.failure_q is None:
                    # See if the target_qubit is lost
                    good = self.next_meas_outcome(starting_point, mc=mc, p=eta, first_traversal=first_traversal)
                    if good:
                        self.success_pattern.append(1)
                        self.qubit_to_fuse = self.current_target
                    else:
                        self.success_pattern.append(0)
                        self.q_lost.append(self.current_target)
                        self.counter['floss'] += 1
                        self.fusion_losses.append(self.current_target)
                    self.update_available_strats()
                    self.finished = self.decoding_finished()
                    if self.finished:
                        break
                    self.strat = self.strategy_picker_new()
                    self.target_pauli = self.strat.pauli
                    self.current_target = self.strat.t
                    self.cache_status()

                if self.qubit_to_fuse is not None:
                    assert self.failure_q is None
                    good = self.next_meas_outcome(starting_point, mc=mc, p=1-pfail, first_traversal=first_traversal)
                    if good:
                        self.success_pattern.append(1)  # here a 1 corresponds to successful fusion
                        self.fused_qubits.append(self.qubit_to_fuse)
                        self.counter['fusion'] += 1
                        self.target_measured = True
                    else:
                        self.success_pattern.append(0)
                        self.failure_q = self.qubit_to_fuse
                    self.qubit_to_fuse = None

                self.update_available_strats()
                self.finished = self.decoding_finished()
                if self.finished:
                    break
                self.strat = self.strategy_picker_new()
                self.target_pauli = self.strat.pauli
                self.current_target = self.strat.t
                self.cache_status()

                if self.failure_q is not None:  # decide in which basis the fusion failure qubit is to be measured
                    good = self.next_meas_outcome(starting_point, mc=mc, p=1-w, first_traversal=first_traversal)
                    if good:
                        self.success_pattern.append(1)  # here a 1 corresponds to measuring in the x basis
                        self.pauli_done.update_xs(self.failure_q, 1)
                        self.counter['ffx'] += 1
                        self.fusion_failures['x'].append(self.failure_q)
                    else:
                        self.success_pattern.append(0)
                        self.pauli_done.update_zs(self.failure_q, 1)
                        self.counter['ffz'] += 1
                        self.fusion_failures['z'].append(self.failure_q)
                    self.failure_q = None

                self.update_available_strats()
                self.finished = self.decoding_finished()
                if not self.finished:
                    self.strat = self.strategy_picker_new()
                    self.target_pauli = self.strat.pauli
                    self.current_target = self.strat.t
                    self.target_measured = self.current_target in self.fused_qubits
                    self.cache_status()
                if printing:
                    self.print_status()

            # Need to save the measurements done in the pre and post fusion parts of the decoder separately
            # Now we want to try the Pauli measurements
            if not self.finished:
                if printing:
                    self.print_status()
                self.strat = self.strategy_picker_new()
                self.target_pauli = self.strat.pauli
                self.current_target = self.strat.t
                self.target_measured = self.current_target in self.fused_qubits
                if self.target_measured or self.current_target is None:
                    q = list(set(self.target_pauli.support) - set(self.pauli_done.support) - set(self.fused_qubits))[0]  # Get next qubit to measure
                    meas_type = self.target_pauli.get_meas_type(q)

                    good = self.next_meas_outcome(starting_point, first_traversal=first_traversal)
                    if good:
                        self.success_pattern.append(1)
                        self.pauli_done.update_zs(q, self.target_pauli.zs[q])
                        self.pauli_done.update_xs(q, self.target_pauli.xs[q])
                        self.counter[meas_type] += 1
                    else:
                        self.success_pattern.append(0)
                        self.q_lost.append(q)
                        self.counter[f'{meas_type}f'] += 1
                    self.update_available_strats()
                    self.finished = self.decoding_finished()
                    self.cache_status()

                if printing:
                    self.print_status()

        return self.fusion_complete, self.xlog_measured, self.zlog_measured, self.success_pattern, self.pauli_done, self.fused_qubits, self.fusion_failures, self.fusion_losses, self.counter

    def cache_status(self):
        pattern = tuple(np.copy(self.success_pattern))
        if pattern not in self.status_dict.keys():
            # print(f'Caching {pattern}')
            s = self.strat.copy()
            self.status_dict[pattern] = (deepcopy(self.strategies_remaining), deepcopy(self.p_log_remaining), s,
                                         self.current_target, deepcopy(self.q_lost), self.fusion_complete,
                                         self.xlog_measured, self.zlog_measured,  deepcopy(self.fused_qubits),
                                         deepcopy(self.fusion_failures), self.fusion_losses.copy(), self.pauli_done.copy(),
                                         self.target_measured, deepcopy(self.counter),  deepcopy(self.failure_q),
                                         self.qubit_to_fuse, self.finished)

    def decoding_finished(self):
        """
        see if we have finished decoding, and whether logical fusion or logical pauli operators have been successfully
        measured
        :return:
        """
        protocol_finished = False
        if self.pauli_done is not None and self.target_pauli is not None:  # TODO don't actually require pauli done to be None here??
            # Remove the fused qubits from the target pauli operator
            p_target_minus_fused = self.target_pauli.copy()
            for i in self.fused_qubits:
                p_target_minus_fused.update_xs(i, 0)
                p_target_minus_fused.update_zs(i, 0)
            if self.pauli_done.contains_other(p_target_minus_fused) and (self.current_target in self.fused_qubits):
                self.fusion_complete = True
                self.xlog_measured = True  #TODO Check these explicitly?
                self.zlog_measured = True
                protocol_finished = True
            elif (not self.strategies_remaining) and (not protocol_finished):  # if the fusion has failed, see if we can still recover a logical xx or zz
                for x_log in self.p_log_remaining[0]:
                    if self.pauli_done.contains_other(x_log.pauli, exclude=self.fused_qubits):
                        self.xlog_measured = True
                        # try:
                        #     assert not self.p_log_remaining[1]
                        # except AssertionError:
                        #     self.print_status()
                        #     exit()
                        protocol_finished = True
                for z_log in self.p_log_remaining[1]:
                    if self.pauli_done.contains_other(z_log.pauli, exclude=self.fused_qubits):
                        self.zlog_measured = True
                        # assert not self.p_log_remaining[0]  #There shouldn't be any xlogicals remaining if pathfinding failed and zlogical succeeded
                        protocol_finished = True
                        break
                # assert not (self.xlog_measured and self.zlog_measured)  # Shouldn't be able to have done both if we failed pathfinding
                # You actually can have measured both - if you did multiple fusions the X and Z can anticommute on multiple qubits, but this isn't found by pathfinding.
            if (not self.p_log_remaining[0]) and (not self.p_log_remaining[1]):  # If no X and no Z left then we fail
                protocol_finished = True
            return protocol_finished

    def update_available_strats(self):
        """
        Update which measurement strategies remain for teleportation and logical x and z measurements.
        This differs from teleportation in the fact that the fused qubit can be used to perform a Pauli measurement
        later if required.
        :return: None
        """
        strats_xz = [deepcopy(self.p_log_remaining[0]), deepcopy(self.p_log_remaining[1])]
        tele_strats = deepcopy(self.strategies_remaining)
        self.strategies_remaining = [s for s in tele_strats if (len(set(self.q_lost) & set(s.pauli.support)) == 0 and s.t not in self.q_lost and s.pauli.commutes_every(self.pauli_done)
                                      and s.t not in set(self.pauli_done.support))]
        self.p_log_remaining = [[s for s in strats_xz[i] if (len(set(self.q_lost) & set(s.pauli.support)) == 0)
                                 and s.pauli.commutes_every(self.pauli_done)] for i in (0, 1)]

    def print_status(self):
        print(f'{self.pauli_done.to_str()=} \n {self.target_pauli.to_str()=} \n {self.current_target=} \n {self.q_lost=} \n '
              f'{self.fused_qubits=} \n {self.fusion_failures=} \n {self.success_pattern=}')
        print([(s.pauli.to_str(), s.t) for s in self.strategies_remaining])
        print([s.pauli.to_str() for s in self.p_log_remaining[0]])
        print([s.pauli.to_str() for s in self.p_log_remaining[1]])
        print(f'{self.xlog_measured=}\n{self.zlog_measured=}\n{self.fusion_complete=}')
        print(f'{self.finished=}')
        print(f'{self.counter=}')
        print('\n')

    def next_meas_outcome(self, starting_point, first_traversal=False, mc=False, p=None):
        """
        Assume we are doing the analytic decision tree building technique - so the outcome is determined by
        which parts of the tree we have already searched
        Always go fail, followed by all successes - this constitutes a depth-first traversal of the tree
        If we are doing shot-by-shot decoding (how you would run an experiment) we need to return a bit determined
        by the transmission probability of the channel
        """
        if mc:
            r = random()
            if r < p:
                return 1
            else:
                return 0
        else:
            if first_traversal:
                return 1
            if len(self.success_pattern) == len(starting_point):
                return 0
            elif len(self.success_pattern) > len(starting_point):
                return 1
            else:
                raise ValueError('outcome list should be longer than the starting point list')

    def build_tree(self, printing=False):
        """
        To find the analytic success probability of this method, perform a depth-first search of the decoder. Terminate
        once a valid solution or a failure is encountered.
        example (first search 1111(s), then 11101(s), then 11100(f), then 11011(s), 11010(f), 10111(s) ... etc.)
        """
        self.status_dict = {}
        success_count = 0
        first_pattern = [1] * self.nq
        first_pass = True
        out = first_pattern
        self.successful_outcomes = []
        while 1 in out:
            while out[-1] == 0:  # Trim trailing zeros from prefix, i.e. go up to the lowest point in the tree where there was a successful measurement
                del out[-1]
            success, xx_done, zz_done, success_pattern, paulis_done, fusions_done, fusion_failures_recovered, fusion_losses, counter = self.decode(
                starting_point=out[:-1], first_traversal=first_pass, printing=False)
            if set(paulis_done.support) & set(fusions_done):
                raise ValueError
            if success or xx_done or zz_done:
                if success:
                    result = 'fusion'
                elif xx_done:
                    result = 'xrec'
                else:
                    result = 'zrec'
                success_count += 1
                if printing:
                    print(f'{success_pattern=}')
                    print(f'{self.strat.pauli.to_str()=}')
                    print(f'{self.fused_qubits=}')
                    print(f'{self.fusion_failures=}')
                    print(f'{self.q_lost=}')
                    print(f'{self.pauli_done.to_str()=}')
                    print(f'{self.counter=}')
                    print(f'{self.fusion_complete=}')
                    print(f'{self.xlog_measured=}')
                    print(f'{self.zlog_measured=}')
                    print('\n')
                r = (result, self.fused_qubits, self.q_lost, self.fusion_failures, self.fusion_losses, self.counter, self.success_pattern)
                self.results.append(r)

            first_pass = False
            out = success_pattern
        self.compile_results_dict()

    def compile_results_dict(self):
        results_dict = {'fusion': {}, 'xrec': {}, 'zrec': {}}
        # print(self.results)
        for r in self.results:
            fused = tuple(r[1])
            fails_x = tuple(r[3]['x'])
            fails_z = tuple(r[3]['z'])
            fusion_losses = tuple(r[4])
            meas_done = r[0]
            key = (fused, fails_x, fails_z, fusion_losses)
            if key in results_dict[meas_done]:
                results_dict[meas_done][key].append(r[5])
            else:
                results_dict[meas_done][key] = [r[5]]
        self.results_dicts = results_dict
        # print(f'{results_dict=}')

    def get_probs_from_outcomes(self, eta, pfail, w):
        # For each possible output qubit in the results list, find the probability of the fusion being implemented
        # The pauli measurements required to get to this qubit are then summed over to find the probability of this strategy

        prob_dict = {'fusion': 0, 'xrec': 0, 'zrec': 0}
        for m_type in prob_dict.keys():
            # print(m_type)

            for key, value in self.results_dicts[m_type].items():
                # Find the probability of the fusion part succeeding
                fusions, ffx, ffz, losses = key
                prob_f = ((eta ** (1/pfail)) * (1-pfail)) ** len(fusions) * ((eta ** (1/pfail)) * pfail * (1 - w)) ** len(ffx) * ((eta ** (1/pfail)) * pfail * w) ** len(ffz) * (1 - eta ** (1/pfail)) ** len(losses)
                # Now find the prob of the Pauli part succeeding
                pp = 0
                for item in value:
                    nx = item['x']
                    ny = item['y']
                    nz = item['z']
                    nxf = item['xf']
                    nyf = item['yf']
                    nzf = item['zf']
                    pp += eta ** (nx + ny + nz) * (1-eta) ** (nxf + nyf + nzf)
                tot_prob = prob_f * (pp ** 2)
                # print(tot_prob, prob_f, pp, ffx, ffz, fusions, losses, value)
                prob_dict[m_type] += tot_prob
                if m_type == 'fusion':
                    prob_dict['xrec'] += tot_prob
                    prob_dict['zrec'] += tot_prob
        return prob_dict

    def strategy_picker_new(self, strategy_list_in=None, free_meas=None):
        """
        Do the MWE here, just take lowest weight measurements
        Want the lowest weight unmeasured thing
        TODO add sophistication to the picking method
        :param strategy_list_in: If none, use self.strategies_remaining.
                                Gives option to look ahead and pick from a subset of the remaining strategies
        :param free_meas: If a qubit is there and we are choosing the fusion failure basis
        :return:
        """
        if strategy_list_in is None:
            strategy_list_in = self.strategies_remaining
        if not strategy_list_in:
            # Look at the measurements for x or z basis measurements
            if self.p_log_remaining[0] or self.p_log_remaining[1]:
                strats = deepcopy(self.p_log_remaining[1]) + deepcopy(self.p_log_remaining[0])  # TODO the order of this list preferences for XX or ZZ
            else:
                strats = []
                assert self.decoding_finished()
                return
        else:
            strats = deepcopy(strategy_list_in)

        n_code_q = self.nq - 1
        lowest_uncorrectable = n_code_q
        lowest_weight = n_code_q
        current_winner = strats[0]
        winner_fusion_done = False
        for s in strats:
            new_winner = False
            unmeasured_support = set(s.pauli.support) - set(self.pauli_done.support)
            unmeasured_weight = len(unmeasured_support)
            if free_meas is not None:
                if free_meas in unmeasured_support:
                    unmeasured_weight -= 1

            # Include the fusion measurement as a required measurement
            if s.t is not None:
                fusion_done = s.t in self.fused_qubits
                if not fusion_done:
                    unmeasured_weight += 1
            else:
                fusion_done = True
            assert s.pauli.commutes_every(self.pauli_done)
            if winner_fusion_done and (not fusion_done):
                pass
            else:
                if unmeasured_weight < lowest_weight:
                    new_winner = True
            if new_winner:
                lowest_weight = unmeasured_weight
                winner_fusion_done = fusion_done
                current_winner = s
        return current_winner


class AdaptiveFailureBasisFusionDecoder(AdaptiveFusionDecoder):
    # This strategy tries to choose optimal fusion failure bases, as opposed to random ones
    def __init__(self, graph):
        super(AdaptiveFailureBasisFusionDecoder, self).__init__(graph)
        self.counter = {'x': 0, 'xf': 0, 'y': 0, 'yf': 0, 'z': 0, 'zf': 0, 'fusion': 0, 'ffx': 0, 'ffy': 0, 'ffz': 0, 'floss':0}

    def decode(self, starting_point=None, first_traversal=False, mc=False, pfail=0.5, w=0.5, eta=1.0, printing=False):
        """
        Given an initial bitstring representing how many qubits have been successfully measured or lost, load the
        status of the decoder (measurements remaining, qubits lost, measurements performed etc.) and calculate the
        next measurement to attempt.
        :param starting_point: bitstring of measurement successes. list
        :param first_traversal: if true, initiate all paramters to default values. bool
        :param mc: monte-carlo. each measurement is successful randomly with some probability. bool
        :param pfail: physical fusion failure probability. float
        :param w:
        :param eta: physical transmission
        :param printing:
        :return:
        """
        if starting_point is None:
            assert first_traversal and mc
        self.set_params(starting_point, first_pass=first_traversal, pick_failure_basis=True)

        # Identify the set of possible measurement strategies
        if first_traversal:
            self.strategies_remaining = gen_strats_from_stabs(self.stab_grp_nt, self.nq, get_individuals=True)
            logical_operators = gen_logicals_from_stab_grp(self.stab_grp_nt, self.nq)
            self.p_log_remaining = [[Strategy(p) for p in logical_operators[i]] for i in (0, 2)]
            self.strat = self.strategy_picker_new()
            self.target_pauli = self.strat.pauli
            self.current_target = self.strat.t
            self.cache_status()

        if printing:
            self.print_status()

        while not self.finished:
            # 1 Identify the best strategy and try to measure the target qubit
            # Only try to do fusion if we haven't done one yet, and there are still fusion strategies remaining
            while (not self.target_measured) and self.strategies_remaining:
                if self.qubit_to_fuse is None and self.failure_q is None:
                    # See if the target_qubit is lost
                    good = self.next_meas_outcome(starting_point, mc=mc, p=eta, first_traversal=first_traversal)
                    if good:
                        self.success_pattern.append(1)
                        self.qubit_to_fuse = self.current_target
                    else:
                        self.success_pattern.append(0)
                        self.q_lost.append(self.current_target)
                        self.counter['floss'] += 1
                        self.fusion_losses.append(self.current_target)
                    self.update_available_strats()
                    self.finished = self.decoding_finished()
                    if self.finished:
                        break
                    self.strat = self.strategy_picker_new()
                    self.target_pauli = self.strat.pauli
                    self.current_target = self.strat.t
                    self.cache_status()

                if self.qubit_to_fuse is not None:
                    assert self.failure_q is None
                    good = self.next_meas_outcome(starting_point, mc=mc, p=1-pfail, first_traversal=first_traversal)
                    if good:
                        self.success_pattern.append(1)  # here a 1 corresponds to successful fusion
                        self.fused_qubits.append(self.qubit_to_fuse)
                        self.counter['fusion'] += 1
                        self.target_measured = True
                    else:
                        self.success_pattern.append(0)
                        failure_basis = self.choose_failure_basis(self.qubit_to_fuse)

                        self.counter[f'ff{failure_basis}'] += 1
                        self.fusion_failures[failure_basis].append(self.qubit_to_fuse)
                        if failure_basis == 'x' or failure_basis == 'y':
                            self.pauli_done.update_xs(self.qubit_to_fuse, 1)
                        if failure_basis == 'y' or failure_basis == 'z':
                            self.pauli_done.update_zs(self.qubit_to_fuse, 1)
                    self.qubit_to_fuse = None

                self.update_available_strats()
                self.finished = self.decoding_finished()
                if self.finished:
                    break
                self.strat = self.strategy_picker_new()
                self.target_pauli = self.strat.pauli
                self.current_target = self.strat.t
                self.target_measured = (self.current_target in self.fused_qubits) or (self.current_target is None)
                self.cache_status()

            # Need to save the measurements done in the pre and post fusion parts of the decoder separately
            # Now we want to try the Pauli measurements
            if not self.finished:
                if printing:
                    self.print_status()
                self.strat = self.strategy_picker_new()
                self.target_pauli = self.strat.pauli
                self.current_target = self.strat.t
                self.target_measured = (self.current_target in self.fused_qubits) or (self.current_target is None)
                self.finished = self.decoding_finished()
                if (not self.finished) and self.target_measured:
                    try:
                        q = list(set(self.target_pauli.support) - set(self.pauli_done.support) - set(self.fused_qubits))[0]  # Get next qubit to measure (don't need to measure fused qubits!)
                    except IndexError:

                        self.print_status()
                        raise ValueError('No qubits left to measure, or the decoder has already finished')
                    meas_type = self.target_pauli.get_meas_type(q)
                    good = self.next_meas_outcome(starting_point, first_traversal=first_traversal)
                    if good:
                        self.success_pattern.append(1)
                        self.pauli_done.update_zs(q, self.target_pauli.zs[q])
                        self.pauli_done.update_xs(q, self.target_pauli.xs[q])
                        self.counter[meas_type] += 1
                    else:
                        self.success_pattern.append(0)
                        self.q_lost.append(q)
                        self.counter[f'{meas_type}f'] += 1
                    self.update_available_strats()
                    self.finished = self.decoding_finished()
                    self.cache_status()

                    if printing:
                        self.print_status()

        return self.fusion_complete, self.xlog_measured, self.zlog_measured, self.success_pattern, self.pauli_done, self.fused_qubits, self.fusion_failures, self.fusion_losses, self.counter

    def choose_failure_basis(self, failed_qubit):
        """
        Consider the set of measurements remaining where this qubit isn't the target. Pick the best one according to
        strategy_picker_new. Is there a measurement on this qubit? If so, use that basis.
        If not, choose the measurement basis that is in the greatest number of strategies.
        If this qubit is in no other strategies, it doesn't matter as it will never contribute
        :param failed_qubit: int
        :return: basis, string
        """
        new_strats_list = [s for s in self.strategies_remaining if s.t != failed_qubit]
        best = self.strategy_picker_new(strategy_list_in=new_strats_list, free_meas=failed_qubit)
        if failed_qubit in best.pauli.support:
            basis = best.pauli.get_meas_type(failed_qubit)
        else:
            other_strats_in = [s for s in new_strats_list if failed_qubit in s.pauli.support]
            if len(other_strats_in) == 0:  # This qubit is not in any more strategies so the basis doesn't matter
                basis = 'x'
            else:
                counts = {'x': 0, 'y': 0, 'z': 0}  # Otherwise, find the best failure basis, even though the qubit isn't in the chosen strategy
                for s in other_strats_in:
                    counts[s.pauli.get_meas_type(failed_qubit)] += 1
                basis = max(counts, key=counts.get)
        return basis

    def compile_results_dict(self):
        results_dict = {'fusion': {}, 'xrec': {}, 'zrec': {}}
        # print(self.results)
        for r in self.results:
            fused = tuple(r[1])
            fails_x = tuple(r[3]['x'])
            fails_y = tuple(r[3]['y'])
            fails_z = tuple(r[3]['z'])
            fusion_losses = tuple(r[4])
            meas_done = r[0]
            key = (fused, fails_x, fails_y, fails_z, fusion_losses)
            if key in results_dict[meas_done]:
                results_dict[meas_done][key].append(r[5])
            else:
                results_dict[meas_done][key] = [r[5]]
        self.results_dicts = results_dict

    def get_probs_from_outcomes(self, eta, pfail, w=None):
        """
        find the probability of fusion, xx and zz measurements from the analysis done by build_tree
        :param eta: transmission value to use
        :param pfail:
        :param w: The weighting of failure bases. Here failure basis is deterministic, so w is None
        :return:
        """
        # For each possible output qubit in the results list, find the probability of the fusion being implemented
        # The pauli measurements required to get to this qubit are then summed over to find the probability of this strategy

        prob_dict = {'fusion': 0, 'xrec': 0, 'zrec': 0}
        for m_type in prob_dict.keys():
            # print(m_type)
            if len(self.results_dicts['fusion'].keys()) == 0:
                self.build_tree()

            for key, value in self.results_dicts[m_type].items():
                # Find the probability of the fusion part succeeding
                fusions, ffx, ffy, ffz, losses = key
                prob_f = ((eta ** (1/pfail)) * (1-pfail)) ** len(fusions) * ((eta ** (1/pfail)) * pfail) ** (len(ffx) + len(ffy) + len(ffz)) * (1 - eta ** (1/pfail)) ** len(losses)
                # Now find the prob of the Pauli part succeeding
                pp = 0
                for item in value:
                    nx = item['x']
                    ny = item['y']
                    nz = item['z']
                    nxf = item['xf']
                    nyf = item['yf']
                    nzf = item['zf']
                    pp += eta ** (nx + ny + nz) * (1-eta) ** (nxf + nyf + nzf)
                tot_prob = prob_f * (pp ** 2)
                # print(tot_prob, prob_f, pp, ffx, ffz, fusions, losses, value)
                prob_dict[m_type] += tot_prob
                if m_type == 'fusion':
                    prob_dict['xrec'] += tot_prob
                    prob_dict['zrec'] += tot_prob
        return prob_dict


class AnalyticShor22(FusionDecoder):
    """
    The (2,2) Shor encoding with analytic success probability from http://arxiv.org/abs/2101.09310 (FBQC paper)
    """
    def __init__(self):
        super(AnalyticShor22, self).__init__()

    def get_probs_from_outcomes(self, eta, pfail, w):
        p0 = 1 - (1 - 0.5 * pfail) * eta ** (1/pfail)
        penc = 1 - 0.5 * ((1 - (1 - p0) ** 2) ** 2 + 1 - (1 - p0 ** 2) ** 2)
        prob_dict = {'fusion': None, 'xrec': penc, 'zrec': penc}
        # print(prob_dict)
        return prob_dict


def best_graphs_func(n=None, shor=False):
    """
    These are the best 6-9 qubit graphs found by Lowe for transversal fusion, optimised at pfail = 0.25
    :param n:
    :return:
    """
    g = nx.Graph()
    if shor:
        g.add_nodes_from(list(range(5)))
        g.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 4)])
    else:
        g.add_nodes_from(list(range(n)))
        if n == 6:
            g.add_edges_from([(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (2, 5)])
        elif n == 7:
            g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 6), (1, 4), (1, 5), (2, 4), (3, 4), (3, 5)])
        elif n == 8:
            g.add_edges_from([(0, 1), (0, 2), (0, 4), (0, 6), (0, 7), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (3, 5), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)])
        elif n == 9:
            g.add_edges_from([(0, 2), (0, 3), (0, 4), (0, 7), (0, 8), (1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 3), (2, 6), (2, 7), (2, 8), (3, 5), (3, 6), (3, 7), (3, 8),
            (4, 5), (4, 6), (4, 7), (4, 8), (5, 6), (5, 8), (6, 8), (7, 8)])
    return g


def get_fusion_performance(g, decoder_type='ACF', printing=False):
    """"
    Write a function which will retrun the results dictionaries for a given graph in a given decoder
    :param g: The graph to analyse (nx.Graph)
    :param decoder_type: Adaptive choose failure 'ACF', transversal choose failure 'TCF' or weighted 'AW', 'TW'
    :param printing: for debugging
    TODO Complete this function
    """
    if decoder_type == 'ACF':
        # Adaptive Fusion Decoder Choose Failure Basis
        dec = AdaptiveFailureBasisFusionDecoder(g)
        dec.build_tree(printing=printing)
        dec.compile_results_dict()
        dicts = dec.results_dicts
        # do stuff
    elif decoder_type == 'AW':
        # Adaptive Fusion decoder weighted failure
        dec = AdaptiveFusionDecoder(g)
        dec.build_tree(printing=printing)
        dec.compile_results_dict()
        dicts = dec.results_dicts

    elif decoder_type == 'TCF':
        # Transversal decoder with pre-chosen failure bases
        # Here we will return the dictionaries for every choice of failure basis
        dec = TransversalFusionDecoder(g)
        dec.decode()
        dec.get_fixed_basis_dicts()
        dicts = dec.fb_dicts

    elif decoder_type == 'TW':
        # Transversal decoder with weighted failure basis
        raise NotImplementedError
        pass

    else:
        raise ValueError
    return dicts


def prob_from_dict(eta, pfail, dicts_in, decoder_type='ACF'):
    """
    From the relevant decoder results dictionary, find the probability of successful fusion, xx or zz measurement
    :param eta: transmission
    :param pfail: probability of physical fusion gate failure
    :param dicts_in: results dictionary from the decoder
    :param decoder_type: which type of decoder to use, currently supported 'ACF' Adaptive choose failure or 'TCF'
    Transversal choose failure
    :return:
    """
    # use a small test graph to initialise the decoder - to get the probability it doesn't actually matter
    test_graph = gen_ring_graph(3)
    if decoder_type == 'ACF':
        # Adaptive Fusion Decoder Choose Failure Basis
        results_dicts = dicts_in

        prob_dict = {'fusion': 0, 'xrec': 0, 'zrec': 0}
        for m_type in prob_dict.keys():
            # print(m_type)
            for key, value in results_dicts[m_type].items():
                # Find the probability of the fusion part succeeding
                fusions, ffx, ffy, ffz, losses = key
                prob_f = ((eta ** (1/pfail)) * (1-pfail)) ** len(fusions) * ((eta ** (1/pfail)) * pfail) ** (len(ffx) + len(ffy) + len(ffz)) * (1 - eta ** (1/pfail)) ** len(losses)
                # Now find the prob of the Pauli part succeeding
                pp = 0
                for item in value:
                    nx = item['x']
                    ny = item['y']
                    nz = item['z']
                    nxf = item['xf']
                    nyf = item['yf']
                    nzf = item['zf']
                    pp += eta ** (nx + ny + nz) * (1-eta) ** (nxf + nyf + nzf)
                tot_prob = prob_f * (pp ** 2)
                # print(tot_prob, prob_f, pp, ffx, ffz, fusions, losses, value)
                prob_dict[m_type] += tot_prob
                if m_type == 'fusion':
                    prob_dict['xrec'] += tot_prob
                    prob_dict['zrec'] += tot_prob
        return prob_dict

    elif decoder_type == 'AW':
        # Adaptive Fusion decoder weighted failure
        dec = AdaptiveFailureBasisFusionDecoder(test_graph)
        dec.results_dicts = dicts_in
        prob_dict = dec.get_probs_from_outcomes(eta, pfail)

    elif decoder_type == 'TCF':
        # Transversal decoder with pre-chosen failure bases
        fb_dicts = dicts_in
        prob_dict = {'fusion': 0, 'xrec': 0, 'zrec': 0}
        na = 1 / pfail

        # Can only choose the best config for one outcome type - do fusion and if evens do x and z min after
        best_config = None
        max_p_fuse = 0
        max_p_rec_min = 0

        def prob(expr_dict):
            return sum(
                [v * (1 - eta ** na) ** k[0] * ((1 - pfail) * eta ** na) ** k[1] * (pfail * eta ** na) ** k[2] for
                 k, v in expr_dict.items()])

        for key, val in fb_dicts['fusion'].items():
            p_fuse = prob(val)
            if p_fuse > max_p_fuse:
                max_p_fuse = p_fuse
                best_config = key
        prob_dict['fusion'] = max_p_fuse
        if eta == 0:
            prob_dict['xrec'] = 0
            prob_dict['zrec'] = 0
        else:
            prob_dict['xrec'] = max_p_fuse + prob(fb_dicts['xrec'][best_config])
            prob_dict['zrec'] = max_p_fuse + prob(fb_dicts['zrec'][best_config])
        return prob_dict

    elif decoder_type == 'TW':
        # Transversal decoder with weighted failure basis
        pass

    else:
        raise ValueError


def rgs_success_probability(n, eta, pf=0.5):
    """
    the success probability of a link in the RGS graph is calculated according to the analytic form in https://www.nature.com/articles/ncomms7787
    :param n: Number of branches in the half RGS state. The number of qubits in the progenitor graph is 2n + 1, including the input
    :param eta:
    :param pf:
    :return:
    """
    px = pz = eta
    pfuse = eta ** (1/pf) * (1 - pf)
    return (1 - (1 - pfuse) ** n) * px ** 2 * pz ** (2*n-2)


def fusion_threshold_from_dict(dict, pf, take_min=True, pthresh=0.88, decoder_type='ACF'):
    """
    Get the FBQC threshold for the 6-ring approach to generating the Raussendorf lattice in http://arxiv.org/abs/2101.09310
    :param dict:
    :param pf:
    :param take_min:
    :param pthresh:
    :return:
    """

    def func_to_optimise(eta):
        probs = prob_from_dict(eta, pf, dict, decoder_type=decoder_type)
        if take_min:
            t = min([probs['xrec'], probs['zrec']]) - pthresh
        else:
            t = 0.5 * (probs['xrec'] + probs['zrec']) - pthresh
        return t

    try:
        threshold = bisection_search([0.88, 1], func_to_optimise)
    except ValueError:
        threshold = 1
    return threshold


if __name__ == '__main__':
    pass
