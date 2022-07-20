import networkx as nx
import numpy as np
from pauli_class import Pauli, Strategy
from stab_formalism import stabilizers_from_graph, gen_stabs_from_generators, gen_logicals_from_stab_grp, \
    gen_strats_from_stabs
from itertools import combinations, product
from graphs import gen_ring_graph, draw_graph
import matplotlib.pyplot as plt
from helpers import bisection_search
from copy import deepcopy
from random import random


class FusionDecoder:
    def __init__(self, graph=None):
        if graph is None:
            pass
        else:
            self.nq = graph.number_of_nodes()
            self.graph = graph
            stab_generators = stabilizers_from_graph(graph)
            self.stab_grp_t, self.stab_grp_nt = gen_stabs_from_generators(stab_generators, split_triviality=True)

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
        return threshold  #TODO also get w_max??

    def plot_threshold(self, n_points=27, pthresh=0.88, optimising_eta=0.96, show=True, w=None, line='--', take_min=True, print_max_w=False, no_w=False):
        pfails = np.linspace(0.001, 0.5, n_points)
        data = [1-self.get_threshold(p, pthresh=pthresh, w=w, optimising_eta=optimising_eta, take_min=take_min, print_max_w=print_max_w, no_w=no_w) for p in pfails]
        # print(data)
        plt.plot(pfails, data, line)
        if show:
            plt.xlabel('pfail')
            plt.ylabel('loss threshold')
            plt.show()


class TransversalFusionDecoder(FusionDecoder):
    def __init__(self, graph):
        super(TransversalFusionDecoder, self).__init__(graph)
        self.logical_operators = gen_logicals_from_stab_grp(self.stab_grp_nt, self.nq)
        self.fusion_outcomes = None

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
                xs = set(p.x_ix)
                zs = set(p.z_ix)
                good = True
                # for q in p.support:
                #     if p.get_meas_type(q) == 'y':
                #         good = False
                if good:
                    log_out[m_type].append((xs - zs, xs & zs, zs - xs, p))
        return log_out

    def fbqc_decode(self, monte_carlo=False):
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
            # Generate all sets of possible outcomes, [loss, fused, fail recover x, fail recover z]
            # Probabilities [(1 - eta ** 1/pfail), eta**1/pfail * (1 - pfail), eta**1/pfail * (1-w)pfail, eta ** 1/pfail * wpfail]
            for outcomes in product(['loss', 'fused', 'x', 'z'], repeat=self.nq-1):
                lost = [i + 1 for i, x in enumerate(outcomes) if x == "loss"]
                fused = set([i + 1 for i, x in enumerate(outcomes) if x == "fused"])
                xrec = set([i + 1 for i, x in enumerate(outcomes) if x == "x"])
                zrec = set([i + 1 for i, x in enumerate(outcomes) if x == "z"])
                xm = fused.union(xrec)
                ym = fused
                zm = fused.union(zrec)
                probs = (len(lost), len(fused), len(xrec), len(zrec))
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

    def get_probs_from_outcomes(self, eta, pfail, w):
        def prob(nlost, nfused, nxrec, nzrec):
            a = 1/pfail
            return ((1 - eta ** a) ** nlost) * (eta ** a * (1 - pfail)) ** nfused *\
                   (eta ** a * (1 - w) * pfail) ** nxrec * (eta ** a * w * pfail) ** nzrec
        prob_dict = {'fusion': 0, 'xrec': 0, 'zrec': 0}
        for f_res in self.fusion_outcomes:
            nl, nf, nx, nz = f_res['probs']
            p = prob(nl, nf, nx, nz)
            if f_res['x']:
                prob_dict['xrec'] += p
                if f_res['z']:
                    prob_dict['zrec'] += p
                    prob_dict['fusion'] += p
            elif f_res['z']:
                prob_dict['zrec'] += p
        return prob_dict


class DiffBasisTransversalDecoder(FusionDecoder):
    """ This strategy we choose the failure basis for the fusion measurements independently for each qubit"""
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
        TODO Prioritise low weight measurements?
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
        TODO Try removing operators from list as you go through qubits to see if this is faster
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

                self.update_available_strats() #TODO Configure this so it will update the teleportation and logical pauli measurements available
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

                q = list(set(self.target_pauli.support) - set(self.pauli_done.support))[0]  # Get next qubit to measure

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
        if self.pauli_done is not None and self.target_pauli is not None:
            if self.pauli_done.contains_other(self.target_pauli) and (self.current_target in self.fused_qubits):
                self.fusion_complete = True
                self.xlog_measured = True  #TODO Check these explicitly?
                self.zlog_measured = True
                protocol_finished = True
            elif (not self.strategies_remaining) and (not protocol_finished):  # if the fusion has failed, see if we can still recover a logical xx or zz
                for x_log in self.p_log_remaining[0]:
                    if self.pauli_done.contains_other(x_log.pauli, exclude=self.fused_qubits):
                        self.xlog_measured = True
                        try:
                            assert not self.p_log_remaining[1]  #TODO maybe print here
                        except AssertionError:
                            self.print_status()
                            exit()
                        protocol_finished = True
                        break
                for z_log in self.p_log_remaining[1]:
                    if self.pauli_done.contains_other(z_log.pauli, exclude=self.fused_qubits):
                        self.zlog_measured = True
                        assert not self.p_log_remaining[0]  #There shouldn't be any xlogicals remaining if pathfinding failed and zlogical succeeded
                        protocol_finished = True
                        break
                assert not (self.xlog_measured and self.zlog_measured)  # Shouldn't be able to have done both if we failed pathfinding
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
                #TODO save these results somewhere so that the success probabilities for fusion, x and z measurements can be calculated

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
                strats = deepcopy(self.p_log_remaining[0]) + deepcopy(self.p_log_remaining[1])
                if self.p_log_remaining[0] and self.p_log_remaining[1] and (not free_meas):
                    self.print_status()
                    # raise ValueError
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
    #TODO merge this functionality in to the parent class
    def __init__(self, graph):
        super(AdaptiveFailureBasisFusionDecoder, self).__init__(graph)
        self.counter = {'x': 0, 'xf': 0, 'y': 0, 'yf': 0, 'z': 0, 'zf': 0, 'fusion': 0, 'ffx': 0, 'ffy': 0, 'ffz': 0, 'floss':0}

    def decode(self, starting_point=None, first_traversal=False, mc=False, pfail=0.5, w=0.5, eta=1.0, printing=False):
        # TODO modify this function so the fusion failure basis is deterministic
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

                        ########TODO WRITE IN NEW FUNCTIONALITY HERE ###########
                            # rewrite ffy into counter
                            # write a choose_failure_basis function
                            #TODO remove any references to w or probabilistic failure bases
                            #TODO Update calculation of probabilities
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
                self.target_measured = self.current_target in self.fused_qubits
                self.cache_status()

            # Need to save the measurements done in the pre and post fusion parts of the decoder separately
            # Now we want to try the Pauli measurements
            if not self.finished:
                if printing:
                    self.print_status()
                self.strat = self.strategy_picker_new()
                self.target_pauli = self.strat.pauli
                self.current_target = self.strat.t
                self.target_measured = self.current_target in self.fused_qubits

                q = list(set(self.target_pauli.support) - set(self.pauli_done.support))[0]  # Get next qubit to measure

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
        TODO If not, think of some other way that chooses but will also give invariance under local complementation
        TODO When considering the weight of these measurements, also consider the fact that we have a free measurement of
                failed_qubit, so that should be excluded from the weight
        :param failed_qubit: int
        :return: basis, string
        """
        new_strats_list = [s for s in self.strategies_remaining if s.t != failed_qubit]
        best = self.strategy_picker_new(strategy_list_in=new_strats_list, free_meas=failed_qubit)
        if failed_qubit in best.pauli.support:
            basis = best.pauli.get_meas_type(failed_qubit)
        else:
            notin = [s for s in new_strats_list if failed_qubit not in s.pauli.support]
            counts = {'x': 0, 'y': 0, 'z': 0}
            for s in notin:
                counts[s.pauli.get_meas_type(failed_qubit)] += 1
            basis = max(counts, key=counts.get)
            if max(counts.values()) == 0:
                raise ValueError("How do we choose a basis fairly here? I guess it doesn't matter as it cannot contribute?")
        # self.print_status()
        print(f'{failed_qubit=}, {basis=}, {best.pauli.to_str()=}')
        print([s.pauli.to_str() for s in new_strats_list])
        print('\n')
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
        # print(f'{results_dict=}')

    def get_probs_from_outcomes(self, eta, pfail, w):
        # For each possible output qubit in the results list, find the probability of the fusion being implemented
        # The pauli measurements required to get to this qubit are then summed over to find the probability of this strategy

        prob_dict = {'fusion': 0, 'xrec': 0, 'zrec': 0}
        for m_type in prob_dict.keys():
            # print(m_type)

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
    def __init__(self):
        super(AnalyticShor22, self).__init__()

    def get_probs_from_outcomes(self, eta, pfail, max_w):
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


if __name__ == '__main__':
    g = gen_ring_graph(5)
    g = best_graphs_func(8)
    draw_graph(g)
    db_dec = DiffBasisTransversalDecoder(g)
    db_dec.decode(picker='maxmin')
    db_dec.plot_threshold(show=False, w=0, take_min=True)

    t_dec = TransversalFusionDecoder(g)
    t_dec.fbqc_decode()
    t_dec.plot_threshold(show=False, take_min=True, print_max_w=True, optimising_eta=0.96, n_points=9)

    adaptive = AdaptiveFusionDecoder(g)
    adaptive.build_tree()
    adaptive.plot_threshold(take_min=True, show=False, print_max_w=True)

    adaptive_choose_failure = AdaptiveFailureBasisFusionDecoder(g)
    adaptive_choose_failure.build_tree()
    adaptive_choose_failure.plot_threshold(take_min=True, no_w=True, show=False)
    plt.legend([1, 2, 3, 4])
    plt.show()


    # adaptive.plot_threshold(take_min=False)

