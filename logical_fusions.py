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
    def __init__(self, graph):
        self.nq = graph.number_of_nodes()
        self.graph = graph
        stab_generators = stabilizers_from_graph(graph)
        self.stab_grp_t, self.stab_grp_nt = gen_stabs_from_generators(stab_generators, split_triviality=True)
        self.logical_operators = gen_logicals_from_stab_grp(self.stab_grp_nt, self.nq)
        self.logical_pairs = self.get_logical_pairs()
        self.fusion_outcomes = None
        # for p in self.logical_pairs:
        #     print(p[0].to_str(), p[1].to_str())
        #     print(p[0].anticommuting_ix(p[1]))

    def get_logical_pairs(self):
        code_qubits = list(range(1, self.nq))
        full_set = [x for x in product(self.logical_operators[0], self.logical_operators[2])]
        # turn each in to a dictionary of the qubit indices that require x, y, z and fusion measurements.
        out = []
        for pair in full_set:
            nu_dict = {'x':[], 'y':[], 'z':[]}
            anticoms = pair[0].anticommuting_ix(pair[1])
            coms = list(set(code_qubits) - set(anticoms))
            nu_dict['fusion'] = anticoms
            for q in coms:
                if q in pair[0].support:
                    m_type = pair[0].get_meas_type(q)
                elif q in pair[1].support:
                    m_type = pair[1].get_meas_type(q)
                else:
                    m_type = None
                if m_type:
                    nu_dict[m_type].append(q)
            out.append(nu_dict)
            # print(pair[0].to_str(), pair[1].to_str(), nu_dict)
        return out

    def get_xz_logicals(self):
        """
        Get the x and z logical operators in the form of lists of the x meas and z meas qubits
        :return:
        """
        log_all = {'x': self.logical_operators[0], 'z': self.logical_operators[2]}
        log_out = {'x': [], 'z': []}
        for m_type in ['x', 'z']:
            for p in log_all[m_type]:
                # Get rid of anything with Y measurement
                good = True
                for q in p.support:
                    if p.get_meas_type(q) == 'y':
                        good = False
                if good:
                    log_out[m_type].append((set(p.x_ix), set(p.z_ix), p))
        return log_out

    def fbqc_decode(self, monte_carlo=False):
        """
        Follow the FBQC approach to finding FBQC loss tolerance thresholds
        xx measurements are independent of zz measurements
        randomize the type of fusion so that zz is achieved on fusion failure with probability w, xx with (1-w)
        do a monte-carlo sim, i.e. randomly generate loss sets and failures
        Calculate the
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
                fused = [i + 1 for i, x in enumerate(outcomes) if x == "fused"]
                xrec = [i + 1 for i, x in enumerate(outcomes) if x == "x"]
                zrec = [i + 1 for i, x in enumerate(outcomes) if x == "z"]
                probs = (len(lost), len(fused), len(xrec), len(zrec))
                ops_measured = {'x': False, 'z': False, 'probs': probs}
                for m_type in ('x', 'z'):
                    for operator in logical_operators[m_type]:
                        if operator[0].issubset(set(fused).union(set(xrec))) and operator[1].issubset(set(fused).union(set(zrec))):
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
        pfusion, pxx, pzz = 0, 0, 0
        for f_res in self.fusion_outcomes:
            nl, nf, nx, nz = f_res['probs']
            p = prob(nl, nf, nx, nz)
            if f_res['x']:
                pxx += p
                if f_res['z']:
                    pzz += p
                    pfusion += p
            elif f_res['z']:
                pzz += p

        return pfusion, pxx, pzz

    def perasure_fbqc(self, transmission, pfail):
        """ Find the p_erasure that is then used for FBQC thresholds. Do this by finding the maximum value of the average"""
        ws = np.linspace(0, 1)
        outs = []
        for w in ws:
            outs.append(self.get_probs_from_outcomes(eta=transmission, pfail=pfail, w=w))
        avgs = [0.5 * (o[1] + o[2]) for o in outs]
        max_ix = np.argmax(avgs)
        return outs[max_ix], ws[max_ix]

    def get_threshold(self, pfail=0.5, pthresh=0.88, optimal=False, n_steps=50):
        """
        Using the flag optimal=True finds the best w for every configuration of pfail and eta separately
        This is very slow, ~ 20 sec for 6 qubits. Instead, Find the maximum w for a particular value and use this for all
        """
        transmissions = np.linspace(0.95, 0.99999, n_steps)
        if optimal:
            for t in transmissions:
                max_av, max_w = self.perasure_fbqc(t, pfail)
                pf, px, pz = self.get_probs_from_outcomes(t, pfail, max_w)
                if 0.5 * (px + pz) > pthresh:
                    return t, max_w
        else:
            max_av, max_w = self.perasure_fbqc(0.98, pfail)
            print(max_w)

            def func_to_optimise(t):
                pf, px, pz = self.get_probs_from_outcomes(t, pfail, max_w)
                return 0.5 * (px + pz) - pthresh

            threshold = bisection_search([0.95, 1], func_to_optimise)
            return threshold, max_w

            # for t in transmissions:
            #     pf, px, pz = self.get_probs_from_outcomes(t, pfail, max_w)
            #     if 0.5 * (px + pz) > pthresh:
            #         return t, max_w
        return 1, None

    def plot_threshold(self, n_points=7, pthresh=0.88, optimal=False, n_steps_sweep=50, show=True):
        pfails = np.linspace(0.001, 0.5, n_points)
        data = [1-self.get_threshold(p, pthresh=pthresh, optimal=optimal, n_steps=n_steps_sweep)[0] for p in pfails]
        print(data)
        plt.plot(pfails, data, 'o-')
        if show:
            plt.xlabel('pfail')
            plt.ylabel('loss threshold')
            plt.show()
    #
    # def brute_force_decode(self, fusion_failure='z'):
    #     """
    #     Try analysing the performance of the graph under transversal fusions to see if we manage to perform a logical
    #     fusion operation.
    #     Try all possible configurations of loss and fusion failure, calculate the probability of each, and see if it
    #     recovers a logical x and logical z simultaneouslu
    #     TODO Allow the weights of Z failures and X failures to vary so we can optimise over this free parameter
    #     :return:
    #     """
    #     successes = []
    #     code_qubits = list(range(self.nq))
    #     code_qubits.remove(0)
    #     for n_lost in range(self.nq):
    #         for losts in combinations(code_qubits, n_lost):
    #             transmitted = list(set(code_qubits) - set(losts))
    #             for n_fail in range(len(transmitted)):
    #                 for fails in combinations(transmitted, n_fail):
    #                     successful_fusion = list(set(transmitted) - set(fails))
    #
    #                     log_pair_measured_probs = []
    #                     for log_pair in self.logical_pairs:
    #                         #TODO Find the probability of each logical pair being measured, given a randomised failure outcome
    #                         #TODO append the maximum probability to the successes list
    #
    #                         if self.logical_pair_measured(log_pair, successful_fusion, fails, losts, fusion_failure):
    #                             # if n_lost == 0:
    #                             #     print('NLOST=0')
    #                             #     exit()
    #                             # print(log_pair, losts, fails, unfails)
    #
    #                             successes.append((log_pair, n_lost, n_fail))
    #                             break
    #     return successes
    #
    # def logical_pair_measured(self, pair, successes, fails, losses, meas_on_failure):
    #     sucset, failset, losset = set(successes), set(fails), set(losses)
    #     paulis = ['x', 'y', 'z']
    #     paulis.remove(meas_on_failure)
    #     #TODO randomize the recovered basis at each attempted measurement
    #     recoverable_ix = set(pair[meas_on_failure])
    #     unrecoverable_ix = set(pair[paulis[0]]+pair[paulis[1]])
    #     if set(pair['fusion']).issubset(sucset) and unrecoverable_ix.issubset(sucset) and recoverable_ix.issubset(sucset.union(failset)):
    #         return True
    #     else:
    #         return False
    #
    # def p_fusion(self, eta, fuse_prob, successes):
    #     win_prob = 0
    #     for result in successes:
    #         logical_pair, n_lost, n_failed = result
    #         n_transmitted = self.nq - 1 - n_lost
    #         n_fused = n_transmitted - n_failed
    #         prob = eta ** (1/(1-fuse_prob) * n_transmitted) * (1 - eta ** (1/(1-fuse_prob))) ** n_lost * fuse_prob ** n_fused * (1-fuse_prob) ** n_failed
    #         # print(f'{logical_pair=}', f'{n_transmitted=}', f'{n_lost=}', f'{n_fused=}', f'{n_failed=}', prob)
    #         win_prob += prob
    #     return win_prob


class AdaptiveFusionDecoder:
    def __init__(self, graph, meas_outcome_prefix=None):
        self.nq = graph.number_of_nodes()
        self.graph = graph
        stab_generators = stabilizers_from_graph(graph)
        self.stab_grp_t, self.stab_grp_nt = gen_stabs_from_generators(stab_generators, split_triviality=True)
        self.q_lost = []
        self.fusion_failures = {'x':[], 'z': []}
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
        self.counter = {'x': 0, 'xf': 0, 'y': 0, 'yf': 0, 'z': 0, 'zf': 0, 'fusion': 0, 'ffx': 0, 'ffz': 0}
        self.current_target = None
        self.strat = None
        self.target_pauli = None
        self.target_measured = False
        self.successful_outcomes = {'fusion': [], 'xrec': [], 'zrec': []}
        self.status_dict = {}
        self.finished = False
        self.depth = 0  #TODO artifact of other decoder, remove when possible

    def set_params(self, successes, first_pass=False):
        if first_pass:
            self.fusion_complete = False
            self.xlog_measured = False
            self.zlog_measured = False
            self.counter = {'x': 0, 'xf': 0, 'y': 0, 'yf': 0, 'z': 0, 'zf': 0, 'fusion': 0, 'ffx': 0, 'ffz': 0}
            self.q_lost = []
            self.fusion_failures = {'x': [], 'z': []}
            self.fused_qubits = []
            self.pauli_done = Pauli([], [], self.nq, 0)
            self.failure_q = None
            self.qubit_to_fuse = None
            self.finished = False
        else:
            self.strategies_remaining, self.p_log_remaining, self.strat, self.current_target, self.q_lost, self.fusion_complete, \
            self.xlog_measured, self.zlog_measured, self.fused_qubits, self.fusion_failures, self.pauli_done, \
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
                    self.cache_status()

                if self.qubit_to_fuse is not None:
                    assert self.failure_q is None
                    good = self.next_meas_outcome(starting_point, mc=mc, p=1-pfail, first_traversal=first_traversal)
                    if good:
                        self.success_pattern.append(1)  # here a 1 corresponds to measuring in the x basis
                        self.fused_qubits.append(self.qubit_to_fuse)
                        self.counter['fusion'] += 1
                        self.target_measured = True
                    else:
                        self.success_pattern.append(0)
                        self.failure_q = self.qubit_to_fuse
                    self.qubit_to_fuse = None

                self.update_available_strats()
                self.cache_status()  # TODO Implement the caching if you want this feature (not necessary but should be faster)

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

                if printing:
                    self.print_status()

        return self.fusion_complete, self.xlog_measured, self.zlog_measured, self.success_pattern, self.pauli_done, self.fused_qubits, self.fusion_failures, self.counter

                # do this
                #4 possible outcomes for an attempted target measurement - need to consider how all of them affect future measurements
                # update the remaining strategies
        #
        # while not self.lt_finished:
        #     # Try to do the remaining pauli mesurements required for teleportation to the fused qubit
        #     pass

    def cache_status(self):
        pattern = tuple(np.copy(self.success_pattern))
        if pattern not in self.status_dict.keys():
            print(f'Caching {pattern}')
            s = self.strat.copy()
            self.status_dict[pattern] = (deepcopy(self.strategies_remaining), deepcopy(self.p_log_remaining), s,
                                         self.current_target, deepcopy(self.q_lost), self.fusion_complete,
                                         self.xlog_measured, self.zlog_measured,  deepcopy(self.fused_qubits),
                                         deepcopy(self.fusion_failures), self.pauli_done.copy(),
                                         self.target_measured, self.counter,  deepcopy(self.failure_q),
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
                        assert not self.p_log_remaining[1]  #TODO maybe print here
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
        :return: None
        """
        strats_xz = [deepcopy(self.p_log_remaining[0]), deepcopy(self.p_log_remaining[1])]
        tele_strats = deepcopy(self.strategies_remaining)
        self.strategies_remaining = [s for s in tele_strats if (len(set(self.q_lost) & set(s.pauli.support)) == 0 and len(
            set(self.fused_qubits) & set(s.pauli.support)) == 0 and s.t not in self.q_lost and s.pauli.commutes_every(self.pauli_done)
                                      and s.t not in set(self.pauli_done.support))]
        self.p_log_remaining = [[s for s in strats_xz[i] if (len(set(self.q_lost) & set(s.pauli.support)) == 0)
                                 and s.pauli.commutes_every(self.pauli_done)] for i in (0, 1)]

    def print_status(self):
        print(f'{self.pauli_done.to_str()=} \n {self.target_pauli.to_str()=} \n {self.current_target=} \n {self.q_lost=} \n '
              f'{self.fused_qubits=} \n {self.fusion_failures=} \n {self.success_pattern=}')
        print([s.pauli.to_str() for s in self.strategies_remaining])
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
            success, xx_done, zz_done, success_pattern, paulis_done, fusions_done, fusion_failures_recovered,  counter = self.decode(
                starting_point=out[:-1], first_traversal=first_pass)
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
                    print(f'{self.fusion_complete=}')
                    print(f'{self.xlog_measured=}')
                    print(f'{self.zlog_measured=}')
                    print('\n')

                #TODO save these results somewhere so that the success probabilities for fusion, x and z measurements can be calculated

            first_pass = False
            out = success_pattern


    def get_probabilitites(self):
        # For each possible output qubit in the results list, find the probability of the fusion being implemented
        # The pauli measurements required to get to this qubit are then summed over to find the probability of this strategy

        pass

    def strategy_picker_new(self):
        """
        Do the MWE here, just take lowest weight measurements
        TODO add sophistication to the picking method
        :return:
        """
        if not self.strategies_remaining:
            # Look at the measurements for x or z basis measurements
            if self.p_log_remaining[0]:
                strats = deepcopy(self.p_log_remaining[0])
            elif self.p_log_remaining[1]:
                strats = deepcopy(self.p_log_remaining[1])
            else:
                strats = []
                self.decoding_finished()
                return
        else:
            strats = deepcopy(self.strategies_remaining)

        n_code_q = self.nq - 1
        lowest_uncorrectable = n_code_q
        lowest_weight = n_code_q
        current_winner = strats[0]
        for s in strats[1:]:
            new_winner = False
            weight = s.pauli.weight()
            if weight < lowest_weight:
                new_winner = True

            if new_winner:
                lowest_weight = weight
                current_winner = s
        return current_winner

def strategy_picker(self):

        """
        We want a combination of the measurement with the smallest support and the largest number of compatible measurements
        want to ensure that as many qubits as possible in the pattern can be lost, and we can still switch to something else
        also want the patterns you switch to to satisfy these criteria - can do this to varying depth for more/less accurate
        decoding
        TODO Calculate a score for each potential next measurement pattern to see how good it would be - do this to varying depths to have faster/slower and less/more accurate decoding for each potential measurement
        TODO This whole function needs to be re-written to be compatible with the adaptive fusion decoder.
        """
        current_winner = self.strategies_remaining[0]
        # print([a.pauli.to_str() for a in self.strategies_remaining])
        # print(f'{current_winner.pauli.to_str()=}')
        num_q = self.nq
        lowest_uncorrectable = num_q
        lowest_weight = num_q
        lowest_av_weight_correction = num_q
        lowest_non_z = num_q
        for s in self.strategies_remaining:
            nu_winner = False
            # print(s.pauli.to_str())
            # print(f'{lowest_uncorrectable=}')
            weight = len(s.pauli.support)
            num_not_z = sum(s.pauli.xs)
            num_not_tolerant = 0
            if s.t == self.current_target or self.current_target is None:
                if self.depth == 0 and weight > lowest_weight:
                    pass
                else:
                    # Find the number of qubits whose loss cannot be tolerated, and the average weight of these measurements
                    tot_weight_corr = 0
                    sub_strats = self.update_available_strats(arb_meas=[s.t])

                    for q in s.to_measure(self.q_lost, self.pauli_done,
                                          None):  # For each qubit that could be lost, are we tolerant?
                        sub_sub_strats = self.update_available_strats(lost=[q], strats=sub_strats)
                        # print(f'{q=}, {[s.pauli.to_str() for s in sub_sub_strats]}')
                        if len(sub_sub_strats) == 0:
                            num_not_tolerant += 1
                        elif self.depth == 1:
                            min_weight_corr = min([len(s.pauli.support) for s in sub_sub_strats])
                            tot_weight_corr += min_weight_corr
                    av_weight_corr = tot_weight_corr / (weight - num_not_tolerant) if num_not_tolerant != weight else 0
                    # print(num_not_tolerant, av_weight_corr)
                    if num_not_tolerant < lowest_uncorrectable:
                        nu_winner = True
                    elif num_not_tolerant == lowest_uncorrectable:
                        if weight < lowest_weight:
                            nu_winner = True
                        elif weight == lowest_weight:
                            if av_weight_corr <= lowest_av_weight_correction:
                                if self.prefer_z:
                                    if num_not_z <= lowest_non_z:
                                        nu_winner = True
                                else:
                                    nu_winner = True
                    if nu_winner:
                        current_winner = s
                        lowest_weight = len(s.pauli.support)
                        lowest_uncorrectable = num_not_tolerant
                        lowest_av_weight_correction = av_weight_corr
                        lowest_non_z = num_not_z
        return current_winner





def best_graphs_func(n):
    g = nx.Graph()
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


def test_new_decoding():
    ## For shor code ##
    # g = nx.Graph()
    # g.add_nodes_from(list(range(5)))
    # g.add_edges_from([(0, 1), (1, 2), (0, 3), (3, 4)])
    # decoder = FusionDecoder(g)
    # decoder.fbqc_decode()
    # decoder.plot_threshold(n_steps_sweep=100, show=False)
    for nq in (6, 7, 8, 9):
        g = best_graphs_func(nq)
        # draw_graph(g)

        decoder = FusionDecoder(g)
        decoder.fbqc_decode()
        decoder.plot_threshold(n_steps_sweep=100, show=False)
    plt.xlabel('pfail')
    plt.ylabel('loss threshold')
    plt.legend(['2,2 Shor', 6, 7, 8, 9])
    plt.show()


def main():
    testing_adaptive_fusion_decoding()
    exit()


def testing_adaptive_fusion_decoding():
    g = gen_ring_graph(3)
    dec = AdaptiveFusionDecoder(g)
    dec.build_tree(printing=True)


if __name__=='__main__':
    main()
