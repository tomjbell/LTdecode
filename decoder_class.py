import random
from pauli_class import Pauli, Strategy, multi_union
from stab_formalism import is_compatible, stabilizers_from_graph, gen_strats_from_stabs, gen_logicals_from_stab_grp, gen_stabs_from_generators
import numpy as np
from error_correction import pauli_error_decoder
from copy import deepcopy


class DecoderOutcome:
    def __init__(self, counter, checks, target_paulis, output):
        self.counter = counter
        self.checks = checks
        self.target_paulis = target_paulis
        self.t = output

    def no_flip_prob(self, ps):
        if self.t is not None:
            probs, confidences = pauli_error_decoder(self.target_paulis, self.checks, ps, ignore=(0, self.t))
        else:
            probs, confidences = pauli_error_decoder(self.target_paulis, self.checks, ps, ignore=(0,))
        return probs[0]

    def outcome_prob(self, transmission):
        counts = [val for val in self.counter.values()]
        # Assuming no cascading
        aa = xx = yy = zz = transmission
        af = xf = yf = zf = 1 - transmission
        azi = xzi = yzi = zzi = 0
        return xx ** counts[0] * xzi ** counts[1] * xf ** counts[2] * zz ** counts[3] * zzi ** counts[4]\
               * zf ** counts[5] * yy ** counts[6] * yzi ** counts[7] * yf ** counts[8] * aa ** counts[9]\
               * azi ** counts[10] * af ** counts[11]


class CascadeDecoder:
    def __init__(self, graph, meas_outcome_prefix=None, depth=0, prefer_z=True):
        self.nq = graph.number_of_nodes()
        self.graph = graph
        stab_generators = stabilizers_from_graph(graph)
        self.stab_grp_t, self.stab_grp_nt = gen_stabs_from_generators(stab_generators, split_triviality=True)
        self.q_lost = []
        self.purgatory_q = None  # If there is a qubit that has been lost, but we haven't tried to measure it indirectly yet
        self.purg_q_basis = None
        self.pauli_done = Pauli([], [], self.nq, 0)
        self.arb_meas = []
        self.strategies_remaining = None
        if meas_outcome_prefix is None:
            self.success_pattern = []
        else:
            self.success_pattern = meas_outcome_prefix
        self.prefer_z = prefer_z
        self.depth = depth
        self.lt_finished = False
        self.lt_success = False
        self.ec_finished = False
        self.ec_checks_to_do = []
        self.ec_checks_done = []
        self.status_dicts = {'spc': {}, 'x': {}, 'y': {}, 'z': {}, 'xy': {}}
        self.current_target = None
        self.strat = None
        self.target_pauli = None
        self.ec_pauli = None
        self.target_measured = False
        self.counter = {'xx':0, 'xzi':0, 'xf':0, 'zz':0, 'zzi':0, 'zf':0, 'yy':0, 'yzi':0, 'yf':0, 'aa':0, 'azi':0, 'af':0}  # Track the number of XX, XZ, Xf, ZZ, Zf measurements that occur
        self.expr_dicts = {'spc': {}, 'x': {}, 'y': {}, 'z': {}, 'xy': {}}
        self.successful_outcomes = {'spc': [], 'x': [], 'y': [], 'z': [], 'xy': []}

    def set_params(self, successes, basis, first_pass=False):
        """
        from the binary string successes, reset decoder parameters to repeat decoding a different way through the
        decision tree
        """
        if first_pass:
            self.lt_success = False
            self.success_pattern = []
            self.q_lost = []
            self.purgatory_q = None  # If there is a qubit that has been lost, but we haven't tried to measure it indirectly yet
            self.purg_q_basis = None
            self.pauli_done = Pauli([], [], self.nq, 0)
            self.arb_meas = []
            self.lt_finished = False
            self.counter = {'xx': 0, 'xzi': 0, 'xf': 0, 'zz': 0, 'zzi': 0, 'zf': 0, 'yy': 0, 'yzi': 0, 'yf': 0, 'aa': 0,
                            'azi': 0, 'af': 0}
            self.ec_checks_done = []
            self.ec_finished = False
            self.target_measured = False
        else:
            self.strategies_remaining, self.strat, self.current_target,  self.q_lost, self.arb_meas, self.pauli_done,\
            self.lt_finished, self.lt_success, self.target_measured, self.purgatory_q, self.purg_q_basis, self.counter, self.ec_checks_done, self.ec_finished = deepcopy(self.status_dicts[basis][tuple(successes)])

            self.success_pattern = successes.copy()
            self.target_pauli = self.strat.pauli

    def decode(self, starting_point=None, first_traversal=False, pathfinding=True, eff_meas_basis=None, mc=False,
               error_correcting=False, p=None, cascading=True):
        """
        :param first_traversal: All measurements succeed on the first traversal
        :param starting_point: If you are doing analytic tree search
        :param pathfinding:
        :param eff_meas_basis:
        :param mc: monte-carlo?
        :param error_correcting: Are we doing error detection?
        :param p: Physical transmission probability (if doing monte-carlo)
        """

        if pathfinding:
            decoder_type = 'spc'
        else:
            decoder_type = eff_meas_basis
        if starting_point is None:
            assert first_traversal and mc
        self.set_params(starting_point, decoder_type, first_pass=first_traversal)
        if not pathfinding:
            self.target_measured = True

        if first_traversal:
            if pathfinding:
                self.strategies_remaining = gen_strats_from_stabs(self.stab_grp_nt, self.nq, get_individuals=True)
            else:
                meas_basis_dict = {'x':0, 'y':1, 'z':2, 'xy':3}
                paulis = gen_logicals_from_stab_grp(self.stab_grp_nt, self.nq)[meas_basis_dict[eff_meas_basis]]
                self.strategies_remaining = [Strategy(p, None) for p in paulis]

            self.strat = self.strategy_picker()
            self.target_pauli = self.strat.pauli
            self.current_target = self.strat.t

        while not self.lt_finished:
            # self.print_status()
            if self.purgatory_q is not None:
                # Try to measure this before anything else
                meas_type = self.purg_q_basis
                good = self.next_meas_outcome(starting_point, mc=mc, p=p)
                if good:
                    self.success_pattern.append(1)
                    self.pauli_done.update_zs(self.purgatory_q, 1)
                    self.counter[f'{meas_type}zi'] += 1
                else:
                    self.success_pattern.append(0)
                    self.q_lost.append(self.purgatory_q)
                    self.counter[f'{meas_type}f'] += 1
                self.purgatory_q = None
                self.purg_q_basis = None
                self.strategies_remaining = self.update_available_strats()
                self.lt_finished, self.lt_success = self.decoder_finished(pf=pathfinding)
                if self.lt_finished:
                    self.cache_status(decoder_type)
                    break
                # self.print_status()

                self.strat = self.strategy_picker()
                self.target_pauli = self.strat.pauli
                self.current_target = self.strat.t
                self.target_measured = (self.current_target in self.arb_meas or not pathfinding)
                self.cache_status(decoder_type)
            if self.strat is not None:
                self.lt_finished, self.lt_success = self.decoder_finished(pf=pathfinding)

            # Select a target
            while not self.target_measured:
                if self.purgatory_q is not None:  # Always get the qubit out of purgatory first
                    good = self.next_meas_outcome(starting_point, mc=mc, p=p)
                    if good:
                        self.success_pattern.append(1)
                        self.pauli_done.update_zs(self.purgatory_q, 1)
                        self.counter['azi'] += 1
                    else:
                        self.success_pattern.append(0)
                        self.q_lost.append(self.purgatory_q)
                        self.counter['af'] += 1
                    self.purgatory_q = None
                    self.purg_q_basis = None

                self.strategies_remaining = self.update_available_strats()
                self.lt_finished, success = self.decoder_finished(pf=pathfinding)
                if self.lt_finished:
                    break
                # print(len(self.strategies_remaining))
                self.strat = self.strategy_picker()
                # print(self.strat.pauli.to_str())
                self.target_pauli = self.strat.pauli
                self.current_target = self.strat.t
                self.cache_status(decoder_type)
                if self.current_target in self.arb_meas:
                    print('Does this ever happen? Surely this would have been flagged already')
                    self.target_measured = True
                else:
                    # Does the next measurement succeed?
                    if self.next_meas_outcome(starting_point, first_traversal=first_traversal, mc=mc, p=p):
                        self.success_pattern.append(1)
                        self.arb_meas.append(self.current_target)
                        self.counter['aa'] += 1
                        self.target_measured = True
                    else:  # If it fails we have a purgatory qubit, unless we aren't doing cascading
                        self.success_pattern.append(0)
                        if not cascading:
                            self.q_lost.append(self.current_target)
                            self.counter['af'] += 1
                        else:
                            self.purgatory_q = self.current_target
                            self.purg_q_basis = 'a'
                    self.strategies_remaining = self.update_available_strats()
                self.lt_finished, self.lt_success = self.decoder_finished(pf=pathfinding)
                if self.strategies_remaining:
                    self.strat = self.strategy_picker()
                    self.target_pauli = self.strat.pauli
                    self.current_target = self.strat.t
                    self.target_measured = (self.current_target in self.arb_meas or not pathfinding)
                self.cache_status(decoder_type)

            self.lt_finished, self.lt_success = self.decoder_finished(pf=pathfinding)
            self.cache_status(decoder_type)
            if not self.lt_finished:
                # Now try to do the Pauli measurements
                q = self.next_qubit()

                meas_type = measurement_type(self.target_pauli, q)

                good = self.next_meas_outcome(starting_point, first_traversal=first_traversal, mc=mc, p=p)
                if good:
                    self.success_pattern.append(1)
                    self.pauli_done.update_zs(q, self.target_pauli.zs[q])
                    self.pauli_done.update_xs(q, self.target_pauli.xs[q])
                    self.counter[meas_type * 2] += 1

                    self.lt_finished, self.lt_success = self.decoder_finished(pf=pathfinding)
                else:
                    self.success_pattern.append(0)
                    if cascading:
                        self.purgatory_q = q
                        self.purg_q_basis = meas_type
                    else:
                        self.q_lost.append(q)
                        self.counter[f'{meas_type}f'] += 1
                self.strategies_remaining = self.update_available_strats()

                if self.strategies_remaining:
                    self.strat = self.strategy_picker()
                    self.target_pauli = self.strat.pauli
                    self.current_target = self.strat.t
                    self.target_measured = (self.current_target in self.arb_meas or not pathfinding)
                self.lt_finished, self.lt_success = self.decoder_finished(pf=pathfinding)

                self.cache_status(decoder_type)
        # When we arrive here we assume we have finished LT decoding. Don't do EC if you couldn't teleport
        if error_correcting and self.lt_success:
            # print('EC decoding...')
            self.ec_checks_to_do, self.ec_pauli = self.get_best_ec_strats()  # return a list of the check measurements we want to measure, and the overall pauli
            # self.print_status()
            if len(self.ec_checks_to_do) == 0 or set(self.ec_pauli.support).issubset(set(self.pauli_done.support)):
                self.ec_finished = True
            while not self.ec_finished:
                # print(self.success_pattern)
                q = self.next_qubit(ec=True)
                meas_type = measurement_type(self.ec_pauli, q)
                good = self.next_meas_outcome(starting_point, mc=mc, first_traversal=first_traversal, p=p)
                if good:
                    self.success_pattern.append(1)
                    self.pauli_done.update_zs(q, self.ec_pauli.zs[q])
                    self.pauli_done.update_xs(q, self.ec_pauli.xs[q])
                    self.counter[meas_type * 2] += 1
                    for check in self.ec_checks_to_do:
                        if q in check.support and set(check.support).issubset(set(self.pauli_done.support)):
                            self.ec_checks_to_do.remove(check)
                            self.ec_checks_done.append(check)
                            break
                else:
                    self.success_pattern.append(0)
                    self.q_lost.append(q)
                    self.counter[meas_type+'f'] += 1
                    self.ec_checks_to_do, self.ec_pauli = self.get_best_ec_strats()
                if not self.ec_checks_to_do:
                    self.ec_finished = True
                self.cache_status(decoder_type)
                if len(self.ec_checks_to_do) == 0 or set(self.ec_pauli.support).issubset(set(self.pauli_done.support)):
                    self.ec_finished = True
        return self.lt_success, self.success_pattern, self.pauli_done, self.arb_meas, self.q_lost, self.counter, self.ec_checks_done

    def next_measurement(self, success_bits, decoder_type, ec=True, cascading=False, print_status=False):
        """
        For a bit string of previously attempted measurements, return the next measurement
        For mc decoding with different error models (Non-local etc.)
        TODO finish this function
        TODO test this function
        :return next qubit to measure, measurement basis, lt decoder finished, lt decoder successful, ec finished, ec checks done
        """
        self.set_params(success_bits, decoder_type)
        if print_status:
            self.print_status()
        if cascading:
            raise NotImplementedError
        else:
            if not self.lt_finished:
                # While we are still doing the loss tolerant part
                # self.strat = self.strategy_picker()
                if not self.target_measured:
                    meas_type = 'a'
                    q = self.strat.t
                else:
                    q = self.next_qubit(ec=self.lt_finished)
                    meas_type = measurement_type(self.strat.pauli, q)
                checks_done = None
                ec_finished = None
            else:
                if not ec:
                    q, meas_type = None, None
                    checks_done, ec_finished = None, None
                else:
                    ec_finished = self.ec_finished
                    if ec_finished:
                        q, meas_type = None, None
                    else:
                        self.ec_checks_to_do, self.ec_pauli = self.get_best_ec_strats()
                        # print([c.to_str() for c in self.ec_checks_to_do])
                        q = self.next_qubit(ec=True)
                        meas_type = measurement_type(self.ec_pauli, q)
                    checks_done = self.ec_checks_done
            return q, meas_type, self.lt_finished, self.lt_success, checks_done, ec_finished

    def decoder_finished(self, pf=True):
        if self.pauli_done is not None and self.target_pauli is not None:
            if self.pauli_done.contains_other(self.target_pauli) and (self.current_target in self.arb_meas or not pf):
                return True, True
            elif not self.strategies_remaining:
                return True, False
            else:
                return False, None
        else:
            return False, None

    def next_qubit(self, ec=False):
        if ec:
            ec_pauli = multi_union(self.ec_checks_to_do)
            return list(set(ec_pauli.support) - set(self.pauli_done.support))[0]
        else:
            return list(set(self.target_pauli.support) - set(self.pauli_done.support))[0]

    def to_measure(self):

        outstanding = list(set(self.target_pauli.support) - set(self.pauli_done.support))
        return outstanding

    def next_meas_outcome(self, starting_point, first_traversal=False, mc=False, p=None):
        """
        Assume we are doing the analytic decision tree building technique - so the outcome is determined by
        which parts of the tree we have already searched
        Always go fail, followed by all successes - this constitutes a depth-first traversal of the tree
        If we are doing shot-by-shot decoding (how you would run an experiment) we need to return a bit determined
        by the transmission probability of the channel
        """
        if mc:
            r = random.random()
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

    def cache_status(self, decoder_type):
        """
        TODO fix the bug in thsi function - the cached results in the dictionary seem to change based on later outcomes?
        """
        pattern = tuple(np.copy(self.success_pattern))
        if pattern not in self.status_dicts[decoder_type].keys():
            s = self.strat.copy()
            self.status_dicts[decoder_type][pattern] = (deepcopy(self.strategies_remaining), s, self.current_target, deepcopy(self.q_lost), deepcopy(self.arb_meas), self.pauli_done.copy(),
                                                  self.lt_finished, self.lt_success, self.target_measured, self.purgatory_q, self.purg_q_basis, self.counter.copy(), self.ec_checks_done.copy(), self.ec_finished)
            # self.print_status()

    def build_tree(self, basis='spc', ec=False, printing=False, cascading=True):
        """
        TODO add a cascaded=False option to speed up decoding where you don't care about the indirect measurements
        """
        pathfinding = basis == 'spc'
        self.status_dicts[basis] = {}
        success_count = 0
        first_pattern = [1] * self.nq
        first_pass = True
        out = first_pattern
        self.successful_outcomes[basis] = []
        while 1 in out:
            while out[-1] == 0:  # Trim trailing zeros from prefix, i.e. go up to the lowest point in the tree where there was a successful measurement
                del out[-1]
            success, success_pattern, pauli_done, arb_meas, q_lost, counter, checks_done = self.decode(starting_point=out[:-1], first_traversal=first_pass, pathfinding=pathfinding, eff_meas_basis=basis, error_correcting=ec, cascading=cascading)
            if success:
                success_count += 1
                if pathfinding:
                    if printing:
                        print(f'{success_pattern=}')
                        print(f'{self.strat.pauli.to_str()}')
                        print(f'{self.strat.s1.to_str()=}')
                        print(f'{self.strat.s2.to_str()=}')
                        print([c.to_str() for c in self.ec_checks_done])
                        print('\n')
                    self.successful_outcomes[basis].append(DecoderOutcome(counter, self.ec_checks_done, [self.strat.s1, self.strat.s2], self.strat.t))
                else:
                    if printing:
                        print(f'{success_pattern=}')
                        print(f'{self.strat.pauli.to_str()}')
                        print(f'{success=}')
                        print('\n')

                    self.successful_outcomes[basis].append(DecoderOutcome(counter, self.ec_checks_done, [self.strat.pauli], None))
            first_pass = False
            out = success_pattern

    def update_available_strats(self, lost=None, arb_meas=None, pauli_meas=None, strats=None):
        """
        return the sub-list of available strategies after losing/measuring the additional qubits
        :param strats: if None, use the decoder strategy list, otherwise use this
        """
        if lost is not None:
            q_lost = self.q_lost + lost
        else:
            q_lost = self.q_lost
        if arb_meas is not None:
            arb = self.arb_meas + arb_meas
        else:
            arb = self.arb_meas
        if pauli_meas is not None:
            p = self.pauli_done.mult(pauli_meas)
        else:
            p = self.pauli_done
        if strats is None:
            strats = self.strategies_remaining

        return [s for s in strats if (len(set(q_lost) & set(s.pauli.support)) == 0 and len(
            set(arb) & set(s.pauli.support)) == 0 and s.t not in q_lost and is_compatible(s.pauli, p, no_supersets=False)
            and s.t not in set(p.support))]

    def strategy_picker(self):

        """
        We want a combination of the measurement with the smallest support and the largest number of compatible measurements
        want to ensure that as many qubits as possible in the pattern can be lost, and we can still switch to something else
        also want the patterns you switch to to satisfy these criteria - can do this to varying depth for more/less accurate
        decoding
        TODO Calculate a score for each potential next measurement pattern to see how good it would be - do this to
        TODO varying depths to have faster/slower and less/more accurate decoding
        TODO for each potential measurement
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

                    for q in s.to_measure(self.q_lost, self.pauli_done, None):  # For each qubit that could be lost, are we tolerant?
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
                        nu_winner=True
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

    def get_best_ec_strats(self):
        from error_correction import best_checks
        banned_q = [0, self.current_target]
        banned_q += self.q_lost
        checks = best_checks(self.graph, self.pauli_done, banned_qs=banned_q, stab_nt=self.stab_grp_nt)  # Needs to be compatible with the pauli measurements done, not just the target (may be extras)
        if checks:
            overall_pauli = multi_union(checks)
            return checks, overall_pauli
        else:
            return checks, None

    def print_status(self):
        print(f'Status:\n    Lost qubits: {self.q_lost}\n    Qubits measured in arb basis: {self.arb_meas}\n    '
              f'Pauli measurements performed: {self.pauli_done.to_str()}\n{self.success_pattern=}\n{self.lt_finished=}\n{self.target_pauli.to_str()=}')
        if self.ec_pauli is not None:
            print(f'{self.ec_finished=}\n{self.ec_pauli.to_str()}\n')
        return

    def get_dict(self, basis='spc', ec=False, rebuild_tree=False, cascading=True):
        self.expr_dicts[basis] = {}
        if len(self.successful_outcomes[basis]) == 0 or rebuild_tree:
            self.successful_outcomes[basis] = []
            self.build_tree(basis=basis, ec=ec, cascading=cascading)
        for r in self.successful_outcomes[basis]:
            k = tuple([v for v in r.counter.values()])
            # print(k)
            if k in self.expr_dicts[basis].keys():
                self.expr_dicts[basis][k] += 1
            else:
                self.expr_dicts[basis][k] = 1
        return self.expr_dicts[basis]

    def success_prob(self, xx, xzi, xf, zz, zzi, zf, yy, yzi, yf, aa, azi, af, basis='spc'):
        return sum([self.expr_dicts[basis][k] * xx**k[0] * xzi ** k[1] * xf ** k[2] * zz ** k[3] * zzi ** k[4] * zf **k[5]
                        * yy ** k[6] * yzi ** k[7] * yf ** k[8] * aa ** k[9] * azi ** k[10] * af ** k[11] for k in self.expr_dicts[basis].keys()])

    def success_prob_outcome_list(self, transmission, depolarizing_noise, basis='spc', ec=True):
        """
        Use the outcome list to find the probability of success and the probability of a pauli error
        """
        tot_prob = 0
        tot_pauli_success = 0
        for item in self.successful_outcomes[basis]:
            # print(item.target_paulis[0].to_str())
            # print([c.to_str() for c in item.checks])
            prob = item.outcome_prob(transmission)
            tot_prob += prob
            if ec:
                tot_pauli_success += prob * item.no_flip_prob([depolarizing_noise/4])
        if ec:
            tot_pauli_success_given_teleportation = tot_pauli_success / tot_prob
            # print(tot_prob, tot_pauli_success, tot_pauli_success_given_teleportation)
            return tot_prob, tot_pauli_success_given_teleportation
        return tot_prob


def measurement_type(pauli, ix):
    if pauli.xs[ix]:
        if pauli.zs[ix]:
            return 'y'
        else:
            return 'x'
    elif pauli.zs[ix]:
        return 'z'


def main():
    pass
    # from CodesFunctions.graphs import gen_linear_graph, gen_ring_graph, gen_tree_graph
    # import matplotlib.pyplot as plt
    # from graphs import draw_graph
    # bases = ['spc', 'x', 'y', 'z', 'xy']
    # b = 'spc'
    # ring5_2l = gen_ring_graph(5)
    # ring5_2l.add_edges_from([(1, 5), (3, 6)])
    # draw_graph(ring5_2l)
    # r5_decoder = CascadeDecoder(ring5_2l)
    # t0 = time()
    # print(t0)
    # r5_decoder.build_tree(basis=b, ec=True, cascading=True)
    # t1 = time()
    # print(t1-t0)
    # # print('Built')
    # r5_decoder.get_dict(b)
    # # for b in ('x', 'y', 'z', 'xy'):
    # #     r5_decoder.get_dict(basis=b)
    # print('HERE')
    #
    # etas = np.linspace(0.01, 0.99)
    # out = []
    # out_new = []
    # zout = []
    # for eta in etas:
    #     # out_new.append(r5_decoder.success_prob_outcome_list(eta, 0.2, basis=b))
    #     xx = yy = zz = aa = eta
    #     xzi = yzi = azi = zzi = 0
    #     xf = yf = zf = af = 1-eta
    #
    #     out.append(r5_decoder.success_prob(xx, xzi, xf, zz, zzi, zf, yy, yzi, yf, aa, azi, af, basis=b))
    #
    # plt.plot(etas, out)
    # # plt.plot(etas, out_new)
    # plt.plot((0, 1), (0, 1), 'k--')
    # # m = min([x[1] for x in out_new])
    # # plt.plot((0, 1), (m, m), 'k--')
    # plt.show()
    # # Concatenation example
    # from cascaded import AnalyticCascadedResult
    # dicts = []
    # for basis in bases:
    #     print(f'{basis=}')
    #     dicts.append(r5_decoder.get_dict(basis, ec=False, rebuild_tree=True, cascading=True))
    # x = AnalyticCascadedResult([dicts], [0, 0, 0, 0, 0])
    # for b in bases:
    #     y = [r5_decoder.success_prob_outcome_list(t, 0, basis=b, ec=True)[0] for t in etas]
    #     if b =='xy':
    #         flag = '+'
    #     else:
    #         flag = '-'
    #     plt.plot(etas, y, flag)
    # plt.plot((0, 1), (0, 1), 'k--')
    # plt.show()
    #
    # for depth in range(1, 6):
    #     y = [x.get_spc_prob(eta, depth) for eta in etas]
    #     plt.plot(etas, y)
    # plt.plot((0, 1), (0, 1), 'k--')
    # plt.show()


if __name__ == '__main__':
    main()



