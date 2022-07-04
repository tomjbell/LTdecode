import networkx as nx
import numpy as np
from pauli_class import Pauli
from stab_formalism import stabilizers_from_graph, gen_stabs_from_generators, gen_logicals_from_stab_grp
from itertools import combinations, product
from graphs import gen_ring_graph, draw_graph
import matplotlib.pyplot as plt
from helpers import bisection_search


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
    test_new_decoding()
    exit()


if __name__=='__main__':
    main()
